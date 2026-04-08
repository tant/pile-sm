"""Chainlit Web UI for Pile — multi-agent visualization with steps, starters, and profiles."""

from __future__ import annotations

import logging

import chainlit as cl

from pile.models.logging import setup_app_logger, setup_inference_logger
from pile.models.manager import ensure_models
from pile.workflows.interactive import create_workflow

logger = logging.getLogger("pile.ui")


def summarize_args(args: dict, max_keys: int = 3, max_val_len: int = 30) -> str:
    """Summarize tool arguments for display: 'key=val, key=val, +N more'."""
    if not args:
        return ""
    items = list(args.items())[:max_keys]
    parts = []
    for k, v in items:
        v_str = str(v)
        if len(v_str) > max_val_len:
            v_str = v_str[:max_val_len] + "..."
        parts.append(f"{k}={v_str}")
    summary = ", ".join(parts)
    if len(args) > max_keys:
        summary += f", +{len(args) - max_keys} more"
    return summary


# Agent display config
AGENT_CONFIG = {
    "TriageAgent": {"type": "run", "label": "Routing"},
    "JiraQueryAgent": {"type": "tool", "label": "Jira Query"},
    "JiraWriteAgent": {"type": "tool", "label": "Jira Write"},
    "BoardAgent": {"type": "tool", "label": "Board"},
    "SprintAgent": {"type": "tool", "label": "Sprint"},
    "EpicAgent": {"type": "tool", "label": "Epic"},
    "GitAgent": {"type": "tool", "label": "Git"},
    "ScrumAgent": {"type": "llm", "label": "Scrum Master"},
}


@cl.set_starters
async def set_starters():
    """Quick action buttons on the welcome screen."""
    return [
        cl.Starter(
            label="Sprint Status",
            message="Sprint hiện tại tiến độ thế nào?",
            icon="/public/icons/sprint.svg",
        ),
        cl.Starter(
            label="Standup",
            message="Tổng hợp standup cho team hôm nay",
            icon="/public/icons/standup.svg",
        ),
        cl.Starter(
            label="Workload",
            message="Ai đang bị quá tải? Phân tích workload team",
            icon="/public/icons/workload.svg",
        ),
        cl.Starter(
            label="Blockers",
            message="Có gì đang bị block không?",
            icon="/public/icons/block.svg",
        ),
        cl.Starter(
            label="Backlog",
            message="Backlog có gì? Gợi ý ưu tiên",
            icon="/public/icons/backlog.svg",
        ),
        cl.Starter(
            label="Data Quality",
            message="Kiểm tra Jira có gì thiếu thông tin không?",
            icon="/public/icons/quality.svg",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Initialize workflow with health checks."""
    setup_app_logger()
    setup_inference_logger()
    ensure_models()  # Downloads missing models on first run, then loads all

    # Setup debug logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("pile").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Health checks
    from pile.health import run_health_checks
    errors = run_health_checks()
    if errors:
        warning = "**Health check warnings:**\n" + "\n".join(f"- {e}" for e in errors)
        await cl.Message(content=warning).send()

    # Clean up expired session facts
    try:
        from pile.memory.store import cleanup_expired_facts
        removed = cleanup_expired_facts(max_age_days=7)
        if removed:
            logger.info("Cleaned up %d expired session facts", removed)
    except Exception:
        pass

    try:
        workflow, tracker = create_workflow()
    except Exception as e:
        await cl.Message(content=f"Failed to initialize: {e}").send()
        return

    cl.user_session.set("workflow", workflow)
    cl.user_session.set("tracker", tracker)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages, including file uploads."""
    workflow = cl.user_session.get("workflow")
    if workflow is None:
        await cl.Message(content="Workflow not initialized. Please refresh.").send()
        return

    # Handle file uploads — ingest into knowledge base
    ingested = await _handle_file_uploads(message)

    # Build user input
    user_input = message.content or ""
    if ingested:
        files_info = ", ".join(f"'{name}'" for name in ingested)
        if not user_input:
            user_input = f"I just uploaded these documents to the knowledge base: {files_info}. Confirm what was loaded."
        else:
            user_input = f"[Uploaded documents: {files_info}]\n\n{user_input}"
    if user_input:
        await _run_workflow(workflow, user_input=user_input)


async def _handle_file_uploads(message: cl.Message) -> list[str]:
    """Process file attachments — ingest each into knowledge base. Returns list of ingested filenames."""
    if not message.elements:
        return []

    from pile.config import settings
    if not settings.memory_enabled:
        await cl.Message(content="Memory is disabled. Cannot ingest documents.").send()
        return []

    from pile.memory.ingest import ingest_file

    ingested: list[str] = []
    for element in message.elements:
        if not element.path:
            continue
        try:
            result = ingest_file(element.path)
            ingested.append(result["doc_name"])
            await cl.Message(
                content=(
                    f"Ingested **{result['doc_name']}**: "
                    f"{result['pages']} pages, {result['chunks']} chunks stored."
                ),
            ).send()
        except Exception as e:
            await cl.Message(content=f"Failed to ingest {element.name}: {e}").send()

    return ingested


async def _run_workflow_once(workflow, *, user_input: str | None = None, responses: dict | None = None):
    """Stream a single workflow run, rendering steps in real-time.

    Layout: one root "Thinking" step (collapsed) containing all agent/tool
    sub-steps, then the final answer as a separate message below.
    """
    import asyncio

    # One root step that contains everything — collapsed by default
    root_step = cl.Step(name="Thinking", type="tool")
    root_step.output = ""
    await root_step.send()

    current_agent_step: cl.Step | None = None
    active_tool_steps: dict[str, list[cl.Step]] = {}
    final_text = ""
    agents_used: list[str] = []

    if user_input is not None:
        run_iter = workflow.run(user_input, stream=True, include_status_events=True)
    else:
        run_iter = workflow.run(responses=responses, stream=True, include_status_events=True)

    try:
        async for event in run_iter:
            executor_id = getattr(event, "executor_id", None)

            if event.type == "executor_invoked" and executor_id:
                if current_agent_step:
                    await current_agent_step.update()
                    current_agent_step = None

                cfg = AGENT_CONFIG.get(executor_id, {"type": "run", "label": executor_id})
                agents_used.append(cfg["label"])
                current_agent_step = cl.Step(
                    name=cfg["label"], type="tool",
                    parent_id=root_step.id,
                )
                current_agent_step.output = ""
                await current_agent_step.send()

            elif event.type == "data" and isinstance(event.data, dict):
                tool_data = event.data

                if tool_data.get("type") == "tool_start" and current_agent_step:
                    tool_name = tool_data["name"]
                    tool_args = tool_data.get("args", {})
                    tool_step = cl.Step(
                        name=f"{tool_name} ({summarize_args(tool_args)})",
                        type="tool",
                        parent_id=current_agent_step.id,
                    )
                    tool_step.input = str(tool_args) if tool_args else ""
                    tool_step.output = ""
                    await tool_step.send()
                    active_tool_steps.setdefault(tool_name, []).append(tool_step)

                elif tool_data.get("type") == "tool_end":
                    record = tool_data["record"]
                    tool_name = record.name
                    steps = active_tool_steps.get(tool_name, [])
                    tool_step = steps.pop(0) if steps else None
                    if not steps:
                        active_tool_steps.pop(tool_name, None)
                    if tool_step:
                        duration = f"{record.duration_ms / 1000:.1f}s"
                        tool_step.output = record.result or ""
                        tool_step.name = f"{tool_name} — {duration}"
                        await tool_step.update()

                elif tool_data.get("type") == "recalled_context":
                    facts = tool_data.get("facts", [])
                    if facts:
                        recall_step = cl.Step(
                            name="Recalled from previous sessions", type="tool",
                            parent_id=root_step.id,
                        )
                        recall_step.output = "\n".join(f"- {f}" for f in facts)
                        await recall_step.send()
                        await recall_step.update()

            elif event.type == "output":
                text = getattr(event.data, "text", None) if not isinstance(event.data, list) else None
                if text:
                    final_text += text
                    if current_agent_step:
                        current_agent_step.output += text

            elif event.type == "executor_completed":
                for steps in active_tool_steps.values():
                    for tool_step in steps:
                        await tool_step.update()
                active_tool_steps.clear()

                if current_agent_step:
                    await current_agent_step.update()
                    current_agent_step = None

            elif event.type == "executor_failed":
                if current_agent_step:
                    error_msg = str(event.data) if event.data else "Unknown error"
                    current_agent_step.output += f"\n\n*Error: {error_msg}*"
                    await current_agent_step.update()
                    current_agent_step = None

    except asyncio.CancelledError:
        workflow._reset_running_flag()
        for steps in active_tool_steps.values():
            for tool_step in steps:
                tool_step.output += "\n*Cancelled*"
                await tool_step.update()
        active_tool_steps.clear()
        if current_agent_step:
            if current_agent_step.output:
                current_agent_step.output += "\n\n*Stopped by user*"
                await current_agent_step.update()
            else:
                await current_agent_step.remove()
        root_step.name = "Stopped"
        await root_step.update()
        await cl.Message(content=final_text or "*Stopped.*", author="Pile SM").send()
        return []
    except Exception as e:
        workflow._reset_running_flag()
        if current_agent_step:
            current_agent_step.output += f"\n\n*Error: {e}*"
            await current_agent_step.update()
        root_step.name = "Error"
        await root_step.update()
        await cl.Message(content=final_text or f"*Error: {e}*", author="Pile SM").send()
        raise

    if current_agent_step:
        await current_agent_step.update()

    # Update root step summary
    summary = ", ".join(agents_used) if agents_used else "Done"
    root_step.name = summary
    await root_step.update()

    # Send final answer AFTER root step — appears at the bottom
    if final_text:
        msg = cl.Message(content=final_text, author="Pile SM")
        await msg.send()
        await _send_charts_if_any(msg)

    # Background: extract and persist key facts from this turn
    if user_input and final_text and len(final_text) >= 50:
        from pile.context import summarize_turn

        _summarize_text = final_text  # capture for closure

        async def _bg_summarize():
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    summarize_turn,
                    user_input,
                    _summarize_text,
                )
            except Exception:
                pass  # Never break the response flow

        asyncio.create_task(_bg_summarize())

    return []


async def _send_charts_if_any(msg: cl.Message):
    """Detect chartable data in message content and send Plotly charts."""
    if not msg.content or len(msg.content) < 20:
        return
    try:
        from pile.ui.charts import detect_charts, build_chart

        charts = detect_charts(msg.content)
        for chart_data in charts:
            fig = build_chart(chart_data)
            element = cl.Plotly(name=chart_data.title, figure=fig, size="large")
            await element.send(for_id=msg.id)
    except Exception:
        pass  # Don't break message flow if chart rendering fails


async def _run_workflow(workflow, *, user_input: str | None = None):
    """Stream workflow output."""
    await _run_workflow_once(workflow, user_input=user_input)


@cl.on_stop
async def on_stop():
    """Handle user pressing Stop button."""
    workflow = cl.user_session.get("workflow")
    if workflow is not None:
        workflow._reset_running_flag()


@cl.on_chat_end
async def on_chat_end():
    """Clean up when chat session ends."""
    cl.user_session.set("workflow", None)


def main():
    """Entry point hint."""
    print("Run with: chainlit run src/pile/ui/chainlit_app.py")


if __name__ == "__main__":
    main()
