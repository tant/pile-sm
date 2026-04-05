"""Chainlit Web UI for Pile — multi-agent visualization with steps, starters, and profiles."""

from __future__ import annotations

import json

import chainlit as cl

from pile.workflows.interactive import create_workflow

# Agent display config
AGENT_CONFIG = {
    "TriageAgent": {"type": "run", "label": "Routing"},
    "JiraAgent": {"type": "tool", "label": "Jira"},
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
            label="Blockers",
            message="Có gì đang bị block không?",
            icon="/public/icons/block.svg",
        ),
        cl.Starter(
            label="Data Quality",
            message="Kiểm tra Jira có gì thiếu thông tin không?",
            icon="/public/icons/quality.svg",
        ),
        cl.Starter(
            label="Knowledge Base",
            message="Có những tài liệu nào trong knowledge base?",
            icon="/public/icons/knowledge.svg",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Initialize workflow with health checks."""
    # Health checks
    from pile.health import run_health_checks
    errors = run_health_checks()
    if errors:
        warning = "**Health check warnings:**\n" + "\n".join(f"- {e}" for e in errors)
        await cl.Message(content=warning).send()

    try:
        workflow = create_workflow()
    except Exception as e:
        await cl.Message(content=f"Failed to initialize: {e}").send()
        return

    cl.user_session.set("workflow", workflow)
    cl.user_session.set("pending_handoff_request", None)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages, including file uploads."""
    workflow = cl.user_session.get("workflow")
    if workflow is None:
        await cl.Message(content="Workflow not initialized. Please refresh.").send()
        return

    # Handle file uploads — ingest into knowledge base
    ingested = await _handle_file_uploads(message)

    pending = cl.user_session.get("pending_handoff_request")
    if pending is not None:
        from agent_framework.orchestrations import HandoffAgentUserRequest

        cl.user_session.set("pending_handoff_request", None)
        responses = {pending.request_id: HandoffAgentUserRequest.create_response(message.content)}
        await _run_workflow(workflow, responses=responses)
    else:
        # If files were uploaded, prepend ingestion context to the message
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
    """Stream a single workflow run, returning pending requests."""
    msg = cl.Message(content="", author="Pile SM")
    await msg.send()

    pending_requests: list = []
    current_step: cl.Step | None = None
    current_executor: str | None = None

    if user_input is not None:
        run_iter = workflow.run(user_input, stream=True, include_status_events=True)
    else:
        run_iter = workflow.run(responses=responses, stream=True, include_status_events=True)

    async for event in run_iter:
        executor_id = getattr(event, "executor_id", None)

        if event.type == "executor_invoked" and executor_id:
            if current_step:
                current_step.output = current_step.output or "Done"
                await current_step.update()

            current_executor = executor_id
            cfg = AGENT_CONFIG.get(executor_id, {"type": "run", "label": executor_id})
            current_step = cl.Step(name=cfg["label"], type=cfg["type"])
            current_step.output = ""

            if isinstance(event.data, str) and event.data:
                current_step.input = event.data

            await current_step.send()

        elif event.type == "output":
            text = getattr(event.data, "text", None) if not isinstance(event.data, list) else None
            if text:
                if current_step and executor_id == current_executor:
                    current_step.output += text
                await msg.stream_token(text)

        elif event.type == "handoff_sent":
            if current_step:
                current_step.output += "\n\n*Handoff to next agent*"
                await current_step.update()

        elif event.type == "executor_completed":
            if current_step and executor_id == current_executor:
                await current_step.update()
                current_step = None
                current_executor = None

        elif event.type == "request_info":
            pending_requests.append(event)

    if current_step:
        await current_step.update()

    await msg.update()

    # Auto-detect numeric data and render charts
    await _send_charts_if_any(msg)

    return pending_requests


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


async def _run_workflow(workflow, *, user_input: str | None = None, responses: dict | None = None):
    """Stream workflow output with iterative approval handling."""
    from agent_framework.orchestrations import HandoffAgentUserRequest

    max_rounds = 20
    pending_requests = await _run_workflow_once(workflow, user_input=user_input, responses=responses)

    for _ in range(max_rounds):
        if not pending_requests:
            break

        next_pending: list = []
        for req in pending_requests:
            if isinstance(req.data, HandoffAgentUserRequest):
                parts = []
                for m in req.data.agent_response.messages[-2:]:
                    if m.text:
                        parts.append(m.text)
                if parts:
                    await cl.Message(content="\n\n".join(parts)).send()
                cl.user_session.set("pending_handoff_request", req)

            elif hasattr(req.data, "type") and req.data.type == "function_approval_request":
                func_call = req.data.function_call
                args = func_call.parse_arguments() if hasattr(func_call, "parse_arguments") else {}
                args_str = json.dumps(args, ensure_ascii=False, indent=2) if isinstance(args, dict) else str(args)

                action_res = await cl.AskActionMessage(
                    content=(
                        f"**Approval Required**\n\n"
                        f"Tool: `{func_call.name}`\n\n"
                        f"```json\n{args_str}\n```"
                    ),
                    actions=[
                        cl.Action(name="approve", payload={"approved": True}, label="Approve"),
                        cl.Action(name="reject", payload={"approved": False}, label="Reject"),
                    ],
                ).send()

                if action_res is not None:
                    approved = action_res.get("name") == "approve" if isinstance(action_res, dict) else (action_res.name == "approve")
                    resp = {req.request_id: req.data.to_function_approval_response(approved=approved)}
                    next_pending.extend(await _run_workflow_once(workflow, responses=resp))

        pending_requests = next_pending
    else:
        if pending_requests:
            await cl.Message(content="Too many approval rounds. Please try again.").send()


def main():
    """Entry point hint."""
    print("Run with: chainlit run src/pile/ui/chainlit_app.py")


if __name__ == "__main__":
    main()
