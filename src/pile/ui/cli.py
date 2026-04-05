"""CLI interface for Pile — interactive terminal chat."""

from __future__ import annotations

import asyncio
import logging
import sys

from pile.workflows.interactive import create_workflow


def _create_slash_workflow(command: str):
    """Create a one-shot workflow for slash commands. Returns (workflow, prompt) or None."""
    if command == "/standup":
        from pile.workflows.standup import create_workflow as create_standup
        return create_standup(), "Generate today's standup report."
    if command == "/planning":
        from pile.workflows.planning import create_workflow as create_planning
        return create_planning(), "Let's start sprint planning. Review the backlog and current sprint status."
    return None


async def _stream_workflow(workflow, message, pending_requests):
    """Run a workflow and stream output, collecting pending requests."""
    async for event in workflow.run(message, stream=True):
        if event.type == "output":
            if hasattr(event.data, "text") and event.data.text:
                print(event.data.text, end="", flush=True)
            elif isinstance(event.data, list):
                for msg in event.data:
                    if hasattr(msg, "text") and msg.text:
                        name = getattr(msg, "author_name", None) or msg.role
                        print(f"\n[{name}]\n{msg.text}")
        elif event.type == "request_info":
            pending_requests.append(event)
    print()


async def _stream_responses(workflow, responses, pending_requests):
    """Run a workflow with responses and stream output, collecting pending requests."""
    async for event in workflow.run(responses=responses, stream=True):
        if event.type == "output" and hasattr(event.data, "text") and event.data.text:
            print(event.data.text, end="", flush=True)
        elif event.type == "request_info":
            pending_requests.append(event)
    print()


async def _handle_pending_requests(workflow, pending_requests):
    """Process pending approval/follow-up requests iteratively."""
    from agent_framework.orchestrations import HandoffAgentUserRequest

    while pending_requests:
        new_pending = []
        for req in pending_requests:
            if isinstance(req.data, HandoffAgentUserRequest):
                for msg in req.data.agent_response.messages[-2:]:
                    if msg.text:
                        print(f"[{msg.author_name}]: {msg.text}")

                try:
                    follow_up = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye!")
                    return False

                if follow_up.lower() in ("quit", "exit", "q"):
                    print("Bye!")
                    return False

                responses = {req.request_id: HandoffAgentUserRequest.create_response(follow_up)}
                await _stream_responses(workflow, responses, new_pending)

            elif hasattr(req.data, "type") and req.data.type == "function_approval_request":
                func_call = req.data.function_call
                args = func_call.parse_arguments() if hasattr(func_call, "parse_arguments") else {}
                print("\n--- Approval Required ---")
                print(f"Tool: {func_call.name}")
                print(f"Args: {args}")

                try:
                    approval = input("Approve? (y/n): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye!")
                    return False

                approved = approval in ("y", "yes")
                responses = {req.request_id: req.data.to_function_approval_response(approved=approved)}
                await _stream_responses(workflow, responses, new_pending)

        pending_requests[:] = new_pending

    return True


async def _run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Health checks
    from pile.health import run_health_checks
    errors = run_health_checks()
    if errors:
        print("Health check warnings:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print(file=sys.stderr)

    print("Pile SM ready. Type 'quit' to exit.")
    print("Slash commands: /standup, /planning\n")

    try:
        workflow, _tracker = create_workflow()
    except Exception as e:
        print(f"Failed to initialize: {e}", file=sys.stderr)
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if not user_input:
            continue

        # Handle slash commands
        slash = _create_slash_workflow(user_input)
        if slash:
            slash_workflow, slash_prompt = slash
            try:
                pending_requests: list = []
                await _stream_workflow(slash_workflow, slash_prompt, pending_requests)
                if not await _handle_pending_requests(slash_workflow, pending_requests):
                    return
            except Exception as e:
                print(f"Workflow error: {e}", file=sys.stderr)
            continue

        # Run routed workflow
        try:
            async for event in workflow.run(user_input, stream=True):
                if event.type == "output" and hasattr(event.data, "text") and event.data.text:
                    print(event.data.text, end="", flush=True)
        except KeyboardInterrupt:
            workflow._reset_running_flag()
            print("\n[Stopped]")
            continue

        print()  # newline after streaming


def main():
    """Entry point for the CLI."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
