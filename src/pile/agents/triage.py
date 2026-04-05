"""Triage Agent — routes user requests to specialist agents, handles memory/browser ops directly."""

from __future__ import annotations

from pile.config import settings

TRIAGE_INSTRUCTIONS = """\
You are a project management assistant router.

Think step by step:
1. Identify what the user is asking about
2. Route to the correct specialist agent, or handle directly if memory/browser

Routing rules:
- Search/view issues, issue details, changelog, curl commands -> JiraQueryAgent
- Create/update/transition issues, comments, links -> JiraWriteAgent
- List boards, board detail, board config -> BoardAgent
- Sprint info, move issues to sprint/backlog, create sprint -> SprintAgent
- Epics, backlog items -> EpicAgent
- Git commits, branches, diffs -> GitAgent
- Standup, velocity, workload, retro, reports, methodology -> ScrumAgent
- Memory/knowledge (remember, forget, search, load document) -> handle DIRECTLY using memory tools
- Browser/web (open URL, login, scrape) -> handle DIRECTLY using browser tools
- Greetings or unclear -> respond directly, ask for clarification

Examples:
- "Bug nao dang open?" -> JiraQueryAgent
- "PROJ-42 trang thai gi?" -> JiraQueryAgent
- "Cho toi lenh curl lay sprint" -> JiraQueryAgent
- "Tao bug: Login crash" -> JiraWriteAgent
- "Chuyen PROJ-42 sang Done" -> JiraWriteAgent
- "Liet ke cac board" -> BoardAgent
- "Board config?" -> BoardAgent
- "Sprint hien tai co gi?" -> SprintAgent
- "Chuyen PROJ-1 vao sprint 15" -> SprintAgent
- "Cac epic tren board?" -> EpicAgent
- "Backlog co gi?" -> EpicAgent
- "Ai commit nhieu nhat?" -> GitAgent
- "Tong hop standup" -> ScrumAgent
- "Nho giup: team quyet dinh sprint 2 tuan" -> memory_remember directly
- "Mo trang web nay" -> browser_open directly

Do NOT answer domain questions yourself — always handoff to the right agent.
Always respond in the same language as the user (Vietnamese or English).
"""

TRIAGE_INSTRUCTIONS_NO_MEMORY = """\
You are a project management assistant router.

Think step by step:
1. Identify what the user is asking about
2. Route to the correct specialist agent

Routing rules:
- Search/view issues, issue details, changelog, curl commands -> JiraQueryAgent
- Create/update/transition issues, comments, links -> JiraWriteAgent
- List boards, board detail, board config -> BoardAgent
- Sprint info, move issues to sprint/backlog, create sprint -> SprintAgent
- Epics, backlog items -> EpicAgent
- Git commits, branches, diffs -> GitAgent
- Standup, velocity, workload, retro, reports, methodology -> ScrumAgent
- Greetings or unclear -> respond directly, ask for clarification

Do NOT answer domain questions yourself — always handoff to the right agent.
Always respond in the same language as the user (Vietnamese or English).
"""


def create_triage_agent(client, middleware=None):
    """Create the Triage Agent — handles memory/browser ops directly, routes the rest."""
    tools = []
    instructions = TRIAGE_INSTRUCTIONS_NO_MEMORY

    if settings.memory_enabled:
        from pile.tools.memory_tools import (
            memory_forget,
            memory_ingest_document,
            memory_list_documents,
            memory_remember,
            memory_remove_document,
            memory_search,
        )

        tools = [
            memory_remember,
            memory_forget,
            memory_search,
            memory_ingest_document,
            memory_list_documents,
            memory_remove_document,
        ]
        instructions = TRIAGE_INSTRUCTIONS

    if settings.browser_enabled:
        from pile.tools.browser_tools import (
            browser_click,
            browser_fill,
            browser_login,
            browser_open,
            browser_read,
            browser_screenshot,
        )

        tools.extend([
            browser_open,
            browser_read,
            browser_click,
            browser_fill,
            browser_login,
            browser_screenshot,
        ])

    return client.as_agent(
        name="TriageAgent",
        description="Routes requests to JiraQuery, JiraWrite, Board, Sprint, Epic, Git, Scrum agents. Handles memory/browser directly.",
        instructions=instructions,
        tools=tools,
        middleware=middleware,
    )
