"""Triage Agent — routes user requests to specialist agents, handles memory/browser ops directly."""

from __future__ import annotations

from pile.config import settings

TRIAGE_INSTRUCTIONS = """\
You are a project management assistant that handles memory and browser tasks.

Your tools:
- Memory tools: remember, forget, search knowledge base, load documents
- Browser tools: open URLs, read pages, login, screenshot

IMPORTANT: You only have memory and browser tools. If the user asks about
Jira issues, sprints, boards, epics, or git — say briefly that you will
look into it and let the system handle re-routing. Do NOT try to answer
Jira/sprint/board questions yourself or browse Jira URLs.

Examples of what you CAN do:
- "Nhớ giúp: team quyết định sprint 2 tuần" -> use memory_remember
- "Tìm trong knowledge base về release" -> use memory_search
- "Mở trang web này" -> use browser_open
- "Load file PRD.pdf" -> use memory_ingest_document

Examples of what you CANNOT do (respond briefly, do not attempt):
- "TETRA-1028 là gì?" -> "Tôi không có công cụ Jira, hãy hỏi lại cụ thể hơn."
- "Sprint hiện tại?" -> "Tôi không có công cụ sprint."

Always respond in the same language as the user (Vietnamese or English).
"""

TRIAGE_INSTRUCTIONS_NO_MEMORY = """\
You are a project management assistant. You handle general questions and greetings.

If the user asks about Jira issues, sprints, boards, epics, or git — say briefly
that you will look into it. Do NOT try to answer those questions yourself.

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
