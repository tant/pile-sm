# Architecture Design: Pile

> **Version**: 0.4
> **Date**: 2026-04-05
> **Status**: In Development (V2 — Memory + RAG)

---

## 1. Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                                                                 │
│   Chainlit Web UI (localhost:8000)     CLI (stdin/stdout)       │
│   - Streaming chat                    - Scripting/automation    │
│   - Agent step visualization          - Slash commands          │
│   - Quick action starters             - Tool approval prompts   │
│   - Tool approval dialogs                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                           │
│              Microsoft Agent Framework 1.0                       │
│                                                                 │
│  ┌─────────────┐                                                │
│  │   Triage     │ ◄── Entry point, phân loại request            │
│  │   Agent      │     và route đến agent phù hợp                │
│  └──────┬──────┘     hoặc trả lời trực tiếp                    │
│         │ handoff                                                │
│         ▼                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Jira Agent  │  │  Git Agent   │  │  Scrum Agent │          │
│  │              │  │              │  │              │          │
│  │  - Search    │  │  - Commits   │  │  - Standup   │          │
│  │  - CRUD      │  │  - Branches  │  │  - Planning  │          │
│  │  - Sprint    │  │  - Diff      │  │  - Retro     │          │
│  │  - Board     │  │  - Blame     │  │  - Coaching  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                   │
└─────────┼─────────────────┼──────────────────┼──────────────────┘
          │                 │                  │
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TOOL LAYER                               │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────┐                         │
│  │  Jira Tools       │  │  Git Tools   │                         │
│  │                   │  │              │                         │
│  │  jira_search      │  │  git_log     │                         │
│  │  jira_get_issue   │  │  git_diff    │                         │
│  │  jira_get_sprint  │  │  git_branch  │                         │
│  │  jira_get_sprint  │  │  git_show    │                         │
│  │    _issues        │  │  git_blame   │                         │
│  │  jira_get_board   │  │              │                         │
│  │  jira_create_issue│  └──────┬───────┘                         │
│  │  jira_transition  │         │                                 │
│  │    _issue         │         │                                 │
│  │  jira_add_comment │         │                                 │
│  └──────┬────────────┘         │                                 │
│         │                      │                                 │
└─────────┼──────────────────────┼────────────────────────────────┘
          │                      │
          ▼                      ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Jira REST API   │ │  Git CLI (local) │ │  Browser         │
│  (httpx)         │ │  (subprocess)    │ │  (Playwright +   │
└──────────────────┘ └──────────────────┘ │   Firefox)       │
                                          └──────────────────┘

┌─────────────────────────────────────────────────────────────────���
│                      MEMORY LAYER                                │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │  Memory Tools     │  │  Document Tools  │                     │
│  │  memory_remember  │  │  memory_ingest   │                     │
│  │  memory_forget    │  │  memory_list     │                     │
│  │  memory_search    │  │  memory_remove   │                     │
│  └──────┬───────────┘  └──────┬───────────┘                     │
│         │                     │                                  │
│         ▼                     ▼                                  │
│  ┌─────────────────────────────────────────┐                    ��
│  │  ChromaDB (embedded, persistent)         │                    │
│  │  Collections: memories | documents       │                    │
│  │  Storage: ~/.pile/chromadb/              │                    │
│  └─────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      LLM LAYER                                   │
│                                                                 │
│  Ollama @ localhost:11434 (configurable)                            │
│  — hoặc OpenAI-compatible endpoint (LM Studio)                  │
│                                                                 │
│  LLM: configurable via .env (OLLAMA_MODEL_ID / OPENAI_MODEL)   │
│  Default: qwen3.5:9b (tested tool calling OK)                   │
│  Embedding: nomic-embed-text (for Memory/RAG)                   │
│  All agents share the same LLM model                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Agent Design

Tất cả agents được tạo qua `client.as_agent()` — method của Agent Framework tạo agent từ LLM client.

### 2.1 Triage Agent (Entry Point)

**Role**: Router + Memory handler — phân loại user request, handoff đến agent chuyên biệt, xử lý trực tiếp memory/knowledge operations.

Khi `MEMORY_ENABLED=true`, Triage Agent được gắn 6 memory tools để xử lý trực tiếp các yêu cầu nhớ/quên/tìm kiếm/nạp tài liệu — không cần handoff, giảm latency cho model 9B.

```python
triage_agent = client.as_agent(
    name="TriageAgent",
    description="Routes requests + handles memory/knowledge operations directly",
    instructions=TRIAGE_INSTRUCTIONS,
    tools=[memory_remember, memory_forget, memory_search,
           memory_ingest_document, memory_list_documents, memory_remove_document],
)
```

Routing rules:
- Jira-related → handoff to JiraAgent
- Git-related → handoff to GitAgent
- Scrum process → handoff to ScrumAgent
- **Memory/knowledge operations → xử lý trực tiếp bằng memory tools**
- General greetings → respond directly

### 2.2 Jira Agent

**Role**: Chuyên gia Jira — đọc/ghi dữ liệu Jira qua REST API.

```python
jira_agent = client.as_agent(
    name="JiraAgent",
    description="Jira specialist: search, CRUD issues, sprints, boards",
    instructions=JIRA_INSTRUCTIONS.format(
        project_key=settings.jira_project_key,
        jira_url=settings.jira_base_url,
    ),
    tools=[
        jira_search, jira_get_issue,
        jira_get_sprint, jira_get_sprint_issues, jira_get_board,
        jira_create_issue, jira_transition_issue, jira_add_comment,
    ],
)
```

Tools:
- **Read**: `jira_search`, `jira_get_issue`, `jira_get_sprint`, `jira_get_sprint_issues`, `jira_get_board`
- **Write** (require approval): `jira_create_issue`, `jira_transition_issue`, `jira_add_comment`

### 2.3 Git Agent

**Role**: Chuyên gia Git — tương tác với Git repositories local. Optional — không tạo nếu chưa config repos.

```python
git_agent = client.as_agent(
    name="GitAgent",
    description="Git specialist: commits, branches, diffs, blame",
    instructions=GIT_INSTRUCTIONS.format(repos=repos_str),
    tools=[git_log, git_diff, git_branch_list, git_show, git_blame],
)
```

Tất cả tools đều read-only, chạy qua subprocess với validation chống injection:
- `_validate_repo()` — chỉ cho phép repos trong allowlist
- `_validate_ref()` — regex kiểm tra branch/commit ref
- `_validate_path()` — chặn path traversal, absolute paths

### 2.4 Scrum Agent

**Role**: Scrum Master ảo — tổng hợp, phân tích, coaching.

Scrum Agent **có access trực tiếp đến Jira + Git tools** để tự thu thập data khi cần tổng hợp báo cáo, thay vì phải handoff qua lại.

```python
scrum_agent = client.as_agent(
    name="ScrumAgent",
    description="Scrum Master: standup, planning, retro, coaching, reports, data quality, timeline tracking",
    instructions=SCRUM_INSTRUCTIONS.format(
        project_key=settings.jira_project_key,
        git_note=git_note,
        memory_note=memory_note,
    ),
    tools=[jira_search, jira_get_issue, jira_get_sprint, jira_get_sprint_issues,
           git_log, git_diff,    # git tools only if repos configured
           memory_search],       # memory tool only if memory enabled
)
```

Khi `MEMORY_ENABLED=true`, Scrum Agent có thêm `memory_search` để tự query knowledge base khi cần tham khảo methodology, past decisions, hoặc tài liệu đã nạp (ví dụ: SAFe whitepaper).

Prompt được tối ưu cho local LLM (~45 dòng) — mỗi use case mô tả bằng bullet ngắn gọn thay vì hướng dẫn step-by-step chi tiết. Các lĩnh vực:

- Standup, Sprint Review, Retrospective
- Data Quality Audit, Timeline & Delays
- Blocker Tracking, Workload Balance, WIP Limits
- Cycle Time, Sprint Goal, Dependencies
- Stakeholder Summary, Meeting Prep

---

## 3. Orchestration

### 3.1 Handoff Workflow (Primary — Interactive Q&A)

```python
from agent_framework.orchestrations import HandoffBuilder

workflow = (
    HandoffBuilder(
        name="pile_sm",
        participants=[triage, jira, scrum, git],  # git only if configured
    )
    .with_start_agent(triage)
    .add_handoff(triage, [jira, git, scrum])
    .add_handoff(scrum, [jira, git, triage])
    .add_handoff(jira, [triage, scrum])
    .add_handoff(git, [triage, scrum])
    .build()
)
```

**Flow examples**:

```
User: "Sprint hiện tại có bao nhiêu issues?"
  → Triage → handoff → Jira Agent → jira_get_sprint → trả lời

User: "Tạo issue mới: Fix login bug, assign cho Minh"
  → Triage → handoff → Jira Agent → confirm với user → jira_create_issue → done

User: "Tổng hợp standup cho team hôm nay"
  → Triage → handoff → Scrum Agent → jira_search (issues updated)
                                    → git_log (commits today)
                                    → tổng hợp report → trả lời
```

### 3.2 Human-in-the-Loop (Write Operations)

Tất cả write tools dùng `@tool(approval_mode="always_require")`. Khi agent gọi write tool, workflow tạm dừng và hỏi user confirm.

```python
from agent_framework import tool

@tool(approval_mode="always_require")
@_safe_jira_call
def jira_create_issue(
    summary: Annotated[str, Field(description="Issue title")],
    issue_type: Annotated[str, Field(description="Type: Task, Bug, Story, Epic")] = "Task",
    description: Annotated[str | None, Field(description="Issue description")] = None,
    assignee_id: Annotated[str | None, Field(description="Assignee account ID")] = None,
    priority: Annotated[str | None, Field(description="Priority: Highest, High, Medium, Low, Lowest")] = None,
) -> str:
    """Create a new Jira issue. Requires user approval before execution."""
    ...
```

Write tools: `jira_create_issue`, `jira_transition_issue`, `jira_add_comment`

UI handling:
- **Chainlit**: Hiển thị dialog Approve/Reject, xử lý iterative (max 20 rounds)
- **CLI**: Hiển thị tool name + args, hỏi `Approve? (y/n)`

### 3.3 Sequential Workflow (Standup Report)

Dùng khi cần chạy pipeline cố định, trigger bằng `/standup` trong CLI:

```python
from agent_framework.orchestrations import SequentialBuilder

standup_pipeline = SequentialBuilder(
    participants=[jira_agent, git_agent, scrum_agent]  # git_agent only if configured
).build()

# Jira Agent gathers issue updates → Git Agent gathers commits → Scrum Agent synthesizes
```

### 3.4 Group Chat Workflow (Sprint Planning)

Dùng khi cần nhiều agent thảo luận, trigger bằng `/planning` trong CLI:

```python
from agent_framework.orchestrations import GroupChatBuilder, GroupChatState

def round_robin(state: GroupChatState) -> str:
    names = list(state.participants.keys())
    return names[state.current_round % len(names)]

planning_session = GroupChatBuilder(
    participants=[jira_agent, git_agent, scrum_agent],
    selection_func=round_robin,
    termination_condition=lambda msgs: sum(1 for m in msgs if m.role == "assistant") >= 8,
).build()
```

---

## 4. Tool Implementation

### 4.1 Jira Tools

Sử dụng singleton `httpx.Client` để reuse TCP connections:

```python
_client: httpx.Client | None = None

def _jira_client() -> httpx.Client:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.Client(
            base_url=settings.jira_base_url,
            auth=(settings.jira_email, settings.jira_api_token),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=30.0,
        )
    return _client
```

Error handling qua decorator `_safe_jira_call` — catch `ConnectError`, `HTTPStatusError` (401/403/404/429), `TimeoutException` và trả message thay vì raise.

API endpoints:
- Search: `GET /rest/api/3/search/jql`
- Issue: `GET /rest/api/3/issue/{key}`
- Sprint: `GET /rest/agile/1.0/board/{id}/sprint`
- Sprint issues: `GET /rest/agile/1.0/sprint/{id}/issue`
- Board: `GET /rest/agile/1.0/board`
- Create: `POST /rest/api/3/issue`
- Transition: `POST /rest/api/3/issue/{key}/transitions`
- Comment: `POST /rest/api/3/issue/{key}/comment`

### 4.2 Git Tools

Read-only tools chạy qua `subprocess.run()` với timeout 30s. Output truncated tại 4000 chars.

Security:
- Repo path allowlist check (`_validate_repo`)
- Ref validation via regex (`_validate_ref`)
- File path validation chống traversal (`_validate_path`)
- `--` separator để ngăn argument injection
- Private repo credentials qua environment variables (không embed trong command)

### 4.3 Browser Tools

6 tools cho web scraping khi API không khả dụng. Playwright + Firefox, persistent profile.

- **Read**: `browser_open`, `browser_read`, `browser_screenshot`
- **Interaction**: `browser_click`, `browser_fill`
- **Login** (require approval): `browser_login`

**Singleton browser context** (`pile.tools.browser_tools`):

Persistent Firefox context tại `~/.pile/browser/`. Headless mặc định, headed chỉ khi `browser_login`. Session cookies tự động lưu qua restarts.

**Auto-login**: Detect login page (Atlassian ID, GitHub, GitLab) → fill credentials từ `.env` → submit. Fallback `browser_login` nếu auto-login thất bại hoặc không có credentials.

**Content extraction**: `page.inner_text("body")`, truncated 4000 chars. Clean text, không HTML — tối ưu cho model 9B.

### 4.4 Memory Tools

6 tools cho memory và knowledge base management:

- **Read**: `memory_search`, `memory_list_documents`
- **Write** (require approval): `memory_forget`, `memory_ingest_document`, `memory_remove_document`
- **Write** (no approval): `memory_remember` — cho phép lưu nhanh không cần confirm

Error handling qua decorator `_safe_memory_call` — catch `FileNotFoundError`, `ValueError`, và general exceptions.

**ChromaDB Store** (`pile.memory.store`):

Singleton `PersistentClient` với 2 collections:

| Collection | Mục đích | Metadata fields |
|---|---|---|
| `memories` | Explicit remember/forget | `type`, `source`, `created_at` |
| `documents` | Chunks từ ingested files | `doc_id`, `doc_name`, `chunk_index`, `page`, `source_path` |

Embedding qua `OllamaEmbeddingFunction` — gọi Ollama `/api/embed` endpoint với model `nomic-embed-text`.

**Document Ingestion** (`pile.memory.ingest`):

1. Extract text: PyMuPDF cho PDF, built-in read cho markdown/text
2. Chunk: ~500 chars, recursive split theo paragraph → sentence, 50-char overlap
3. Store: mỗi chunk là 1 document trong ChromaDB `documents` collection

### 4.4 Utility Functions

- `extract_text(adf_node)` — parse Atlassian Document Format thành plain text
- `make_adf(text)` — convert plain text thành ADF cho Jira API

---

## 5. Health Checks

Startup health checks (`pile.health`):

- **Ollama**: `GET /api/tags` → kiểm tra server reachable + LLM model available
- **Jira**: `GET /rest/api/3/myself` → kiểm tra auth valid
- **Embedding model**: `GET /api/tags` → kiểm tra `nomic-embed-text` available (chỉ khi `MEMORY_ENABLED=true`)
- **Browser**: Kiểm tra Playwright Firefox browser installed (chỉ khi `BROWSER_ENABLED=true`)

Chạy khi khởi động cả CLI và Chainlit. Warnings hiển thị nhưng không block — cho phép partial functionality.

---

## 6. Configuration

`pydantic-settings` + `.env` file:

```python
class Settings(BaseSettings):
    llm_provider: str = "ollama"          # "ollama", "openai", hoặc "ollama-native"
    ollama_host: str = "http://localhost:11434"
    ollama_model_id: str = "qwen3.5:9b"
    openai_base_url: str = "http://localhost:1234/v1"
    openai_model: str = "qwen3.5:9b"
    openai_api_key: str = "lm-studio"
    jira_base_url: str = "https://your-instance.atlassian.net"
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = ""
    git_repos: str = ""                   # comma-separated paths
    git_repos_json: str = ""              # JSON array with credentials
    memory_enabled: bool = True           # enable Memory + RAG
    memory_store_path: str = "~/.pile/chromadb"  # ChromaDB persist dir
    embedding_model_id: str = "nomic-embed-text" # Ollama embedding model
    browser_enabled: bool = True          # enable Browser tools
    browser_profile_path: str = "~/.pile/browser" # Firefox persistent profile
    browser_jira_email: str = ""          # auto-login credentials
    browser_jira_password: str = ""
    browser_github_username: str = ""
    browser_github_password: str = ""
    browser_gitlab_username: str = ""
    browser_gitlab_password: str = ""
    chainlit_host: str = "0.0.0.0"
    chainlit_port: int = 8000
```

LLM providers:
- `"ollama"` (default): OpenAI-compat client → Ollama `/v1/` endpoint. Tránh bug HandoffBuilder với native client.
- `"openai"`: OpenAI-compat client cho LM Studio hoặc endpoint khác.
- `"ollama-native"`: Native Ollama client. Single-agent only, không support workflows.

---

## 7. Project Structure

```
src/pile/
├── __init__.py
├── config.py           # Settings (pydantic-settings + .env)
├── client.py           # LLM client factory
├── health.py           # Startup health checks
├── agents/
│   ├── triage.py       # Triage (router + memory handler) agent
│   ├── jira.py         # Jira specialist agent
│   ├── git.py          # Git specialist agent
│   └── scrum.py        # Scrum Master agent (+ memory_search)
├── tools/
│   ├── jira_tools.py   # 19 Jira REST API tools (11 read + 8 write)
│   ├── git_tools.py    # Git CLI tools + input validation
│   ├── memory_tools.py # Memory + knowledge base tools (6 tools)
│   ├── browser_tools.py# Browser tools (6 tools, Playwright + Firefox)
│   └── utils.py        # ADF conversion helpers
├── memory/
│   ├── store.py        # ChromaDB wrapper (2 collections, singleton)
│   └── ingest.py       # PDF/markdown extraction + text chunking
├── workflows/
│   ├── interactive.py  # Handoff workflow (primary Q&A)
│   ├── standup.py      # Sequential workflow (standup pipeline)
│   └── planning.py     # GroupChat workflow (sprint planning)
└── ui/
    ├── chainlit_app.py # Chainlit web UI
    ├── charts.py       # Auto chart detection + Plotly builders
    └── cli.py          # Terminal CLI
```

---

## 8. Design Decisions

| Quyết định | Lý do |
|---|---|
| Dùng OpenAI-compat client cho Ollama thay vì native client | Native `OllamaChatClient` có bug với `HandoffBuilder` (#4402) |
| Scrum Agent có access trực tiếp Jira + Git tools | Giảm handoff overhead cho reports cần data từ nhiều nguồn |
| Singleton httpx client cho Jira | Reuse TCP/TLS connections, giảm latency khi gọi nhiều tools liên tiếp |
| `@property` thay vì `@cached_property` cho `git_repo_list` | Cho phép config changes trong tests và reload |
| Prompt Scrum Agent ngắn gọn (~45 dòng) | Tiết kiệm context window cho model 9B |
| Error handling qua decorator `_safe_jira_call` | DRY — tất cả Jira tools share cùng error handling |
| Git tools validate input qua regex | Chống command injection qua user-controlled parameters |
| Iterative approval loop trong Chainlit (max 20 rounds) | Tránh stack overflow từ recursive calls, hỗ trợ batch operations |
| ChromaDB embedded thay vì client-server | Chạy in-process, không cần server riêng, phù hợp local-first |
| `nomic-embed-text` qua Ollama thay vì sentence-transformers | Tận dụng Ollama infra sẵn có, tránh thêm ~2GB torch dependencies |
| Tool-based retrieval thay vì ContextProvider | ContextProvider inject vào mọi call → lãng phí token cho model 9B. Tool-based = chỉ gọi khi cần |
| Memory tools gắn vào Triage thay vì Memory Agent riêng | Giảm handoff = giảm latency, memory ops đơn giản không cần agent riêng suy nghĩ |
| Scrum Agent có `memory_search` trực tiếp | Cho phép Scrum tự tra knowledge base khi tư vấn methodology, không cần handoff |
| File upload bypass agent → ingest trực tiếp | Upload file qua Chainlit UI tự ingest ngay, không cần agent gọi tool — nhanh, không tốn LLM call |
| Playwright + Firefox thay vì Browser Use/MCP | Direct Playwright sync API đơn giản, không thêm LangChain dependency, phù hợp tool pattern hiện tại |
| Playwright chạy trong thread riêng (`ThreadPoolExecutor`) | Sync API xung đột với asyncio event loop của agent framework — chạy trong dedicated thread giải quyết |
| Auto-login + persistent session | Login 1 lần (auto hoặc manual), persistent profile giữ cookies — không cần re-login |
| Clean text extraction thay vì accessibility tree | `inner_text("body")` truncated 4000 chars — tiết kiệm token, đủ context cho model 9B |
| Browser tools gắn vào Triage (interaction) + Scrum (read-only) | Triage xử lý login/click/fill, Scrum chỉ cần đọc data từ web pages |
| Auto chart detection thay vì dedicated chart tool | Post-processing transparent, không thêm tool cho model 9B, user muốn automatic |
| Plotly dark theme | Consistent với Chainlit dark theme default |
