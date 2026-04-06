# Architecture Design: Pile

> **Version**: 0.6
> **Date**: 2026-04-06
> **Status**: In Development (V2 — Prefetch + Multi-model routing + Recovery)

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
│                    ROUTING LAYER                                 │
│                                                                 │
│  Phase 1: Keyword regex (<1ms, ~75% queries)                    │
│  Phase 2: LLM classify — Gemma 2B (~500ms, semantic)           │
│  Phase 3: Embedding fallback (if no router model)               │
│                                                                 │
│  Cache: exact-match (5min TTL, read-only queries)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ agent key
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREFETCH LAYER (scrum only)                    │
│                                                                 │
│  Detect query type (standup/velocity/blockers/...)              │
│  → Fetch Jira data deterministically (API calls, no LLM)        │
│  → Inject data into agent prompt                                │
│  Other agents: skip prefetch, use tools directly                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT LAYER                                   │
│              Microsoft Agent Framework 1.0                       │
│                                                                 │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│  │ JiraQuery  │ │ JiraWrite  │ │   Board    │ │  Sprint    │  │
│  │ 4 tools    │ │ 5 tools    │ │ 3 tools    │ │ 5 tools    │  │
│  │ read-only  │ │ approval   │ │ read-only  │ │ mixed      │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐  │
│  │   Epic     │ │    Git     │ │   Scrum    │ │  Triage    │  │
│  │ 3 tools    │ │ 5 tools    │ │ prefetch   │ │ mem+browser│  │
│  │ read-only  │ │ read-only  │ │ or tools   │ │ fallback   │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘  │
│                                                                 │
│  Loop detection: block same tool called 3+ times                │
│  Recovery: agent fail → fallback chain → retry 1x               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│  Jira REST API   │ │  Git CLI     │ │  Browser         │
│  (httpx)         │ │  (subprocess)│ │  (Playwright)    │
└──────────────────┘ └──────────────┘ └──────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      MEMORY LAYER                                │
│                                                                 │
│  ChromaDB (embedded, persistent at ~/.pile/chromadb/)            │
│  Collections: memories | documents                              │
│  Embedding: configured provider (LM Studio / Ollama)            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      LLM LAYER                                   │
│                                                                 │
│  Provider: LM Studio (OpenAI-compatible) / Ollama               │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ Router Model     │  │ Agent Model      │  │ Embedding    │  │
│  │ gemma-4-e2b-it   │  │ qwen3.5-4b-mlx   │  │ nomic-embed  │  │
│  │ classify only    │  │ tool calling     │  │ memory/RAG   │  │
│  │ ~500ms, 1 token  │  │ ~20-60s          │  │ ~50ms        │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Routing — 3-Phase Smart Router

Deterministic routing thay thế HandoffBuilder (bị overwhelm model 9B với 7 transfer tools).

### Phase 1: Keyword Matching (<1ms)

Regex patterns kiểm tra theo thứ tự ưu tiên. First match wins.

```python
_ROUTES = [
    ("memory",     [r"nhớ giúp", r"remember", r"forget", ...]),
    ("browser",    [r"mở trang", r"open url", r"https?://", ...]),
    ("jira_query", [r"[A-Z]+-\d+", ...]),           # issue key → jira_query
    ("jira_query", [r"curl", r"changelog", ...]),
    ("jira_write", [r"tạo issue", r"create bug", r"assign", ...]),
    ("board",      [r"\bboard\b", ...]),
    ("scrum",      [r"standup", r"velocity", r"workload", r"báo cáo", ...]),
    ("sprint",     [r"\bsprint\b", ...]),
    ("epic",       [r"\bepic\b", r"\bbacklog\b"]),
    ("git",        [r"\bgit\b", r"\bcommit\b", ...]),
    ("jira_query", [r"tìm", r"search", r"issue", r"status", ...]),
    ("triage",     [r"^chào", r"^hello", r"^hi", ...]),
]
```

Xử lý ~70% queries. Thứ tự quan trọng — scrum trước sprint để bắt "sprint review", "báo cáo sprint".

### Phase 2: LLM Classifier (~500ms)

Khi keyword không match, gọi router model (Gemma 2B) với prompt classify ngắn:

```
Pick one agent for this query. Reply ONLY the agent name.

jira_query = search/view/list issues, who is assigned what, ...
scrum = standup, workload, velocity, review, retro, ...
...

Query: "Khanh đang bận gì vậy?"
Agent:
```

Model trả 1 token duy nhất (ví dụ `jira_query`). Không cần tool calling — bất kỳ model nào follow instruction được đều dùng được (Gemma, Phi, SmolLM...).

Config qua `ROUTER_MODEL` trong `.env`. Cùng provider endpoint, chỉ khác model ID.

### Phase 3: Embedding Fallback

Nếu `ROUTER_MODEL` không set, dùng cosine similarity giữa query embedding và agent descriptions. Kém chính xác hơn LLM classify (~60% vs ~89%).

### Caching

Exact-match cache (MD5 key, 5-min TTL, max 100 entries). Chỉ cache read-only queries — skip `jira_write`, `memory`, `browser`.

---

## 3. Agent Design — 8 Specialist Agents

Mỗi agent chỉ thấy 3-5 tools riêng. Tránh overwhelm model nhỏ.

### 3.1 JiraQuery Agent (4 tools)

Đọc data Jira: `jira_search`, `jira_get_issue`, `jira_get_changelog`, `jira_curl_command`.

### 3.2 JiraWrite Agent (5 tools, approval required)

Ghi data Jira: `jira_create_issue`, `jira_update_issue`, `jira_transition_issue`, `jira_add_comment`, `jira_link_issues`. Tất cả require approval.

### 3.3 Board Agent (3 tools)

Board info: `jira_list_boards`, `jira_get_board`, `jira_get_board_config`.

### 3.4 Sprint Agent (5 tools)

Sprint management: `jira_get_sprint`, `jira_get_sprint_issues`, `jira_create_sprint`, `jira_move_to_sprint`, `jira_move_to_backlog`. Write ops require approval.

### 3.5 Epic Agent (3 tools)

Epics + backlog: `jira_get_epics`, `jira_get_epic_issues`, `jira_get_backlog`.

### 3.6 Git Agent (5 tools, optional)

Git repos: `git_log`, `git_diff`, `git_branch_list`, `git_show`, `git_blame`. Chỉ tạo khi `GIT_REPOS` configured. Input validation chống injection.

### 3.7 Scrum Agent (2 modes)

Scrum Master — operates in 2 modes depending on data availability:

**Prefetch mode** (primary): Workflow fetches Jira data deterministically before agent runs. Agent receives data in system prompt, no Jira tools — only `git_log`, `git_diff`, `memory_search` for deep-dive.

**Fallback mode**: If prefetch not possible (no board_id), agent gets full Jira tools: `jira_search`, `jira_get_issue`, `jira_get_board`, `jira_get_sprint_issues`, `jira_get_changelog` + optional git/memory.

Tại sao 2 modes: model 4B phân tích tốt khi có data sẵn, nhưng loop tool calls khi phải tự tìm data. Prefetch loại bỏ đúng phần model yếu.

Xử lý: standup, sprint review, velocity, workload, blockers, cycle time, data quality, stakeholder summary, meeting prep.

### 3.8 Triage Agent (memory + browser tools)

Xử lý memory operations (remember, forget, search, ingest documents) và browser tasks (open URLs, scrape, login). Không có Jira/Git tools — nếu nhận query Jira, nó thú nhận không có tool phù hợp → trigger recovery.

### Tool Overlap (fallback mode)

| Tool | Agents có access |
|---|---|
| `jira_search`, `jira_get_issue` | JiraQuery, Scrum (fallback) |
| `jira_get_changelog` | JiraQuery, Scrum (fallback) |
| `jira_get_board` | Board, Scrum (fallback) |
| `jira_get_sprint_issues` | Sprint, Scrum (fallback) |
| `git_log`, `git_diff` | Git, Scrum (both modes) |
| `memory_search` | Triage, Scrum (both modes) |
| `browser_open`, `browser_read` | Triage |

Trong prefetch mode, Scrum Agent không có Jira tool overlap — data đã inject sẵn.

---

## 4. Safety Mechanisms

### 4.1 Loop Detection (middleware)

Middleware `ToolCallTracker` block cùng tool gọi lần thứ 3+:

```
jira_search({jql: "..."}) → OK
jira_search({jql: "..."}) → OK (different args)
jira_search({...})        → BLOCKED: "analyze the data you have"
```

Model 4B hay lặp tool calls với JQL hơi khác nhau. Loop detection cắt sớm, buộc model phân tích data đã có.

### 4.2 Recovery Mechanism

Khi agent cho kết quả kém, workflow tự re-route sang agent khác.

**Failure Detection** (deterministic, không cần LLM):
- Response quá ngắn (<20 chars)
- Agent có tools nhưng không gọi tool nào (trừ triage và scrum prefetch)
- Tất cả tool calls đều trả error

**Fallback Chains**:

```python
_FALLBACK_CHAINS = {
    "triage":     ["jira_query", "scrum", "sprint"],
    "board":      ["sprint", "jira_query"],
    "sprint":     ["scrum", "jira_query"],
    "epic":       ["sprint", "jira_query"],
    "scrum":      ["jira_query", "sprint"],
    "jira_query": ["scrum", "sprint"],
    "git":        ["jira_query"],
}
```

**Flow**:

```
Agent A chạy → _detect_failure() → True?
  → _get_fallback(A) → Agent B
  → Agent B chạy → return (dù tốt hay xấu)
```

- Max 1 retry (tổng cộng 2 agent runs)
- Không retry `jira_write` (write ops quá rủi ro)
- Scrum prefetch mode: `tools=0` là expected, không trigger recovery
- Chỉ cache response nếu quality OK

### 4.3 Data Prefetch (Scrum)

Thay vì để model 4B tự quyết gọi tool nào (gây loop), workflow fetch data trước:

```
"velocity?" → detect_scrum_type() → "velocity"
           → prefetch: jira_get_board() + jira_get_sprint_issues() + closed sprints
           → inject 5-6K data vào prompt
           → Scrum Agent phân tích ngay (không cần gọi tool)
```

Query type mapping (`prefetch.py`):

| Query type | Data fetched |
|-----------|-------------|
| standup | sprint issues + issues updated last 24h |
| velocity | board info + sprint issues + closed sprints |
| sprint_review, stakeholder | board info + sprint issues |
| workload | sprint issues |
| blockers | sprint issues + blocked issues search |
| retro | board info + sprint issues + recently done issues |
| data_quality | sprint issues |
| general | board info + sprint issues |

---

## 5. Orchestration Workflows

### 5.1 Interactive (Primary — Q&A)

```
User → smart_route() → [Prefetch if scrum] → Agent → [Recovery if fail] → Response
```

Mỗi agent có riêng `AgentSession` để giữ conversation history.

### 5.2 Standup Pipeline (`/standup`)

Sequential: Jira Agent → Git Agent → Scrum Agent (synthesis).

### 5.3 Sprint Planning (`/planning`)

GroupChat: Round-robin giữa Jira/Git/Scrum agents, max 8 turns.

### 5.4 Human-in-the-Loop

Write tools dùng `@tool(approval_mode="always_require")`. Chainlit hiển thị dialog, CLI hỏi `Approve? (y/n)`.

---

## 6. Tool Implementation

### 6.1 Jira Tools (19 total: 11 read + 8 write)

Singleton `httpx.Client`, error handling qua `@_safe_jira_call` decorator. Auto-detect `board_id` on startup.

### 6.2 Git Tools (5, read-only)

Subprocess với timeout 30s, output truncated 4000 chars. Input validation: repo allowlist, ref regex, path traversal check.

### 6.3 Memory Tools (6)

ChromaDB vector store, 2 collections (memories + documents). Document ingestion: PDF (PyMuPDF) / markdown → chunk ~500 chars → embed → store.

### 6.4 Browser Tools (6)

Playwright + Firefox, persistent profile at `~/.pile/browser/`. Auto-login cho Jira/GitHub/GitLab. Headless mặc định.

---

## 7. Health Checks

Startup checks (`pile.health`), warnings không block:

| Check | Condition | What |
|---|---|---|
| LLM provider | always | Endpoint reachable + LLM model loaded |
| Router model | if `ROUTER_MODEL` set | Router model loaded on same provider |
| Jira | always | Auth valid |
| Embedding model | if `MEMORY_ENABLED` | Embedding model loaded on configured provider |
| Browser | if `BROWSER_ENABLED` | Playwright Firefox installed |

---

## 8. Configuration

```ini
# LLM Provider
LLM_PROVIDER=openai                        # "ollama" | "openai" | "ollama-native"
OPENAI_BASE_URL=http://localhost:1234/v1    # LM Studio
OPENAI_MODEL=qwen3.5-4b-mlx                # Agent model (tool calling)
OPENAI_API_KEY=lm-studio

# Router Model
ROUTER_MODEL=gemma-4-e2b-it                # Query classifier (no tool calling needed)

# Jira
JIRA_BASE_URL=https://instance.atlassian.net
JIRA_EMAIL=user@email.com
JIRA_API_TOKEN=token
JIRA_PROJECT_KEY=PROJ

# Agent Limits
AGENT_MAX_ITERATIONS=5
AGENT_MAX_FUNCTION_CALLS=15

# Memory / RAG
MEMORY_ENABLED=true
EMBEDDING_MODEL_ID=text-embedding-nomic-embed-text-v1.5

# Browser
BROWSER_ENABLED=true
```

Tất cả models (agent, router, embedding) chạy trên cùng provider endpoint.

---

## 9. Project Structure

```
src/pile/
├── config.py              # Settings (pydantic-settings + .env)
├── client.py              # LLM client factory
├── health.py              # Startup health checks (provider-aware)
├── router.py              # 3-phase router (keyword → LLM → embedding)
├── prefetch.py            # Data prefetch for Scrum Agent
├── middleware.py           # ToolCallTracker (timing, logging, loop detection)
├── cache.py               # Semantic cache (exact-match, 5min TTL)
├── agents/
│   ├── triage.py          # Memory + browser handler
│   ├── jira_query.py      # Jira read (search, details, changelog, curl)
│   ├── jira_write.py      # Jira write (create, update, transition, comment)
│   ├── board.py           # Board info
│   ├── sprint.py          # Sprint management
│   ├── epic.py            # Epics + backlog
│   ├── git.py             # Git specialist (optional)
│   └── scrum.py           # Scrum Master (prefetch mode + fallback mode)
├── tools/
│   ├── jira_tools.py      # 19 Jira REST API tools
│   ├── git_tools.py       # 5 Git CLI tools + validation
│   ├── memory_tools.py    # 6 Memory/RAG tools
│   ├── browser_tools.py   # 6 Browser tools (Playwright)
│   └── utils.py           # ADF conversion
├── memory/
│   ├── store.py           # ChromaDB wrapper (provider-aware embedding)
│   └── ingest.py          # PDF/markdown → chunks
├── workflows/
│   ├── interactive.py     # Routed workflow + recovery
│   ├── standup.py         # Sequential pipeline
│   └── planning.py        # GroupChat (sprint planning)
└── ui/
    ├── cli.py             # Terminal CLI
    ├── chainlit_app.py    # Web UI
    └── charts.py          # Auto chart detection + Plotly
```

---

## 10. Design Decisions

| Quyết định | Lý do |
|---|---|
| Deterministic router thay HandoffBuilder | HandoffBuilder thêm 7 transfer tools → overwhelm model 9B/4B |
| LLM classifier (Gemma 2B) thay embedding similarity | Embedding scores quá sát nhau (~0.44-0.47), LLM hiểu ngữ nghĩa tốt hơn (89% vs 60%) |
| Router model riêng, không dùng agent model | Gemma 2B nhanh (~500ms), không cần tool calling. Qwen 4B chuyên tool calling |
| Keyword patterns thu hẹp, đẩy ambiguous sang LLM | Keyword bắt sai 7% → thu hẹp pattern rộng, LLM xử lý ambiguous tốt hơn |
| Prefetch data cho Scrum Agent | Model 4B loop tool calls khi tự tìm data, nhưng phân tích tốt khi có data sẵn |
| Scrum 2 modes (prefetch + fallback) | Prefetch cho 90% queries; fallback giữ full tools cho edge cases |
| Loop detection (block tool 3+ lần) | Model 4B hay lặp cùng tool với args hơi khác — cắt sớm buộc phân tích |
| Recovery fallback chains | One-shot routing không perfect → cơ chế tự sửa khi sai |
| `_scrum_prefetch` key riêng | Không ghi đè fallback scrum agent — cả 2 modes tồn tại cùng lúc |
| Max 1 retry | Trade-off speed vs resilience — 2 agent runs là chấp nhận được |
| Triage không có Jira tools | Giữ mỗi agent focused. Recovery sẽ chuyển sang đúng agent |
| Singleton httpx client cho Jira | Reuse TCP/TLS connections |
| Cùng 1 provider cho tất cả models | Đơn giản ops — 1 endpoint (LM Studio hoặc Ollama) |
| Embedding giữ cho Memory/RAG, bỏ khỏi routing | Embedding tốt cho semantic search docs, kém cho classify intent |
| `@_safe_jira_call` decorator | DRY error handling cho tất cả Jira tools |
| Git tools validate input qua regex | Chống command injection |
| Tool-based retrieval thay ContextProvider | ContextProvider inject mọi call → lãng phí token |
