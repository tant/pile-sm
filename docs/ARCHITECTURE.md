# Architecture Design: Pile

> **Version**: 0.8
> **Date**: 2026-04-07
> **Status**: In Development (V3 — Self-contained inference + Memory-aware + Prefetch + Multi-model routing)

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
│  Phase 2: LLM classify — Gemma E2B (~500ms, semantic)          │
│  Phase 3: Embedding fallback (if router model unavailable)      │
│                                                                 │
│  Cache: exact-match (5min TTL, read-only queries)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ agent key
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CONTEXT LAYER                                  │
│                                                                 │
│  Auto-recall: search ChromaDB memories → inject into message    │
│  Prefetch (scrum): fetch Jira data → inject into prompt         │
│  Auto-learn: recovery success → compress lesson → save memory   │
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
│  Embedding: local llama-cpp (nomic-embed-text-v1.5 GGUF)       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      LLM LAYER                                   │
│               Self-contained (llama-cpp-python)                  │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ Router Model     │  │ Agent Model      │  │ Embedding    │  │
│  │ Gemma 4 E2B      │  │ Qwen 3.5 4B     │  │ nomic-embed  │  │
│  │ GGUF Q4_K_M      │  │ GGUF Q4_K_M     │  │ GGUF Q8_0    │  │
│  │ classify only    │  │ tool calling     │  │ memory/RAG   │  │
│  │ n_ctx=4096       │  │ n_ctx=32768     │  │ n_ctx=2048   │  │
│  │ ~500ms, 1 token  │  │ ~20-60s          │  │ ~50ms        │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                 │
│  Auto-download từ HuggingFace on first run                      │
│  Parallel download (ThreadPoolExecutor) + hf_xet acceleration   │
│  Auto-resume nếu bị ngắt giữa chừng                            │
│  GPU auto-detect: macOS Metal / Linux CUDA / CPU fallback       │
│  Storage: ~/.pile/models/{role}/{filename}                      │
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

Khi keyword không match, gọi router model (Gemma E2B) với prompt classify ngắn:

```
Pick one agent for this query. Reply ONLY the agent name.

jira_query = search/view/list issues, who is assigned what, ...
scrum = standup, workload, velocity, review, retro, ...
...

Query: "Khanh đang bận gì vậy?"
Agent:
```

Model trả 1 token duy nhất (ví dụ `jira_query`). Không cần tool calling — bất kỳ model nào follow instruction được đều dùng được.

### Phase 3: Embedding Fallback

Nếu router model không khả dụng, dùng cosine similarity giữa query embedding và agent descriptions. Kém chính xác hơn LLM classify (~60% vs ~89%).

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

### 4.4 Memory Context (Auto-recall + Auto-learn)

Tích hợp memory vào workflow chính — không chỉ khi user chủ động "nhớ giúp".

**Auto-recall** (`context.recall`): Trước mỗi agent run, search ChromaDB memories bằng user query. Nếu có context liên quan (distance < 1.0), inject vào message:

```
User: "Tìm issue resolved tuần này"
Recall: "TETRA project uses 'Done' instead of 'Resolved' status"
→ Agent nhận: "Tìm issue resolved tuần này\n\nRelevant context from memory:\n- TETRA uses Done not Resolved"
→ Agent dùng status=Done ngay lần đầu
```

**Auto-learn** (`context.learn`): Khi recovery trigger thành công (agent A fail → agent B OK), nén bài học qua router model (Gemma E2B) rồi lưu vào ChromaDB:

```
Recovery: board agent failed, jira_query succeeded for "Khanh đang làm gì?"
→ Compress: "Queries about who is doing what should use jira_query, not board"
→ Save to memories collection (type=auto_learn, source=system)
```

**Compression**: Dùng Gemma E2B prompt ngắn "Compress this into ONE short factual statement (under 50 words)". Tiết kiệm token lưu trữ — từ ~100 words xuống ~20 words.

Vòng lặp tự cải thiện:
```
Query fail → recovery → learn lesson → save memory
    ↓
Next similar query → recall lesson → agent đúng từ đầu
```

---

## 5. Orchestration Workflows

### 5.1 Interactive (Primary — Q&A)

```
User → smart_route() → recall memory → [prefetch if scrum] → Agent → Response
                                                                ↓ fail?
                                                            recovery → fallback agent → learn lesson → Response
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

## 7. LLM Layer — Self-Contained Inference

### 7.1 Architecture

Không phụ thuộc external server (Ollama, LM Studio). App tự tải GGUF model từ HuggingFace và chạy inference nội bộ qua `llama-cpp-python`.

### 7.2 Model Registry

Models cố định trong code (`models/registry.py`):

| Role | Model | GGUF | Quantization | Size | Context |
|---|---|---|---|---|---|
| Agent | Qwen 3.5 4B | `unsloth/Qwen3.5-4B-GGUF` | Q4_K_M | 2.55 GB | 32768 |
| Router | Gemma 4 E2B | `unsloth/gemma-4-E2B-it-GGUF` | Q4_K_M | 2.89 GB | 4096 |
| Embedding | nomic-embed-text v1.5 | `nomic-ai/nomic-embed-text-v1.5-GGUF` | Q8_0 | 0.14 GB | 2048 |

Storage: `~/.pile/models/{role}/{filename}`

### 7.3 Model Manager

Lifecycle: check → download → load → serve.

**Download:**
- `huggingface_hub.hf_hub_download()` — auto-resume nếu bị ngắt
- Parallel download 3 models cùng lúc (`ThreadPoolExecutor`)
- `hf_xet` acceleration cho per-file download speed
- First-run auto-trigger, hiển thị progress

**Load:**
- `llama_cpp.Llama()` — 3 singleton instances
- GPU auto-detect: macOS Metal (`n_gpu_layers=-1`), Linux CUDA, fallback CPU
- Agent model: `chat_format="chatml-function-calling"` cho tool calling
- Embedding model: `embedding=True`

### 7.4 MAF Integration

Custom `LlamaCppClient` subclass `BaseChatClient` với `FunctionInvocationLayer` + `ChatMiddlewareLayer`:

```python
class LlamaCppClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        # Convert MAF Messages → llama-cpp dicts
        # Call local engine
        # Convert response → ChatResponse with Content objects
```

Tương thích toàn bộ MAF orchestration: workflows, tool calling, handoffs.

### 7.5 Inference Logging

Logger `pile.inference`, file rotation (50MB, 7 files) tại `~/.pile/logs/inference.log`.

| Level | Nội dung |
|---|---|
| WARNING | Errors, anomalies (empty response, context overflow) |
| INFO | Mỗi LLM call: role, latency, input/output tokens, tool calls, status |
| DEBUG | Full prompt + response content (troubleshoot & phân tích chất lượng) |

---

## 8. Health Checks

Startup checks (`pile.health`), warnings không block:

| Check | Condition | What |
|---|---|---|
| Models | always | GGUF files exist at `~/.pile/models/` |
| Jira | always | Auth valid |
| Browser | if `BROWSER_ENABLED` | Playwright Firefox installed |

---

## 9. Configuration

```ini
# Model Context Limits
AGENT_MAX_TOKENS=32768              # Qwen context window
ROUTER_MAX_TOKENS=4096              # Gemma context window

# Logging
LOG_LEVEL=INFO                      # ERROR, WARNING, INFO, DEBUG
LOG_DIR=~/.pile/logs

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
MEMORY_STORE_PATH=~/.pile/chromadb

# Browser
BROWSER_ENABLED=true
```

Models cố định trong code — không cần config model IDs.

---

## 10. Project Structure

```
src/pile/
├── config.py              # Settings (pydantic-settings + .env)
├── client.py              # LLM client factory (LlamaCppClient)
├── health.py              # Startup health checks (model files, Jira, browser)
├── router.py              # 3-phase router (keyword → LLM → embedding)
├── prefetch.py            # Data prefetch for Scrum Agent
├── context.py             # Auto-recall + auto-learn (memory integration)
├── middleware.py           # ToolCallTracker (timing, logging, loop detection)
├── cache.py               # Semantic cache (exact-match, 5min TTL)
├── models/
│   ├── registry.py        # Fixed GGUF model definitions (HF repos, filenames)
│   ├── manager.py         # Download from HF + load llama-cpp instances
│   ├── engine.py          # Inference functions (chat, router, embed)
│   ├── llm_client.py      # MAF-compatible LlamaCppClient (BaseChatClient)
│   └── logging.py         # Inference logger (rotation, structured output)
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
│   ├── store.py           # ChromaDB wrapper (local embedding)
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

## 11. Design Decisions

| Quyết định | Lý do |
|---|---|
| Self-contained inference (llama-cpp-python) | Zero external dependency — không cần Ollama/LM Studio |
| GGUF format duy nhất | Một format cho cả 3 roles, cross-platform, community lớn |
| Models cố định trong code | User không cần config model IDs — đơn giản hóa setup |
| Parallel download + hf_xet | Giảm thời gian first-run setup (~5.6GB total) |
| Auto-resume download | Mạng yếu không mất progress |
| GPU auto-detect | macOS Metal / Linux CUDA / CPU — user không config |
| LlamaCppClient subclass BaseChatClient | Tương thích MAF orchestration mà không đổi agent/workflow code |
| Inference logging 3 levels | INFO cho monitor, DEBUG cho full prompt/response troubleshoot |
| Deterministic router thay HandoffBuilder | HandoffBuilder thêm 7 transfer tools → overwhelm model 9B/4B |
| LLM classifier (Gemma E2B) thay embedding similarity | Embedding scores quá sát nhau (~0.44-0.47), LLM hiểu ngữ nghĩa tốt hơn (89% vs 60%) |
| Router model riêng, không dùng agent model | Gemma E2B nhanh (~500ms), không cần tool calling. Qwen 4B chuyên tool calling |
| Prefetch data cho Scrum Agent | Model 4B loop tool calls khi tự tìm data, nhưng phân tích tốt khi có data sẵn |
| Loop detection (block tool 3+ lần) | Model 4B hay lặp cùng tool với args hơi khác — cắt sớm buộc phân tích |
| Recovery fallback chains | One-shot routing không perfect → cơ chế tự sửa khi sai |
| Auto-recall trước mỗi agent run | Agent có context từ memory → gọi đúng tool lần đầu, tránh loop |
| Auto-learn từ recovery | Mỗi lần fail-recover thành lesson → lần sau không lặp sai lầm |
| Singleton httpx client cho Jira | Reuse TCP/TLS connections |
| `@_safe_jira_call` decorator | DRY error handling cho tất cả Jira tools |
| Git tools validate input qua regex | Chống command injection |
