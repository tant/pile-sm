# Pile

**Local-first Scrum Master assistant** — quản lý sprint, phân tích team performance, tư vấn Agile methodology. Chạy hoàn toàn trên máy nội bộ, không gửi data ra cloud.

Built on [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) · LLM via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (self-contained) · UI via [Chainlit](https://chainlit.io)

---

## Highlights

| | Feature | Detail |
|---|---|---|
| **Self-contained** | Zero external LLM dependency | Models auto-download from HuggingFace on first run. No Ollama or LM Studio needed |
| **Jira** | 19 tools | Search, CRUD issues, sprints, boards, backlog, epics, changelog, linking. Write ops require approval |
| **Git** | Read-only | Commit history, branches, diffs, blame. Private repo support |
| **Scrum** | 17 capabilities | Standup, sprint review, retro, workload balance, cycle time, data quality audit, delay alerts, stakeholder summary... |
| **Memory** | Remember & Forget | Long-term memory across sessions. Semantic search via ChromaDB + local embeddings |
| **Knowledge** | RAG | Ingest PDF/markdown → chunk → embed → query. Feed whitepapers, meeting notes, process docs |
| **Browser** | Web Scraping | Playwright + Firefox. Auto-login Jira/GitHub/GitLab. Persistent sessions |
| **Charts** | Auto-visualization | Detect numeric data in responses → render interactive Plotly charts |
| **Bilingual** | VI + EN | Auto-detect ngôn ngữ, trả lời cùng ngôn ngữ |

---

## Quick Start

```bash
git clone git@github.com:tant/pile-sm.git && cd pile-sm
cp .env.sample .env              # configure Jira credentials
uv sync                          # install dependencies
playwright install firefox       # browser engine
```

```bash
# Web UI
uv run chainlit run src/pile/ui/chainlit_app.py

# CLI
uv run pile
```

On first run, Pile automatically downloads ~5.6 GB of GGUF models from HuggingFace:
- **Qwen 3.5 4B** (agent, tool calling) — 2.55 GB
- **Gemma 4 E2B** (router, query classification) — 2.89 GB
- **nomic-embed-text v1.5** (embedding, memory/RAG) — 0.14 GB

Downloads run in parallel with auto-resume. Subsequent runs load models from `~/.pile/models/`.

---

## Configuration

All settings via `.env`:

```env
# --- Model Context Limits ---
AGENT_MAX_TOKENS=32768            # Qwen context window
ROUTER_MAX_TOKENS=4096            # Gemma context window

# --- Logging ---
# DEBUG logs full LLM prompts/responses for troubleshooting
LOG_LEVEL=INFO
LOG_DIR=~/.pile/logs

# --- Jira ---
JIRA_BASE_URL=https://your-instance.atlassian.net
JIRA_EMAIL=your@email.com
JIRA_API_TOKEN=your-token
JIRA_PROJECT_KEY=YOUR_KEY

# --- Git (optional) ---
# GIT_REPOS=/path/to/repo1,/path/to/repo2
# GIT_REPOS_JSON=[{"path":"/repo","url":"https://...","token":"ghp_xxx"}]

# --- Memory / RAG ---
MEMORY_ENABLED=true
MEMORY_STORE_PATH=~/.pile/chromadb

# --- Browser (optional) ---
BROWSER_ENABLED=true
# BROWSER_JIRA_EMAIL=...
# BROWSER_JIRA_PASSWORD=...
# BROWSER_GITHUB_USERNAME=...
# BROWSER_GITHUB_PASSWORD=...
```

Models are fixed in code — no model configuration needed.

---

## Usage Examples

**Sprint & Issues**
```
Sprint hiện tại tiến độ thế nào?
Ai đang bị quá tải?
Có gì đang bị block không?
Tạo bug: Login crash trên mobile, priority High, assign cho Minh
```

**Reports & Analysis**
```
Tổng hợp standup cho team hôm nay
So sánh velocity sprint này vs sprint trước
Cycle time team mình thế nào?
Tóm tắt cho sếp
```

**Memory & Knowledge**
```
Nhớ giúp: team quyết định sprint 2 tuần
Load file /path/to/scale-agile.pdf
SAFe phù hợp team bao nhiêu người?
Quên thông tin về sprint 2 tuần
```

**Browser**
```
Mở trang https://vnexpress.net và tóm tắt tin đầu tiên
Login vào GitHub
```

---

## Architecture

```mermaid
graph TD
    User["User<br/><small>Chainlit Web UI / CLI</small>"]

    subgraph Orchestration["Orchestration Layer — Microsoft Agent Framework"]
        Router["<b>Smart Router</b><br/><small>Keyword → LLM classify → Embedding</small>"]
        Triage["<b>Triage Agent</b><br/><small>Memory + Browser ops</small>"]
        JiraQ["<b>Jira Query Agent</b><br/><small>4 tools · read-only</small>"]
        JiraW["<b>Jira Write Agent</b><br/><small>5 tools · approval required</small>"]
        Board["<b>Board Agent</b><br/><small>3 tools</small>"]
        Sprint["<b>Sprint Agent</b><br/><small>5 tools</small>"]
        Epic["<b>Epic Agent</b><br/><small>3 tools</small>"]
        Git["<b>Git Agent</b><br/><small>5 tools · read-only</small>"]
        Scrum["<b>Scrum Agent</b><br/><small>Prefetch + Git + Memory</small>"]
    end

    subgraph LLM["LLM Layer — Self-contained (llama-cpp-python)"]
        AgentModel["Qwen 3.5 4B<br/><small>Tool calling · n_ctx=32768</small>"]
        RouterModel["Gemma 4 E2B<br/><small>Classify · n_ctx=4096</small>"]
        EmbedModel["nomic-embed-text<br/><small>Embedding · n_ctx=2048</small>"]
    end

    subgraph Infrastructure["Infrastructure"]
        JiraAPI["Jira REST API<br/><small>httpx</small>"]
        GitCLI["Git CLI<br/><small>subprocess</small>"]
        ChromaDB["ChromaDB<br/><small>memories + documents</small>"]
        Browser["Playwright<br/><small>Firefox headless</small>"]
    end

    Charts["Plotly Charts<br/><small>auto-detect numeric data</small>"]

    User -->|request| Router
    Router -->|route| Triage
    Router -->|route| JiraQ
    Router -->|route| JiraW
    Router -->|route| Board
    Router -->|route| Sprint
    Router -->|route| Epic
    Router -->|route| Git
    Router -->|route| Scrum
    JiraQ --> JiraAPI
    JiraW --> JiraAPI
    Board --> JiraAPI
    Sprint --> JiraAPI
    Epic --> JiraAPI
    Triage --> ChromaDB
    Triage --> Browser
    Scrum --> ChromaDB
    Orchestration --> AgentModel
    Router --> RouterModel
    ChromaDB --> EmbedModel
    User -.->|render| Charts
```

**Key design decisions:**
- **Self-contained inference** — llama-cpp-python runs GGUF models locally, no external server
- **3 specialized models** — agent (tool calling), router (classify), embedding (RAG)
- **GPU auto-detect** — macOS Metal, Linux CUDA, CPU fallback
- **Parallel download** — first-run setup downloads all models concurrently
- **8 specialist agents** — each sees only 3-5 tools, prevents model overwhelm
- **All write ops require approval** — human-in-the-loop for safety

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | [Microsoft Agent Framework 1.0](https://github.com/microsoft/agent-framework) |
| LLM Inference | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (GGUF, Metal, CUDA) |
| Agent Model | Qwen 3.5 4B (Q4_K_M) |
| Router Model | Gemma 4 E2B (Q4_K_M) |
| Embeddings | nomic-embed-text v1.5 (Q8_0) |
| Model Download | [huggingface-hub](https://github.com/huggingface/huggingface_hub) + hf_xet |
| Vector Store | ChromaDB (embedded, persistent) |
| Browser | Playwright + Firefox |
| Charts | Plotly (interactive, dark theme) |
| Web UI | Chainlit |
| HTTP Client | httpx (Jira API) |
| Config | pydantic-settings + `.env` |
| Package Manager | uv |

---

## Project Structure

```
src/pile/
├── models/
│   ├── registry.py        # Fixed GGUF model definitions
│   ├── manager.py         # Download + load lifecycle
│   ├── engine.py          # Inference (chat, router, embed)
│   ├── llm_client.py      # MAF-compatible LlamaCppClient
│   └── logging.py         # Inference logger (rotation, levels)
├── agents/
│   ├── triage.py          # Router + memory/browser handler
│   ├── jira_query.py      # Jira read specialist
│   ├── jira_write.py      # Jira write specialist
│   ├── board.py           # Board info
│   ├── sprint.py          # Sprint management
│   ├── epic.py            # Epics + backlog
│   ├── git.py             # Git specialist
│   └── scrum.py           # Scrum Master (prefetch + fallback)
├── tools/
│   ├── jira_tools.py      # 19 Jira REST API tools
│   ├── git_tools.py       # 5 Git CLI tools
│   ├── memory_tools.py    # 6 memory/knowledge tools
│   ├── browser_tools.py   # 6 browser tools (Playwright)
│   └── utils.py           # ADF text conversion
├── memory/
│   ├── store.py           # ChromaDB wrapper (local embedding)
│   └── ingest.py          # PDF/markdown extraction + chunking
├── workflows/
│   ├── interactive.py     # Routed workflow + recovery
│   ├── standup.py         # Sequential pipeline
│   └── planning.py        # GroupChat session
├── ui/
│   ├── chainlit_app.py    # Web UI + file upload + chart rendering
│   ├── charts.py          # Auto chart detection + Plotly builders
│   └── cli.py             # Terminal interface
├── client.py              # LLM client factory (LlamaCppClient)
├── config.py              # Settings (pydantic-settings)
└── health.py              # Startup health checks
```

---

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src/
```

---

## Documentation

- [PRD](docs/PRD.md) — Product requirements
- [Architecture](docs/ARCHITECTURE.md) — Technical design, agent patterns, tool details
- [Roadmap](docs/ROADMAP.md) — Future plans

---

## License

[MIT](LICENSE) — Tan Tran <me@tantran.dev>
