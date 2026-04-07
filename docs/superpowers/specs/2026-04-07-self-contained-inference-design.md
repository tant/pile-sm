# Self-Contained Inference — Design Spec

> **Date**: 2026-04-07
> **Status**: Implemented
> **Goal**: Loại bỏ dependency vào Ollama/LM Studio. App tự tải GGUF model từ HuggingFace và chạy inference nội bộ qua llama-cpp-python.

---

## 1. Context

Pile hiện tại phụ thuộc external LLM server (Ollama hoặc LM Studio) cho 3 roles:

- **Agent model** (Qwen 3.5 4B) — tool calling, chat
- **Router model** (Gemma 4 E2B) — query classification
- **Embedding model** (nomic-embed-text-v1.5) — memory/RAG

User phải cài và chạy Ollama/LM Studio trước khi dùng Pile. Mục tiêu: zero external LLM dependency.

---

## 2. Approach

**llama-cpp-python duy nhất** làm inference engine cho cả 3 model roles.

- Một dependency, một format (GGUF), một code path
- Auto-detect GPU: macOS Metal, Linux CUDA, fallback CPU
- Model fix cứng trong code, user không cần config model

---

## 3. Model Registry

Module `src/pile/models/registry.py`.

3 models cố định, map từ model đang dùng hiện tại sang GGUF format:

| Role | Model hiện tại | GGUF tương ứng (xác định lúc implement) |
|---|---|---|
| agent | qwen3.5-4b-mlx | Qwen 3.5 4B GGUF |
| router | gemma-4-e2b-it | Gemma 4 E2B GGUF |
| embedding | text-embedding-nomic-embed-text-v1.5 | nomic-embed-text-v1.5 GGUF |

Registry chứa: HF repo, filename, size_gb, role description.

Storage: `~/.pile/models/{role}/{filename}`

---

## 4. Model Manager — Download & Lifecycle

Module `src/pile/models/manager.py`.

### Download

- `huggingface_hub.hf_hub_download()` tải GGUF từ HuggingFace
- Progress bar hiển thị: tên model, role, size, % tiến trình
- Check file tồn tại → skip (không tải lại)
- Tải tuần tự từng model

### Load

- `llama_cpp.Llama()` load model vào RAM/GPU
- 3 singleton instances: `_agent_model`, `_router_model`, `_embed_model`
- Embedding model load với `embedding=True`
- Auto-detect GPU:
  - macOS → Metal (`n_gpu_layers=-1`)
  - Linux CUDA → `n_gpu_layers=-1`
  - Fallback CPU → `n_gpu_layers=0`

### First-run Flow

```
pile (lần đầu)
  → manager.ensure_models()
  → detect missing models
  → "Downloading models for first-time setup..."
  → [1/3] Router model (~1.5GB) ████████░░ 80%
  → [2/3] Agent model (~2.7GB)  ██████████ Done
  → [3/3] Embedding model (~140MB) ██████████ Done
  → "Setup complete. Starting Pile..."
  → load models → app chạy bình thường
```

Lần chạy sau: models đã có → load trực tiếp.

---

## 5. Inference Engine

Module `src/pile/models/engine.py`.

3 functions thay thế toàn bộ external API calls:

```python
def chat_completion(messages, tools=None, max_tokens=2048) -> dict
    # Agent model — tool calling, chat

def router_completion(prompt, max_tokens=20) -> str
    # Router model — classify query, 1 token response

def embed(texts: list[str]) -> list[list[float]]
    # Embedding model — vector embeddings cho ChromaDB
```

---

## 6. MAF Integration — LlamaCppClient

Module `src/pile/models/llm_client.py`.

MAF dùng protocol-based design. Custom client subclass `BaseChatClient` với đủ 4 layers:

```python
class LlamaCppClient(
    FunctionInvocationLayer,    # tool calling support
    ChatMiddlewareLayer,        # middleware
    ChatTelemetryLayer,         # logging
    BaseChatClient,             # base interface
):
    def _inner_get_response(self, *, messages, stream, options, **kwargs):
        # wrap llama-cpp-python chat_completion → ChatResponse
```

Tương thích toàn bộ MAF orchestration: workflows, tool calling, handoffs.

---

## 7. Integration Points — Files thay đổi

| File | Thay đổi |
|---|---|
| `client.py` | `create_client()` → trả `LlamaCppClient`. `call_router_model()` → gọi `router_completion()` |
| `memory/store.py` | `_embedding_fn()` → custom `EmbeddingFunction` wrap `embed()` |
| `config.py` | Xóa: `llm_provider`, `ollama_*`, `openai_*`, `router_model`, `embedding_model_id` |
| `health.py` | Check model files exist thay vì check endpoint reachable |
| `pyproject.toml` | Thêm: `llama-cpp-python`, `huggingface-hub`. Xóa: `agent-framework-ollama`, `agent-framework-openai` |

---

## 8. Files KHÔNG thay đổi

- `agents/` — toàn bộ 8 agents
- `workflows/` — interactive, standup, planning
- `router.py` — 3-phase routing logic
- `prefetch.py` — scrum data prefetch
- `context.py` — auto-recall, auto-learn
- `middleware.py` — loop detection
- `cache.py` — query cache
- `tools/` — jira, git, memory, browser tools
- `ui/` — chainlit, cli, charts

---

## 9. Config đơn giản hóa

`.env` sau refactor — xóa toàn bộ LLM provider config, thêm context limits và log level:

```env
# --- Model Context Limits ---
AGENT_MAX_TOKENS=32768
ROUTER_MAX_TOKENS=4096

# --- Logging ---
# Levels: ERROR, WARNING, INFO, DEBUG
# INFO  = mỗi call: role, latency, token count, success/error
# DEBUG = full prompt + response content (troubleshoot & phân tích chất lượng)
LOG_LEVEL=INFO
LOG_DIR=~/.pile/logs

# --- Jira ---
JIRA_BASE_URL=https://instance.atlassian.net
JIRA_EMAIL=user@email.com
JIRA_API_TOKEN=token
JIRA_PROJECT_KEY=PROJ

# --- Git (optional) ---
GIT_REPOS=

# --- Memory / RAG ---
MEMORY_ENABLED=true
MEMORY_STORE_PATH=~/.pile/chromadb

# --- Browser (optional) ---
BROWSER_ENABLED=true
BROWSER_PROFILE_PATH=~/.pile/browser

# --- UI ---
CHAINLIT_HOST=0.0.0.0
CHAINLIT_PORT=8000
```

---

## 10. Dependencies

### Thêm

- `llama-cpp-python` — inference engine (GGUF, Metal, CUDA, CPU)
- `huggingface-hub` — download models từ HF

### Xóa

- `agent-framework-ollama`
- `agent-framework-openai`

### Giữ nguyên

- `agent-framework-orchestrations` — workflows, handoff, tool decorator
- `chromadb`, `chainlit`, `httpx`, `playwright`, `plotly`, `pymupdf`, etc.

---

## 11. Inference Logging

Logger: `pile.inference`. Log file: `~/.pile/logs/inference.log` (rotation 7 ngày, max 50MB/file).

### Log Levels

**WARNING** — chỉ errors và anomalies:

```
[WARN] router call failed: model returned empty response (latency=1200ms)
[WARN] agent call error: context length exceeded (input=35000 tokens, max=32768)
```

**INFO** — mỗi LLM call, đủ để monitor health:

```
[INFO] role=agent latency=3200ms input_tokens=1200 output_tokens=85 tool_calls=2 status=ok
[INFO] role=router latency=450ms input_tokens=80 output_tokens=1 status=ok
[INFO] role=embedding latency=50ms texts=3 status=ok
```

**DEBUG** — full prompt và response content, dùng để troubleshoot và phân tích chất lượng LLM:

```
[DEBUG] role=agent request:
  messages=[{"role":"system","content":"You are a Jira specialist..."},{"role":"user","content":"Tìm issue của Khanh"}]
  tools=["jira_search","jira_get_issue","jira_get_changelog","jira_curl_command"]
  max_tokens=32768
[DEBUG] role=agent response:
  content="Tôi sẽ tìm các issue được assign cho Khanh..."
  tool_calls=[{"name":"jira_search","args":{"jql":"assignee=khanh AND project=TETRA"}}]
  finish_reason=tool_calls
```

### Config

`config.py` thêm:

```python
log_level: str = "INFO"          # ERROR, WARNING, INFO, DEBUG
log_dir: str = "~/.pile/logs"
agent_max_tokens: int = 32768    # Qwen context window
router_max_tokens: int = 4096    # Gemma context window
```

`agent_max_tokens` và `router_max_tokens` truyền vào `llama_cpp.Llama(n_ctx=...)` khi load model.
