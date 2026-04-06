# Multi-Model Architecture: Right Model, Right Job

> **Status**: Implemented (Router + Agent + Memory Compression) / Planned (Synthesizer)
> **Date**: 2026-04-06
> **Context**: Pile Scrum Master agent, 8GB RAM, Apple Silicon

---

## Ý tưởng

Thay vì dùng 1 model cho mọi việc, phân chia 3 vai trò cho 3 model khác nhau — mỗi model chỉ làm việc nó giỏi nhất.

## Hiện tại: 3-Model Pipeline (Đã implement)

```
User Input
  │
  ├── [1] Keyword Router (regex, <1ms)
  │     └── match → agent key (70% queries)
  │
  ├── [2] Gemma 4 E2B — Router Model (~500ms)
  │     └── classify query → 1 token output (agent name)
  │     └── không cần tool calling, chỉ follow instruction
  │
  ├── [3] Qwen 3.5-4B — Agent Model (~20-60s)
  │     └── tool calling + response generation
  │     └── 3-5 tools per agent, max 5 iterations
  │
  ├── [4] nomic-embed-text — Embedding Model (~50ms)
  │     └── memory search, document RAG
  │     └── KHÔNG dùng cho routing nữa
  │
  └── [*] Gemma 4 E2B — Memory Compression (~500ms)
        └── compress lessons before saving to memory
        └── reuse same model as router (no extra RAM)
```

### Tại sao tách Router Model?

| Aspect | Embedding routing (cũ) | LLM classify (mới) |
|--------|----------------------|-------------------|
| Accuracy | ~60% (scores sát nhau 0.44-0.47) | ~89% (hiểu ngữ nghĩa) |
| Latency | ~200ms | ~500ms |
| "Khanh đang làm gì?" | → board (SAI) | → jira_query (ĐÚNG) |
| "Tình hình thế nào?" | → triage (SAI) | → scrum (ĐÚNG) |

Embedding chỉ so mặt chữ giữa query và agent description. LLM 2B hiểu context: "Khanh đang làm gì" = hỏi về tasks của 1 người = cần Jira search.

### Resource Usage (8GB Mac)

| Model | Role | RAM | Latency |
|---|---|---|---|
| Gemma 4 E2B (2B) | Router classify + memory compress | ~1.5GB | ~500ms |
| Qwen 3.5-4B | Agent (tool calling) | ~3GB | 20-60s |
| nomic-embed-text | Memory/RAG embedding | ~0.3GB | ~50ms |
| **Tổng** | | **~5GB** | Dư cho OS + context |

LM Studio trên Apple Silicon serve tất cả cùng lúc qua unified memory.

## Cấu hình

```ini
# .env
LLM_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_MODEL=qwen3.5-4b-mlx           # Agent model
ROUTER_MODEL=gemma-4-e2b-it           # Router model (bỏ trống = dùng embedding)
EMBEDDING_MODEL_ID=text-embedding-nomic-embed-text-v1.5
```

Tất cả dùng cùng 1 provider endpoint. Chỉ khác model ID.

## Recovery Mechanism

Khi router classify sai hoặc agent cho kết quả kém:

```
Agent A fail → _detect_failure() → fallback chain → Agent B retry
```

Failure signals (deterministic, không cần LLM):
- Response < 20 chars
- Agent có tools nhưng không gọi tool nào
- Tất cả tool calls trả error

Max 1 retry. Không retry write operations.

## Benchmark tham khảo

### Tool Calling (TAU2-Bench)

| Model | TAU2 Score | Vai trò |
|---|---|---|
| Qwen 3.5-4B | **79.9** | Agent (tool executor) — giỏi nhất trong tầm giá |
| Qwen 3.5-2B | 48.8 | Backup agent nếu cần tiết kiệm |
| Gemma 4 E4B | 42.2 | KHÔNG dùng cho tool calling |
| Gemma 4 E2B | 24.5 | KHÔNG dùng cho tool calling — chỉ classify/synthesize |

### Router Accuracy (18 test queries)

| Method | Accuracy | Avg latency |
|---|---|---|
| Keyword only | 70% (miss ambiguous) | <1ms |
| Keyword + Embedding | 60% (scores quá sát) | ~200ms |
| Keyword + Gemma 2B | **89%** | ~500ms |

## Planned: Synthesizer Layer (chưa implement)

Thêm step cuối: nếu agent response cần phân tích sâu, pass qua synthesizer model.

```
Agent (Qwen 4B)  →  raw tool data  →  Synthesizer (Gemma 2B/4B)  →  báo cáo đẹp
```

Khi nào cần synthesizer:
- **Không cần**: "có mấy board", "tạo bug", "curl command" → Qwen trả trực tiếp
- **Cần**: "phân tích sprint chi tiết", "standup cho team" → data cần tổng hợp

Criteria: response > 200 chars + query chứa keyword phân tích/tổng hợp/báo cáo.

## Mở rộng: 16GB RAM

| Role | Model | RAM |
|---|---|---|
| Router | Gemma 4 E2B | ~1.5GB |
| Agent | Qwen 3.5-9B | ~6GB |
| Synthesizer | Gemma 4 E4B | ~4GB |
| Embedding | nomic-embed-text | ~0.3GB |
| **Tổng** | | **~12GB** |

9B agent cho tool calling chính xác hơn, E4B synthesizer cho output quality cao hơn.

## Kết luận

Multi-model là hướng đi thực tế cho edge deployment:

1. **Router model** (2B) — phân loại nhanh, không cần tool calling
2. **Agent model** (4B) — tool calling chuyên sâu, accuracy cao
3. **Embedding model** — memory/RAG, không dùng cho routing
4. **Synthesizer** (planned) — optional, chỉ khi cần báo cáo đẹp

Mỗi model chạy đúng việc nó giỏi. Tổng RAM ~5GB, vừa 8GB machine.
