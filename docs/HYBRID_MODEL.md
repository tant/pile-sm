# Hybrid Model Architecture: 2 Small Models > 1 Large Model

> **Status**: Research / Experiment
> **Date**: 2026-04-06
> **Context**: Pile Scrum Master agent, 8GB RAM, Apple Silicon

---

## Ý tưởng

Thay vì chạy 1 model lớn (9B-14B), phối hợp 2 model nhỏ chuyên biệt chạy song song trên cùng máy. Mỗi model phát huy thế mạnh riêng.

## Tại sao?

- Model 9B cần ~6GB RAM → chiếm gần hết 8GB, không còn chỗ cho context dài
- Model 4B + model 2B = ~5GB tổng → dư RAM cho context + OS
- Mỗi model có thế mạnh khác nhau: Qwen giỏi tool calling, Gemma giỏi ngôn ngữ + multimodal
- LM Studio trên Apple Silicon serve nhiều model cùng lúc qua unified memory

## Cặp đôi đề xuất

| Role | Model | RAM | Thế mạnh |
|---|---|---|---|
| **Tool Executor** | Qwen 3.5-4B | ~3GB | TAU2: 79.9 (tool calling cực tốt, hơn nhiều model 70B+) |
| **Synthesizer** | Gemma 4 E2B | ~2GB | 128K context, multilingual, multimodal (image+audio) |
| **Tổng** | | **~5GB** | Vừa 8GB, dư cho OS + context |

## Flow

```
User: "Sprint hiện tại tiến độ thế nào? Phân tích chi tiết."
  │
  ├── [1] Keyword Router (regex, 0ms)
  │     → "sprint" → SprintAgent
  │
  ├── [2] Qwen 3.5-4B (Tool Executor)
  │     System: "You are a sprint specialist. Call tools to get data."
  │     → jira_get_board() → raw data (board + sprint + counts)
  │     → jira_get_sprint_issues() → raw data (issues by status)
  │     Output: structured data / bullet points
  │
  └── [3] Gemma 4 E2B (Synthesizer) — optional, chỉ khi cần
        System: "Analyze this data and write a clear Vietnamese report."
        Input: raw data from step 2
        Output: báo cáo đẹp, có phân tích, insights
```

## Khi nào dùng Synthesizer?

Không phải query nào cũng cần 2 model. Criteria:

| Query type | Model dùng | Lý do |
|---|---|---|
| "có mấy board" | Qwen only | Tool call → trả kết quả đơn giản |
| "tạo bug: Login crash" | Qwen only | Tool call → confirm |
| "cho tôi lệnh curl" | Qwen only | Tool call → trả string |
| "phân tích sprint chi tiết" | Qwen + Gemma | Cần data (Qwen) + phân tích sâu (Gemma) |
| "tổng hợp standup cho team" | Qwen + Gemma | Multi-tool data + synthesis |
| "đọc screenshot Jira board" | Gemma only | Multimodal (image input) |

## Ưu điểm

1. **Tổng RAM thấp (~5GB)** — chạy thoải mái trên 8GB machine
2. **Mỗi model chỉ làm việc nó giỏi** — Qwen không cần viết báo cáo dài, Gemma không cần gọi tool
3. **Parallel loading** — LM Studio Apple Silicon serve song song, không cần swap
4. **Fallback graceful** — nếu Gemma down, Qwen vẫn trả kết quả (chỉ kém đẹp)
5. **Upgradable** — thay Gemma E2B bằng E4B khi có thêm RAM, không đổi architecture

## Nhược điểm

1. **2 LLM calls** — chậm hơn 1 call (nhưng mỗi call nhỏ → nhanh hơn 1 call lớn)
2. **Phức tạp hơn** — cần logic quyết định khi nào cần synthesizer
3. **Context transfer** — phải pass data từ model 1 sang model 2
4. **Gemma 4 E2B tool calling kém** (TAU2: 24.5) — đừng bao giờ cho nó gọi tool

## Benchmark tham khảo

### Tool Calling (TAU2-Bench)

| Model | TAU2 Score | Vai trò trong hybrid |
|---|---|---|
| Qwen 3.5-4B | **79.9** | Tool executor |
| Qwen 3.5-2B | 48.8 | Backup executor (nếu cần tiết kiệm hơn) |
| Gemma 4 E4B | 42.2 | KHÔNG dùng cho tool calling |
| Gemma 4 E2B | 24.5 | KHÔNG dùng cho tool calling |

### General (khi không cần tool)

| Model | Multilingual | Context | Multimodal | Vai trò |
|---|---|---|---|---|
| Gemma 4 E2B | Tốt | 128K | Image+Video+Audio | Synthesizer, tóm tắt, phân tích |
| Qwen 3.5-4B | Tốt | 32K | Không | Tool executor |

## Implementation sketch

```python
# config.py
tool_model: str = "qwen3.5-4b-mlx"        # Fast tool calling
synthesis_model: str = "gemma-4-e4b-it"     # Rich text synthesis (optional)
synthesis_enabled: bool = True               # Disable to use tool_model for everything

# workflow
async def run_with_synthesis(query, agent, tools_result):
    """Optional: pass tool results through synthesis model for richer output."""
    if not settings.synthesis_enabled:
        return tools_result

    # Only synthesize for complex queries (reports, analysis)
    if len(tools_result) < 200:
        return tools_result  # Simple results don't need synthesis

    synthesis_client = create_client(model=settings.synthesis_model)
    response = await synthesis_client.run(
        f"Analyze this data and write a clear report:\n\n{tools_result}",
    )
    return response.text
```

## Mở rộng: 3 model combo (16GB RAM)

Nếu có 16GB:

| Role | Model | RAM |
|---|---|---|
| **Router** | Qwen 3.5-0.8B | ~1GB |
| **Tool Executor** | Qwen 3.5-9B | ~6GB |
| **Synthesizer** | Gemma 4 E4B | ~5GB |
| **Embedding** | nomic-embed-text | ~0.3GB |
| **Tổng** | | ~12.3GB |

Router 0.8B phân loại intent → Executor 9B gọi tool chính xác → Synthesizer E4B viết báo cáo đẹp. Tối ưu nhất cho cả speed và quality.

## Kết luận

Hybrid model là hướng đi thực tế cho edge deployment. Thay vì chạy đua parameter count, phối hợp đúng model đúng việc. Qwen cho structured tasks, Gemma cho creative/analytical tasks. Cả hai chạy song song trên cùng Apple Silicon unified memory.

**Next step**: Test Qwen 3.5-4B single model trước → nếu output quality cần cải thiện → thêm Gemma synthesizer layer.
