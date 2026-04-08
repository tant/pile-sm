# Roadmap: Pile

> Ghi nhận ý tưởng phát triển dài hơi. Chưa commit timeline — sẽ đánh giá lại sau khi v1 chạy thực tế.

---

## V1 — Scrum Assistant (done)

Reactive Q&A: user hỏi → agent trả lời.

- 4 agents: Triage, Jira, Git (optional), Scrum
- Jira read + write (HITL approval via `@tool`)
- Git read-only (5 tools, private repo support)
- 17 Scrum Master features
- Chainlit UI (agent steps, starters) + CLI (slash commands)
- 3 workflow patterns: Handoff, Sequential, GroupChat
- Ollama + LM Studio + Ollama native support
- Health checks on startup
- 43 unit tests

---

## Proactive Mode

Pile tự chạy định kỳ, không cần ai hỏi.

- Mỗi sáng tự tổng hợp standup summary
- Tự scan Jira hàng ngày: cảnh báo issues stuck, overdue, thiếu data
- Cuối sprint tự generate retrospective draft
- Gửi kết quả qua Slack hoặc email
- Cron/scheduler local

## V2 — Memory + RAG + Browser (in progress)

Nhớ context qua sessions, học từ project documents, truy cập web khi API không khả dụng.

**Memory + RAG:**
- ChromaDB embedded (local, persist to `~/.pile/chromadb/`)
- Embedding qua Ollama (`nomic-embed-text`)
- 6 memory tools: remember, forget, search, ingest document, list/remove documents
- Triage Agent xử lý trực tiếp memory ops (không cần Memory Agent riêng)
- Scrum Agent có `memory_search` để tra knowledge base khi tư vấn
- Hỗ trợ PDF, markdown, plain text
- Use case: nạp whitepapers (Scale Agile, SAFe), meeting notes → Pile tư vấn dựa trên tài liệu

**Browser:**
- Playwright + Firefox (headless mặc định, headed khi cần login)
- 6 browser tools: open, read, click, fill, login, screenshot
- Auto-login Jira/GitHub/GitLab từ credentials trong .env
- Persistent profile giữ session cookies qua restarts
- Triage xử lý trực tiếp, Scrum có read-only browser tools

## Multi-project

Hỗ trợ nhiều Jira project cùng lúc.

- Config nhiều project trong `.env` hoặc file config riêng
- Switch context: "Chuyển sang project ABC"
- So sánh metrics cross-project
- Tổng hợp báo cáo cho PM quản lý nhiều dự án

## Team Interaction

Từ single-user tool → team tool.

- Multi-user sessions: mỗi dev hỏi "task của tôi hôm nay?"
- Slack/Teams bot — team dùng trực tiếp trong chat
- Sprint planning session interactive: cả team + Pile cùng discuss
- Role-based responses: dev thấy khác PM thấy

## Deeper Git Intelligence

- PR summary: tóm tắt changes + risk assessment (không review code)
- Auto-link: PR merged → Jira issue tự transition
- Detect: branch tồn tại lâu không merge, conflict potential
- Commit pattern analysis: team commit frequency, hotspot files

## Metrics Dashboard

- Velocity chart, burndown, cycle time — render trong Chainlit hoặc export HTML
- Team health metrics qua thời gian
- Sprint-over-sprint comparison tự động
- Export PDF/image cho stakeholder report

## Process Automation

- Sprint kết thúc → tự tạo sprint mới, move carry-over issues
- Issue tạo thiếu field → suggest fill hoặc auto-fill defaults
- Code merged → auto transition Jira issue
- Stale branch cleanup suggestions

## Fine-tune Agent Model

LoRA fine-tune Qwen 3.5-4B cho Scrum Master domain — cải thiện tool calling accuracy và Vietnamese Scrum context.

### Phân tích hiện trạng

Kiến trúc hiện tại đã bù đắp tốt cho hạn chế model 4B (prefetch, loop detection, recovery, auto-learn). Fine-tune không phải yêu cầu cấp bách, nhưng có thể nâng chất lượng ở 3 điểm mà prompt engineering khó giải quyết triệt để:

| Vấn đề | Fine-tune giải quyết? | ROI |
|---|---|---|
| Tool calling format lỗi (JSON malformed, args sai type) | Có — win lớn nhất | Cao |
| Vietnamese Scrum terminology ("velocity", "sprint review", "quá tải") | Có — model hiểu context domain tốt hơn | Trung bình |
| Output format inconsistent (markdown tables, issue keys) | Có — model học pattern output chuẩn | Trung bình |
| Routing accuracy | Không cần — đã dùng Gemma riêng | — |
| Phân tích data Jira (prefetch mode) | Vừa — prompt engineering đã khá ổn | Thấp |

### Giai đoạn 1 — Thu thập Training Data

Chưa fine-tune. Tập trung xây pipeline thu thập và curate data chất lượng.

- Parse conversation logs từ `~/.pile/logs` thành training format (ChatML + tool calls)
- Log structured: mỗi turn ghi `(system_prompt, user_msg, tool_calls, tool_results, assistant_response)`
- Đặc biệt capture recovery cases: agent A fail → agent B thành công = correction signal
- Curate: loại bỏ turns lỗi, chuẩn hóa tool call format, đảm bảo mỗi example có expected output đúng
- Bổ sung synthetic data: dùng model lớn hơn generate thêm examples cho các case hiếm
- Mục tiêu: **500-2000 examples** chất lượng cao, đa dạng (tool calling + analysis + Vietnamese/English)

### Giai đoạn 2 — LoRA Fine-tune

Khi có đủ data, train LoRA adapter trên base model.

**Pipeline:**
```
Qwen3.5-4B base (HF, BF16, ~8GB)
  → LoRA fine-tune (unsloth hoặc axolotl)
  → Merge LoRA weights → base
  → Quantize Q4_K_M (llama.cpp)
  → Thay model trong ~/.pile/models/agent/
```

**Training config dự kiến:**
- LoRA rank: 16-32 (đủ cho domain adaptation, không quá nặng)
- Learning rate: 2e-4 → 5e-5 (cosine decay)
- Epochs: 3-5 (4B dễ overfit, cần early stopping)
- Train trên Mac M-series được (LoRA chỉ cần ~1-2GB VRAM thêm)

**Focus areas (theo priority):**
1. Tool calling accuracy — correct JSON format, đúng tool cho đúng query
2. Vietnamese Scrum terminology — hiểu "velocity", "sprint review", "standup" trong context VN
3. Output format — markdown tables, issue keys (TETRA-XX), structured responses

### Giai đoạn 3 — DPO Alignment (optional)

Nếu LoRA chưa đủ, thêm DPO (Direct Preference Optimization) để model "prefer" response style phù hợp.

- Tạo preference pairs: (chosen response, rejected response) cho cùng 1 query
- Nguồn: recovery logs (agent B thành công = chosen, agent A fail = rejected)
- DPO nhẹ hơn RLHF, không cần reward model riêng

### Rủi ro & Mitigation

| Rủi ro | Mitigation |
|---|---|
| Catastrophic forgetting — mất general reasoning | LoRA rank thấp (16-32), không full fine-tune, eval trên general benchmark trước/sau |
| Quantization loss — merge LoRA → re-quantize mất quality | So sánh perplexity trước/sau quantize, thử Q5_K_M nếu Q4 mất nhiều |
| Maintenance — mỗi Qwen version mới phải re-train | Giữ training pipeline reproducible, pin base model version |
| Overfitting — 4B model dễ overfit dataset nhỏ | Early stopping, validation split 80/20, dropout 0.05 |
| Eval khó — không có benchmark cho Vietnamese Scrum | Tự build eval set (~50-100 queries) với expected tool calls + response quality |

### Điều kiện bắt đầu

- [ ] Đã chạy production ít nhất 2-3 sprints (đủ log data thực tế)
- [ ] Có ít nhất 500 curated training examples
- [ ] Đã xác định top-5 failure patterns từ recovery logs
- [ ] Eval set sẵn sàng (50-100 queries + expected outputs)

---

## Ghi chú

- Thứ tự không phải priority — sẽ đánh giá lại sau khi v1 chạy thực tế 1-2 sprint
- Phụ thuộc vào: MAF có đáng dùng không, model 9B có đủ thông minh không, workflow patterns nào hoạt động tốt
- Mỗi mục trên có thể trở thành 1 epic riêng hoặc bị bỏ hoàn toàn
