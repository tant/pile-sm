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

---

## Ghi chú

- Thứ tự không phải priority — sẽ đánh giá lại sau khi v1 chạy thực tế 1-2 sprint
- Phụ thuộc vào: MAF có đáng dùng không, model 9B có đủ thông minh không, workflow patterns nào hoạt động tốt
- Mỗi mục trên có thể trở thành 1 epic riêng hoặc bị bỏ hoàn toàn
