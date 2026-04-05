# PRD: Pile - Trợ lý quản lý dự án phần mềm

> **Version**: 0.4
> **Author**: Tan Tran
> **Date**: 2026-04-05
> **Status**: In Development (V2 — Memory + RAG)

---

## 1. Tổng quan

### 1.1 Vấn đề

Quản lý dự án phần mềm theo Agile/Scrum đòi hỏi nhiều thao tác lặp đi lặp lại trên Jira và Git: tạo/cập nhật issues, theo dõi sprint, review code, tổng hợp báo cáo... Những công việc này tốn thời gian và dễ bị bỏ sót.

### 1.2 Giải pháp

Xây dựng hệ thống multi-agent chạy **hoàn toàn local**, sử dụng LLM self-hosted qua Ollama, có khả năng:

- Đọc/ghi Jira: quản lý issues, sprints, backlog
- Tương tác Git: theo dõi code changes, liên kết commit-issue
- Hỗ trợ Scrum ceremonies: standup, planning, retrospective
- Tổng hợp báo cáo sprint, velocity, burndown
- Giao tiếp bằng **Tiếng Việt** và **Tiếng Anh**
- Giao diện web (Chainlit) cho daily use

### 1.3 Ràng buộc

| Ràng buộc | Chi tiết |
|---|---|
| **Không dùng cloud LLM** | Toàn bộ inference chạy local |
| **LLM provider** | Ollama, OpenAI-compatible (LM Studio), hoặc Ollama native (single-agent only), configurable qua `.env` |
| **LLM model** | Mặc định `qwen3.5:9b` (model mặc định, hỗ trợ tool calling) |
| **Ngôn ngữ lập trình** | Python |
| **Framework** | Microsoft Agent Framework 1.0 |
| **UI** | Chainlit (localhost, self-hosted) |
| **Jira** | Jira Cloud (your-instance.atlassian.net), auth bằng API token |
| **Package manager** | uv |
| **Config** | Tất cả settings (Ollama host/model, Jira URL/token, Git repos) trong `.env` |

---

## 2. Users & Personas

### 2.1 Primary: Project Manager / Scrum Master

- Theo dõi sprint progress hàng ngày
- Tổng hợp báo cáo nhanh (standup, sprint review)
- Hỏi đáp bằng ngôn ngữ tự nhiên thay vì thao tác Jira thủ công
- Quản lý backlog, ưu tiên issues

### 2.2 Secondary: Developer / Tech Lead

- Xem issues được assign, trạng thái sprint
- Tổng hợp code changes liên quan đến issue
- Hỗ trợ estimation và planning

---

## 3. Functional Requirements

### 3.1 Jira Integration

| ID | Tính năng | Mô tả |
|---|---|---|
| J-01 | Tìm kiếm issues | Tìm, lọc issues theo project/sprint/assignee/status bằng JQL |
| J-02 | Xem chi tiết issue | Lấy thông tin đầy đủ: summary, description, status, assignee, comments |
| J-03 | Xem sprint | Sprint hiện tại, backlog, velocity, burndown data |
| J-04 | Tạo issue | Tạo mới issue với summary, description, type, priority, assignee |
| J-05 | Cập nhật issue | Chuyển trạng thái (transition), thêm comment |
| J-06 | Sprint report | Tổng hợp sprint progress: done, in-progress, blocked, remaining |
| J-07 | Backlog grooming | Gợi ý sắp xếp ưu tiên backlog dựa trên dependencies, business value |

### 3.2 Git Integration (optional)

Git là optional — nếu không config repos thì Git Agent sẽ không được tạo. Hỗ trợ private repos qua username+password hoặc API token.

| ID | Tính năng | Mô tả |
|---|---|---|
| G-01 | Commit history | Liệt kê commits theo branch/author/thời gian |
| G-02 | Branch management | Liệt kê, so sánh branches |
| G-03 | Commit-issue linking | Liên kết commits với Jira issues qua commit message pattern |
| G-04 | Code change summary | Tóm tắt thay đổi code trong branch/between refs |
| G-05 | Diff analysis | Xem chi tiết thay đổi giữa 2 commits/branches |
| G-06 | Private repo auth | Hỗ trợ token (GitHub PAT, GitLab PAT) hoặc username+password cho private repos |

### 3.3 Scrum Assistant

| ID | Tính năng | Mô tả |
|---|---|---|
| S-01 | Daily standup summary | Tổng hợp từ Jira + Git: yesterday/today/blockers cho từng member |
| S-02 | Sprint planning | Gợi ý capacity, story points dựa trên velocity lịch sử |
| S-03 | Retrospective insights | Phân tích sprint vừa qua: what went well, what didn't, action items |
| S-04 | Agile/Scrum knowledge | Trả lời câu hỏi methodology, best practices, coaching |
| S-05 | Velocity tracking | Phân tích trend velocity qua các sprints |
| S-06 | Jira data quality audit | Phát hiện issues thiếu thông tin: no description, no estimate, no assignee, no priority, missing acceptance criteria |
| S-07 | Proactive recommendations | Đề xuất cải thiện process, cảnh báo risks, gợi ý hành động dựa trên data patterns |
| S-08 | Timeline tracking | Theo dõi timeline sprint/release, so sánh actual vs plan, tính % hoàn thành theo thời gian |
| S-09 | Delay alerts | Cảnh báo trễ: issues quá due date, sprint burn rate chậm hơn plan, bottleneck tại stage cụ thể |
| S-10 | Blocker tracking | Theo dõi issues bị blocked: bị block bao lâu, block bởi gì, gợi ý escalation |
| S-11 | Workload balance | Phân tích workload từng member: ai overload, ai có capacity, gợi ý redistribute |
| S-12 | WIP limits | Cảnh báo khi member có quá nhiều issues In Progress đồng thời |
| S-13 | Cycle time analysis | Tính trung bình thời gian issue đi từ To Do → In Progress → Done, phát hiện stage nào chậm nhất |
| S-14 | Sprint goal tracking | Theo dõi tiến độ theo sprint goal (không chỉ đếm issues mà đánh giá mục tiêu sprint) |
| S-15 | Dependencies tracking | Phát hiện issue blocked by / blocks issue khác, cảnh báo dependency chain có risk |
| S-16 | Stakeholder summary | Tạo báo cáo ngắn gọn cho PO/manager: progress, risks, decisions needed |
| S-17 | Meeting preparation | Chuẩn bị agenda + data cho sprint planning, review, retrospective |

### 3.4 Memory & Knowledge Base (RAG)

Pile nhớ thông tin qua sessions và tham khảo tài liệu đã được nạp vào knowledge base. Toàn bộ chạy local — embedding qua Ollama, vector store bằng ChromaDB embedded.

| ID | Tính năng | Mô tả |
|---|---|---|
| M-01 | Remember | Lưu thông tin vào long-term memory (decisions, patterns, notes) |
| M-02 | Forget | Xoá memory item theo ID, cần user approval |
| M-03 | Semantic search | Tìm kiếm ngữ nghĩa trong memories và documents |
| M-04 | Ingest document | Nạp PDF/markdown/text vào knowledge base (chunk + embed), cần user approval |
| M-05 | List documents | Liệt kê tài liệu đã nạp |
| M-06 | Remove document | Xoá tài liệu khỏi knowledge base, cần user approval |
| M-07 | Knowledge-augmented answers | Scrum Agent tự query knowledge base khi tư vấn methodology |
| M-08 | File upload (Chainlit) | Drag & drop hoặc attach PDF/markdown/text qua web UI → tự động ingest |

**Ràng buộc bổ sung:**

| Ràng buộc | Chi tiết |
|---|---|
| **Vector store** | ChromaDB embedded mode, persist tại `~/.pile/chromadb/` |
| **Embedding model** | `nomic-embed-text` qua Ollama (cùng server với LLM) |
| **Document formats** | PDF, Markdown, Plain text |
| **Chunking** | ~500 chars/chunk, recursive split theo paragraph/sentence, 50-char overlap |
| **Tích hợp** | Tool-based retrieval (không inject context tự động — tiết kiệm token cho model nhỏ) |
| **Approval** | Forget, ingest, remove document require human approval |

### 3.5 Browser (Web Scraping)

Khi API không khả dụng, Pile truy cập Jira, GitHub, GitLab qua browser (Playwright + Firefox). Auto-login từ credentials trong `.env`, persistent session sau lần đầu.

| ID | Tính năng | Mô tả |
|---|---|---|
| B-01 | Browse URL | Mở trang web, trả về title + clean text content |
| B-02 | Read element | Extract text từ CSS selector cụ thể |
| B-03 | Click | Click element bằng selector hoặc visible text |
| B-04 | Fill form | Điền form field |
| B-05 | Manual login | Mở headed Firefox để user login thủ công (fallback) |
| B-06 | Screenshot | Chụp screenshot debug |
| B-07 | Auto-login | Tự đăng nhập Jira/GitHub/GitLab từ credentials trong .env |
| B-08 | Persistent session | Giữ login session qua restarts (Firefox profile) |

**Ràng buộc bổ sung:**

| Ràng buộc | Chi tiết |
|---|---|
| **Browser** | Firefox via Playwright (headless mặc định) |
| **Profile** | Persistent tại `~/.pile/browser/` |
| **Auto-login** | Jira (Atlassian ID), GitHub, GitLab — detect login page + fill credentials |
| **Content** | Clean text, truncated 4000 chars — tối ưu cho model 9B |
| **Approval** | Chỉ `browser_login` (manual) cần user approval |

### 3.6 Metrics Visualization

Tự động render interactive charts khi agent output chứa số liệu. Post-processing detection — không cần agent gọi tool riêng.

| ID | Tính năng | Mô tả |
|---|---|---|
| V-01 | Status distribution | Pie/donut chart cho Done/In Progress/To Do counts |
| V-02 | Workload balance | Horizontal bar chart cho issues/points per member |
| V-03 | Sprint velocity | Bar chart cho story points per sprint |
| V-04 | Cycle time | Bar chart cho thời gian per stage (days) |
| V-05 | Auto-detect | Regex patterns trên agent text output → tự build chart |

**Ràng buộc:**

| Ràng buộc | Chi tiết |
|---|---|
| **Chart library** | Plotly (interactive, dark theme) |
| **UI** | Chainlit only (CLI không hỗ trợ) |
| **Detection** | Post-processing, transparent — không sửa agent/tools |

### 3.7 Ngôn ngữ

| ID | Tính năng | Mô tả |
|---|---|---|
| L-01 | Bilingual | Hiểu và trả lời Tiếng Việt + Tiếng Anh |
| L-02 | Auto-detect | Tự nhận diện ngôn ngữ input và trả lời cùng ngôn ngữ |

### 3.8 User Interface

| ID | Tính năng | Mô tả |
|---|---|---|
| U-01 | Web chat UI | Chainlit interface tại localhost:8000 với quick action starters |
| U-02 | Streaming responses | Real-time streaming với agent step visualization |
| U-03 | Session history | Lưu lịch sử hội thoại trong session |
| U-04 | CLI | Terminal chat với slash commands (`/standup`, `/planning`) |

---

## 4. Non-Functional Requirements

| Yêu cầu | Target |
|---|---|
| **Response time** | < 30s truy vấn đơn giản, < 60s multi-agent workflow |
| **Availability** | Khả dụng khi Ollama server và Jira instance online |
| **Data privacy** | Không gửi data ra ngoài mạng nội bộ |
| **Extensibility** | Dễ thêm agent/tool mới (plugin architecture) |
| **Observability** | Log tất cả agent interactions, tool calls |
| **Error handling** | Graceful degradation khi Ollama/Jira unavailable |
| **Health checks** | Startup checks cho Ollama (server + model) và Jira (auth) |

---

## 5. Out of Scope

- CI/CD integration
- Notification/alerting tự động
- Multi-user authentication
- Confluence/wiki integration
- Code review automation (chỉ tóm tắt, không review)
- Cloud embedding/vector DB services

---

## 6. Success Metrics

| Metric | Target |
|---|---|
| Agent trả lời đúng câu hỏi về sprint/issues | > 80% |
| Standup summary generation time | < 45s |
| Tool calling success rate (Jira + Git) | > 90% |
| User satisfaction | Hữu ích hơn thao tác Jira thủ công |

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Local LLM (9B) không đủ cho complex reasoning | Cao | **Chain-of-thought prompting** bắt buộc trong mọi system prompt, phân nhỏ task tối đa qua multi-agent, few-shot examples. Nâng model nếu cần. |
| Tool calling quality với model nhỏ | Trung bình | Benchmark đã confirm qwen3.5:9b tool calling ổn định. Retry logic cho edge cases |
| Jira API rate limiting | Thấp | Response caching, batch requests |
| Tiếng Việt generation quality | Trung bình | Prompt engineering, few-shot examples trong system prompt |
| Jira write operations gây lỗi | Cao | **Tất cả** write operations require human approval via `@tool(approval_mode="always_require")` |
| Ollama server unavailable | Trung bình | Health check on startup, clear error message |

### Model: `qwen3.5:9b`

Đã benchmark ngày 2026-04-05. `qwen3.5:9b` hỗ trợ tool calling ổn định. Nếu không đủ thông minh cho reasoning phức tạp → nâng lên bản lớn hơn qua `.env`.

### Chiến lược bù đắp model nhỏ

Do dùng model 9B, **chain-of-thought** và **phân task** là cốt lõi:

1. **Chain-of-thought trong mọi system prompt**: Yêu cầu agent "think step by step" trước khi đưa kết luận
2. **Mỗi agent chỉ làm 1 việc**: Triage chỉ route, Jira chỉ query/write, Git chỉ đọc repo, Scrum chỉ tổng hợp
3. **Task decomposition**: Câu hỏi phức tạp được Triage Agent chia nhỏ → handoff lần lượt → tổng hợp
4. **Few-shot examples trong prompt**: Mỗi agent có 2-3 ví dụ input/output mẫu
5. **Structured output**: Agent trả về dạng có cấu trúc (bullet points, tables) thay vì free-form text
