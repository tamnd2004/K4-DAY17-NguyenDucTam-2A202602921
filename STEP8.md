# Phân tích kết quả benchmark — Day 17 Memory Systems

Toàn bộ số liệu dưới đây lấy từ chế độ **offline (tất định)**, chạy trên trạng thái sạch. Output gốc nằm trong `results/`:

| File | Lệnh tạo ra |
|---|---|
| `results/benchmark_offline.txt` | `python src/benchmark.py --details` (cấu hình nộp bài) |
| `results/ablation_no_compact.txt` | `COMPACT_THRESHOLD_TOKENS=1000000 python src/benchmark.py` (tắt compact) |
| `results/ablation_no_confidence_threshold.txt` | `PROFILE_MIN_CONFIDENCE=0 python src/benchmark.py` (tắt bonus) |

Hai lần chạy liên tiếp trên `state/` sạch cho output giống hệt nhau (đã `diff`).

## 1. Cách chạy lại

```bash
rm -rf state                      # PowerShell: Remove-Item -Recurse -Force state
python src/benchmark.py           # 2 bảng: Standard + Long-Context Stress
pytest src/test_agents.py -v      # 7 test, không cần API key
```

Benchmark mặc định chạy offline và bỏ qua mọi API key. Cờ `--live` dùng LLM thật, `--judge` chấm Response quality bằng judge model, `--details` in từng câu hỏi recall kèm câu trả lời. Benchmark tự xóa `state/profiles/<user>/User.md` trước mỗi bộ dữ liệu, nên số liệu không phụ thuộc lần chạy trước.

**Quy ước biến môi trường** (`.env` ở root, mẫu trong `.env.example`; tất cả đều tùy chọn):

| Biến | Ý nghĩa | Mặc định |
|---|---|---|
| `LLM_PROVIDER`, `LLM_MODEL` | provider (`openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`) và model chính | `openai`, `gpt-4o-mini` |
| `LLM_TEMPERATURE`, `LLM_REQUESTS_PER_MINUTE` | nhiệt độ; giới hạn tốc độ gọi phía client | `0`; không giới hạn |
| `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_*` | judge model; nếu không đặt thì dùng chung cấu hình với `LLM_*` | = `LLM_*` |
| `OPENAI_API_KEY`, `GEMINI_API_KEY` (hoặc `GOOGLE_API_KEY`), `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` | key của provider tương ứng | — |
| `CUSTOM_BASE_URL` + `CUSTOM_API_KEY`, `OLLAMA_BASE_URL`, `OPENROUTER_BASE_URL` | endpoint; Ollama và `custom` chỉ chạy live khi có base URL | — |
| `COMPACT_THRESHOLD_TOKENS`, `COMPACT_KEEP_MESSAGES` | ngưỡng compact; số message giữ nguyên văn | `1000`, `4` |
| `PROFILE_MIN_CONFIDENCE` | ngưỡng tin cậy để ghi fact vào `User.md` (bonus) | `0.7` |

Bài làm đã chạy thử live với `gemini-3.5-flash-lite` ở 15 RPM: kiểm tra Baseline quên khi sang thread mới, Advanced recall qua `User.md`, và middleware tóm tắt được đếm vào Compactions. Chưa chạy trọn benchmark live (hơn 250 lệnh gọi ở 15 RPM), nên mọi bảng dưới đây là số offline.

## 2. Kết quả

**Standard Benchmark** — 10 hội thoại, 101 lượt, 14 câu hỏi recall

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 929 | 16990 | 0.000 | 0.000 | 0 | 0 |
| Advanced | 959 | 23945 | 1.000 | 1.000 | 360 | 0 |

**Long-Context Stress Benchmark** — 1 hội thoại, 16 lượt dài, 3 câu hỏi recall

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 196 | 22590 | 0.000 | 0.000 | 0 | 0 |
| Advanced | 175 | 11348 | 1.000 | 1.000 | 256 | 3 |

Tỷ lệ Advanced / Baseline về Prompt tokens processed: **1.41x** ở Standard, **0.50x** ở Stress.

Cách đo:
- **Agent tokens only** là token của câu trả lời agent.
- **Prompt tokens processed** là tổng ngữ cảnh gửi vào mỗi lượt, cộng dồn theo từng lượt. Baseline gồm system prompt và toàn bộ thread; Advanced gồm system prompt, `User.md`, summary và các message còn giữ nguyên văn.
- Token được ước lượng bằng `ceil(len / 4)`.
- **Recall** chấm 1 / 0.5 / 0 trên `expected_contains`, hỏi ở thread `<id>::recall` mới cho cả hai agent.
- **Response quality** là heuristic: 70% độ phủ fact, 15% độ ngắn gọn, 15% có bullet. Câu trả lời không chứa fact nào được 0 điểm.

## 3. Chuỗi logic đọc từ bảng

1. **Baseline không nhớ dài hạn.** Recall của Baseline bằng 0.000 ở cả hai bảng, và Memory growth bằng 0 vì nó không ghi file nào. Lý do: `BaselineAgent.sessions` khóa theo `thread_id`, nên thread `::recall` luôn bắt đầu trống. Baseline vẫn nhớ trong cùng thread; test `test_cross_session_recall` kiểm tra cả hai chiều.
2. **Advanced thêm `User.md` nên recall tăng.** Recall của Advanced là 1.000 ở cả hai bảng, tức đủ 33/33 chuỗi kỳ vọng ở Standard và 8/8 ở Stress. Memory growth dương: 360 B và 256 B.
3. **Hội thoại dài làm prompt cost tăng mạnh.** Prompt mỗi lượt của Baseline trong bảng stress tăng tuyến tính: 230 → 402 → … → 2537 ở lượt 16. Tổng cộng là 22.590 token, so với 16.990 token cho cả 101 lượt ngắn của bộ Standard.
4. **Compact kéo chi phí ngữ cảnh xuống.** Ở bảng stress, Advanced chỉ xử lý 11.348 prompt token (0.50x Baseline), với Compactions = 3. Thí nghiệm tắt compact cho thấy đây đúng là nguyên nhân: khi compact không chạy, con số này tăng lên 23.172 (1.03x Baseline) và Compactions về 0.
5. **Mạnh hơn nhưng phức tạp hơn.** Xem rủi ro ở mục 4.4 và phần bonus ở mục 5.

## 4. Bốn câu hỏi của Guide (Bước 8)

### 4.1 Vì sao Advanced có recall tốt hơn Baseline

- **Số liệu.** Cross-session recall là 1.000 so với 0.000 ở cả hai bảng. Riêng câu stress "nếu ai đó nhắc Huế, Hà Nội hay product manager…", Advanced vẫn trả lời "Đà Nẵng" và "MLOps engineer".
- **Cơ chế.** Mỗi lượt, `extract_profile_updates()` trích fact ổn định từ câu của người dùng. `UserProfileStore.upsert_fact()` ghi fact thành dòng `- key: value` trong `state/profiles/<user>/User.md`. Ở thread mới, `_offline_response()` đọc lại file qua `facts()` và trả lời theo khóa mà câu hỏi yêu cầu.
- **Tách hai đường.** Fact đi vào `User.md`, còn hội thoại đi vào compact memory. Nhờ vậy recall không phụ thuộc vào việc bản tóm tắt có giữ được tên hay không.
- **Công bằng.** Hai agent dùng chung một bộ sinh câu trả lời (`compose_offline_reply`). Baseline chỉ lấy fact từ các message trong thread hiện tại, Advanced lấy từ `User.md`. Chênh lệch recall vì vậy đến hoàn toàn từ memory, không đến từ cách trả lời.
- **Giới hạn.** Recall 1.000 là trên chính bộ câu hỏi đã dùng để thiết kế regex trích xuất. Gặp cách diễn đạt khác, ví dụ "tôi sinh sống tại…" hoặc tên viết thường, bộ trích sẽ bỏ sót. Bản live có tool `save_user_fact` để LLM tự ghi bổ sung. Trong lần chạy thử live, LLM đã dùng tool này để sửa nơi ở khi regex bỏ sót, nhưng bài chưa đo việc này thành số liệu.

### 4.2 Vì sao Advanced có thể tốn hơn ở hội thoại ngắn

- **Số liệu.** Ở bảng Standard, Prompt tokens processed của Advanced là 23.945 so với 16.990 (1.41x). Agent tokens only là 959 so với 929 (+3%).
- **Cơ chế.** Mỗi lượt Advanced phải mang theo `User.md`. Cộng qua 101 lượt, riêng `User.md` chiếm 6.853 token, tức gần trọn phần chênh 6.955 token. 102 token còn lại đến từ câu trả lời dài hơn: Advanced trả lời thật các câu hỏi xen giữa hội thoại như "Bạn có thể nhắc lại tên mình không?", còn Baseline chỉ nói "chưa có".
- **Vì sao compact không bù lại được.** Mỗi thread Standard chỉ khoảng 250–300 token, chưa chạm ngưỡng 1000, nên Compactions = 0. Ở hội thoại ngắn, Advanced chỉ có thêm chi phí mà chưa có khoản tiết kiệm nào.
- **Giới hạn.** Chi phí này tăng theo kích thước `User.md`, không theo độ dài thread. Profile càng lớn thì mọi lượt càng đắt; đây là lý do làm bonus ở mục 5.

### 4.3 Vì sao compact có lợi thế ở hội thoại dài

- **Số liệu.** Ở bảng stress, Prompt tokens processed là 11.348 so với 22.590 (0.50x). Agent tokens only gần như không đổi: 175 so với 196.
- **Compact tối ưu đúng cột Prompt tokens processed, không phải Agent tokens only.** Thí nghiệm tắt compact cho thấy Agent tokens only của Advanced vẫn là 175, trong khi Prompt tokens processed tăng từ 11.348 lên 23.172. Compact chỉ thu nhỏ phần ngữ cảnh mang vào mỗi lượt. Phần agent sinh ra do bộ sinh câu trả lời quyết định.
- **Cơ chế.** `CompactMemoryManager.append()` nén khi summary cộng các message vượt 1000 token. Lúc đó mọi message trừ 4 message gần nhất được thay bằng `summarize_messages()`: một dòng chủ đề tối đa 20 tên riêng (NASA, Artemis III, X-59, WMO, El Nino…) và ý chính của 6 lượt người dùng gần nhất. Summary cũ được gộp vào summary mới, nên dài khoảng 195 token và không lồng nhau.
- **Diễn biến từng lượt.** Prompt mỗi lượt của Advanced có dạng răng cưa: 257 → 914, xuống 530 sau lần nén 1 ở lượt 6, rồi các lần nén ở lượt 10 và 14, đỉnh không vượt 1014. Baseline tăng thẳng tới 2537.
- **Phân rã 11.348 token của Advanced:**

  | Thành phần | Token |
  |---|---|
  | Message gần nhất | 8.096 |
  | Summary | 1.841 |
  | `User.md` | 723 |
  | System prompt | 688 |

- **Giới hạn.** Summary là heuristic, nên mất chi tiết. Nó giữ "X-59" và "Mach" nhưng không giữ con số "Mach 1.1" hay độ cao 29.500 feet. Một câu follow-up về chi tiết cũ ở lượt 3 sẽ không trả lời được. Lợi thế 0.50x đổi lấy độ chính xác của ngữ cảnh cũ.

### 4.4 File memory tăng trưởng ra sao và rủi ro gì

- **Số liệu.** Memory growth là 360 B (Standard, 10 phiên) và 256 B (Stress, 16 lượt dài). Kích thước `User.md` sau từng phiên Standard: 226 → 234 → 275 → 317 → 362 → 360 → 360 → 360 → 360 → 360 B. File phình ở 5 phiên đầu khi có fact mới, rồi đứng yên.
- **Cơ chế giữ file nhỏ.** `User.md` lưu fact chứ không lưu log hội thoại. Fact một giá trị như nơi ở hay nghề được thay tại chỗ qua `edit_text()`. Vì vậy ở phiên 6, khi "backend engineer" thành "MLOps engineer", file còn giảm 2 byte. Bảng stress có 16 lượt, gần 2.300 token người dùng và 3 lần compact, nhưng file vẫn chỉ 256 B: độ dài hội thoại đi vào compact memory, không đi vào `User.md`.
- **Rủi ro quan sát được:**
  - **Fact dạng danh sách chỉ tăng.** `style` và `interests` được gộp nhưng không bao giờ bị xóa. Ở bộ Standard, `style` có 7 mục ("ngắn gọn, rõ ý, có ví dụ thực tế, bullet, có ví dụ thực chiến, nhấn trade-off, có cấu trúc"), và mỗi mục đều bị mang vào mọi lượt.
  - **Lưu nhầm từ lượt nhiễu hoặc câu dặn một lần.** Khi chạy live, câu dặn một lần "Trả lời một câu ngắn" bị lưu vĩnh viễn thành style. Đây là bằng chứng cho bonus ở mục 5.
  - **Lỗi correction đã gặp trong quá trình làm.** Mệnh đề bắt đầu bằng phủ định ("…ra Đà Nẵng rồi, *không còn ở Huế* nữa") từng khiến Huế ghi đè lại Đà Nẵng. Lỗi đã sửa và có test `test_correction_replaces_old_fact`. Bài học: một regex sai là đủ để fact cũ sống lại.
  - **Summary vẫn chứa nhiễu.** Dòng chủ đề có "Hà Nội" và "Huế". An toàn ở bản offline vì fact chỉ đọc từ `User.md`, nhưng ở bản live LLM đọc cả summary, nên có thể nhầm nếu `User.md` thiếu fact.
  - **Quyền riêng tư.** `User.md` là văn bản thuần trên đĩa, chứa tên, nơi ở và thú cưng. Môi trường thật cần mã hóa, chính sách xóa, và cho người dùng xem hoặc sửa.

## 5. Bonus: Confidence threshold trước khi ghi `User.md`

**Vấn đề giải quyết.** Bộ trích fact cũ ghi mọi thứ khớp regex. Hai bằng chứng cho thấy nó quá tham:
- Một câu dặn một lần là "bạn **thử** nêu thêm một ví dụ số liệu minh họa" (conv-04) trở thành preference vĩnh viễn "có số liệu minh họa".
- "Mình **đang học thêm** về RAG và evaluation" (một việc tạm thời) trở thành interest vĩnh viễn.

Ở bản live, "Trả lời một câu ngắn" cũng bị lưu thành style.

**Cách làm** (`fact_confidence()` trong `src/memory_store.py`):
- Mỗi fact được chấm điểm theo câu chứa nó, và chỉ ghi khi điểm ≥ `PROFILE_MIN_CONFIDENCE` = 0.7.
- Fact định danh (tên, nơi ở, nghề, đồ uống, món ăn, thú cưng) bắt đầu ở 0.8, vì đến từ câu tự giới thiệu rõ ràng.
- Preference và interest bắt đầu ở 0.5, cộng 0.3 khi câu có tín hiệu lâu dài: "mình muốn/thích", "hãy", "ưu tiên", "dài hạn", "luôn", "nhớ".
- Mọi fact bị trừ 0.4 khi câu có tín hiệu do dự hoặc một lần: "thử", "có lẽ", "lần này", "một câu".
- Hai agent dùng cùng ngưỡng. Test `test_confidence_threshold_skips_one_off_preferences` khẳng định câu dặn một lần không vào `User.md`, nhưng sẽ lọt vào nếu tắt ngưỡng.

**Cải thiện đo được** (so với `results/ablation_no_confidence_threshold.txt`):

| Chỉ số (Advanced, Standard) | Ngưỡng 0 | Ngưỡng 0.7 | Thay đổi |
|---|---|---|---|
| Memory growth (bytes) | 405 | 360 | −11% |
| Token `User.md` mang vào prompt (101 lượt) | 7.542 | 6.853 | −9% |
| Prompt tokens processed | 24.791 | 23.945 | −3.4% |
| Cross-session recall | 1.000 | 1.000 | không đổi |

Trên toàn bộ dữ liệu, ngưỡng chỉ bỏ đúng 2 fact: `interests: RAG, evaluation` và `style: có số liệu minh họa`. Không fact nào cần cho recall bị mất. Bảng stress không đổi, vì mọi preference ở đó đều được nói kèm tín hiệu rõ ("Dài hạn: mình thích…", "Mình muốn…").

**Rủi ro tạo thêm:**
- **Bỏ sót fact thật (false negative).** Một preference thật mà người dùng chỉ nói một lần, với từ "thử" hoặc không có từ khóa nào, sẽ không bao giờ được nhớ. Recall của những fact đó về 0 mà benchmark không phát hiện được, vì bộ câu hỏi không hỏi tới.
- **Correction bị chặn.** Ngưỡng áp cho cả fact định danh. Một correction nói kiểu do dự, ví dụ "chắc là mình chuyển ra Đà Nẵng rồi", sẽ không được ghi, và `User.md` giữ fact cũ sai. Tức là ngưỡng có thể làm hỏng conflict handling.
- **Tín hiệu là danh sách từ khóa tiếng Việt cố định.** Diễn đạt khác hoặc ngôn ngữ khác sẽ lệch điểm.
- **Ngưỡng là một tham số cần chỉnh.** Đặt cao thì profile nghèo, đặt thấp thì quay lại vấn đề cũ.
- **Không tích lũy bằng chứng.** Bản hiện tại không nhớ một preference đã được nói hai lần ở hai phiên. Hướng tiếp theo là lưu số lần nhắc để kết hợp với memory decay, đổi lại phải lưu thêm state ngoài `User.md`.

**Các guardrail khác đã có sẵn trong bài** (thuộc nhóm Conflict handling và "tránh lưu sai khi người dùng hỏi" của Guide bước 9):
- Correction thay dòng cũ, mỗi khóa chỉ một giá trị.
- Lượt chỉ có câu hỏi không ghi gì.
- Câu đùa, câu điều kiện "Nếu…", mệnh đề "chỉ là…" và phần sau từ phủ định bị bỏ.
- Các trường hợp này có test `test_correction_replaces_old_fact` và `test_noise_and_questions_are_not_saved`.

## 6. Kiểm chứng và giới hạn của phép đo

**7 test** trong `src/test_agents.py` chạy trên `tmp_path`, không cần key, tổng cộng dưới 1 giây. Để chứng minh test bắt được lỗi thật, đã gài tạm ba lỗi và chạy lại:

| Lỗi gài vào | Test chuyển đỏ |
|---|---|
| Baseline lưu session theo `user_id` | cross-session recall, prompt load |
| Advanced không ghi `User.md` | cross-session recall, correction |
| Compact không bao giờ chạy | compact trigger, prompt load |

**Giới hạn của phép đo:**
- Token là ước lượng `len / 4`, không phải tokenizer thật.
- Response quality là heuristic. Điểm 1.000 nghĩa là đủ fact, ngắn và có bullet, không phải câu trả lời tự nhiên; câu trả lời offline là template.
- Recall đo bằng tìm chuỗi con, nên không phạt câu trả lời có thêm fact cũ sai. Câu stress Q2 được kiểm tra thủ công qua `--details`.
- Số live sẽ khác: token lấy từ `usage_metadata` của provider, và summary do LLM viết.
