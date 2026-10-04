# Step-Back Prompting trong RAG

## Step-Back Prompting là gì?

Step-back prompting là kỹ thuật yêu cầu LLM tạm lùi lại khỏi một câu hỏi cụ thể để tạo ra một câu hỏi tổng quát hơn về nguyên lý, chủ đề hoặc bối cảnh liên quan.

Thay vì tìm kiếm trực tiếp bằng câu hỏi quá chi tiết, hệ thống dùng câu hỏi tổng quát này để truy xuất kiến thức nền. Sau đó, LLM sử dụng context đã truy xuất để trả lời câu hỏi ban đầu.

Ví dụ:

```text
Câu hỏi cụ thể:
Why did Randy Pausch create the First Penguin Award?

Câu hỏi step-back:
What role does failure play in learning and personal growth?
```

Câu hỏi ban đầu hỏi về một chi tiết cụ thể. Câu hỏi step-back mở rộng lên chủ đề lớn hơn là thất bại, việc học và sự phát triển cá nhân.

## Vì sao Step-Back Prompting hữu ích?

Một câu hỏi cụ thể đôi khi chứa quá ít ngữ cảnh để semantic search tìm được đoạn văn phù hợp. Tài liệu có thể không nhắc lại đúng tên riêng hoặc cách diễn đạt trong câu hỏi, nhưng lại giải thích nguyên lý tổng quát đứng phía sau.

Step-back prompting giúp hệ thống:

- Tìm được kiến thức nền liên quan đến câu hỏi.
- Giảm sự phụ thuộc vào từ khóa quá cụ thể.
- Kết nối câu hỏi chi tiết với chủ đề lớn hơn trong tài liệu.
- Cải thiện context trước khi tạo câu trả lời cuối cùng.

Ví dụ, câu hỏi `Why did he create the First Penguin Award?` có thể chỉ khớp với một vài đoạn chứa đúng tên giải thưởng. Câu hỏi step-back về vai trò của thất bại trong học tập có thể tìm được thêm các đoạn nói về rủi ro, sai lầm, bài học và sự kiên trì.

## Quy trình hoạt động

```text
Câu hỏi cụ thể của người dùng
            |
            v
LLM tạo một câu hỏi tổng quát hơn
            |
            v
Vector search bằng câu hỏi step-back
            |
            v
Các chunk liên quan được truy xuất
            |
            v
LLM trả lời câu hỏi ban đầu
```

Trong file `step_back.py`, pipeline gồm các bước:

1. Đọc file PDF của _The Last Lecture_.
2. Chia văn bản thành các chunk bằng `RecursiveCharacterTextSplitter`.
3. Chia tiếp các chunk theo giới hạn token.
4. Tạo embedding cho các đoạn văn bằng `all-MiniLM-L6-v2`.
5. Lưu các embedding vào `TurboQuantVectorStore`.
6. Nhận câu hỏi cụ thể từ người dùng.
7. Gọi Ollama để tạo một câu hỏi step-back tổng quát hơn.
8. Dùng câu hỏi step-back để truy xuất top 5 chunk.
9. Gửi câu hỏi ban đầu và context đã truy xuất cho LLM.
10. In ra câu trả lời cuối cùng.

Điểm quan trọng là hệ thống **truy xuất bằng câu hỏi step-back nhưng trả lời câu hỏi gốc**.

## Hai loại câu hỏi trong pipeline

### Câu hỏi gốc

Đây là câu hỏi người dùng thực sự muốn được trả lời. Nó có thể rất cụ thể, chứa tên người, sự kiện, giải thưởng hoặc một chi tiết trong tài liệu.

Câu hỏi này được giữ lại để gửi cho hàm `generate_answer()`.

### Câu hỏi step-back

Đây là câu hỏi do LLM tạo ra. Nó không trả lời câu hỏi gốc mà chuyển câu hỏi sang mức khái niệm rộng hơn.

Câu hỏi này được dùng cho vector search trong:

```python
vector_store.similarity_search(step_back_question, k=5)
```

Nếu LLM trả về markdown code fence, hàm `parse_step_back_question()` sẽ loại bỏ phần định dạng đó trước khi dùng kết quả.

## Step-Back khác Query Expansion như thế nào?

### Query Expansion

Query Expansion tạo ra nhiều cách diễn đạt khác nhau nhưng vẫn giữ nguyên cùng một ý định tìm kiếm.

```text
Câu hỏi gốc
    |
    v
Query 1, Query 2, Query 3, Query 4
    |
    v
Nhiều lần tìm kiếm
```

Ví dụ:

- What did Randy Pausch say about failure?
- How did Randy Pausch discuss making mistakes?
- What did _The Last Lecture_ teach about overcoming failure?

Mục tiêu chính là tăng recall bằng cách tìm kiếm nhiều cách diễn đạt.

### Step-Back Prompting

Step-back prompting tạo ra một câu hỏi duy nhất ở mức trừu tượng hơn.

```text
Câu hỏi cụ thể
    |
    v
Một câu hỏi tổng quát về nguyên lý
    |
    v
Một lần tìm kiếm chính
```

Mục tiêu chính là lấy được kiến thức nền và nguyên lý liên quan đến câu hỏi chi tiết.

| Tiêu chí | Query Expansion | Step-Back Prompting |
|---|---|---|
| Số lượng query | Nhiều query | Một query tổng quát |
| Cách thay đổi | Đổi cách diễn đạt | Lùi lên mức khái niệm rộng hơn |
| Mục tiêu | Tăng recall | Tìm kiến thức nền |
| Chi phí retrieval | Nhiều lần search | Một lần search chính |
| Rủi ro | Có thể sinh query nhiễu | Có thể quá tổng quát |

## Step-Back khác Decomposition như thế nào?

Decomposition chia một câu hỏi phức tạp thành nhiều câu hỏi nhỏ độc lập.

```text
Câu hỏi phức tạp
    |
    v
Câu hỏi nhỏ 1, câu hỏi nhỏ 2, câu hỏi nhỏ 3
```

Ví dụ:

```text
Câu hỏi gốc:
What were Pausch's lessons about dreams and failure?

Câu hỏi nhỏ:
1. What did he say about achieving childhood dreams?
2. What did he say about brick walls?
3. What did he say about learning from failure?
```

Step-back không chia câu hỏi thành nhiều phần. Nó tạo ra một câu hỏi tổng quát hơn để tìm nguyên lý chung.

- Decomposition phù hợp với câu hỏi có nhiều phần cần trả lời.
- Step-back phù hợp với câu hỏi chi tiết nhưng cần thêm bối cảnh hoặc kiến thức nền.

## Step-Back khác RAG Fusion và Reranking như thế nào?

### RAG Fusion

RAG Fusion tạo ra nhiều query, tìm kiếm cho từng query, sau đó hợp nhất các danh sách kết quả bằng RRF hoặc một chiến lược fusion khác.

Step-back chỉ tạo một query tổng quát và dùng query đó để retrieval. Vì vậy, step-back không cần RRF.

### Reranking

Reranking lấy một danh sách chunk ứng viên rồi dùng cross encoder hoặc mô hình khác để chấm điểm và sắp xếp lại.

Step-back thay đổi query trước khi retrieval. Nó không chấm điểm lại từng cặp query-document.

Hai kỹ thuật có thể kết hợp:

```text
Câu hỏi gốc
    |
    v
Tạo câu hỏi step-back
    |
    v
Retrieval bằng câu hỏi step-back
    |
    v
Cross encoder reranking
    |
    v
LLM trả lời câu hỏi gốc
```

Tuy nhiên, demo hiện tại chỉ tập trung vào Step-Back Prompting để giữ pipeline đơn giản.

## Ưu điểm

- Giúp truy xuất thông tin nền cho những câu hỏi quá cụ thể.
- Có thể tìm được các đoạn văn không dùng đúng từ khóa trong câu hỏi gốc.
- Pipeline đơn giản hơn multi-query hoặc decomposition.
- Dễ kết hợp với semantic search hiện có.
- Giữ câu hỏi gốc để LLM tạo câu trả lời cuối cùng.

## Hạn chế

- Chất lượng phụ thuộc vào câu hỏi step-back do LLM tạo ra.
- Nếu câu hỏi được tổng quát hóa quá mức, kết quả có thể bị lan sang chủ đề khác.
- Một câu hỏi step-back có thể bỏ sót chi tiết chỉ xuất hiện trong câu hỏi gốc.
- Demo hiện tại chỉ search bằng câu hỏi step-back, không search bổ sung bằng câu hỏi gốc.
- Cần thêm reranking hoặc kết hợp hai kết quả nếu dữ liệu lớn và yêu cầu độ chính xác cao.

## Cách chạy

Từ thư mục root của repo:

```bash
pip install -r step_back/docker/requirement.txt
python step_back/step_back.py
```

Đặt file PDF tại:

```text
step_back/data/last_lecture.pdf
```

Ngoài ra cần có Ollama server đang chạy và model `qwen3:0.6b`.

## Tóm tắt

Step-back prompting là kỹ thuật chuyển một câu hỏi cụ thể thành một câu hỏi tổng quát hơn trước khi retrieval.

```text
Specific Question
    -> Step-Back Question
    -> Retrieve Background Context
    -> Answer Original Question
```

So với các kỹ thuật khác:

- Query Expansion tạo nhiều cách diễn đạt.
- Decomposition chia câu hỏi thành nhiều câu hỏi nhỏ.
- RAG Fusion hợp nhất nhiều danh sách kết quả.
- Reranking sắp xếp lại các chunk ứng viên.
- Step-Back Prompting tìm kiến thức nền bằng một câu hỏi ở mức khái niệm rộng hơn.
