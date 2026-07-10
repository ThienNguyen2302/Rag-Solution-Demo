# Rag-Solution-Demo

This project is a small RAG learning demo built around _The Last Lecture_ by Randy Pausch. It shows how different retrieval strategies change the quality of the final context before the LLM answers the question.

## RAG là gì

RAG, hay Retrieval-Augmented Generation, là cách kết hợp giữa retrieval và generation:

1. Người dùng đặt câu hỏi.
2. Hệ thống tìm các đoạn văn liên quan trong tài liệu.
3. LLM dùng các đoạn đã truy xuất để tạo câu trả lời cuối cùng.

Điểm quan trọng của RAG là câu trả lời không chỉ dựa trên kiến thức có sẵn trong model, mà còn dựa trên dữ liệu thực tế được lấy từ tài liệu của bạn. Vì vậy, chất lượng retrieval ảnh hưởng trực tiếp đến chất lượng output.

## Các technique trong repo

Repo này đang có các demo chính sau:

- Answer expansion trong [expansion_with_answers/expansion_answers.py](expansion_with_answers/expansion_answers.py)
- Query expansion trong [expansion_with_queries/expansion_queries.py](expansion_with_queries/expansion_queries.py)
- Reranking bằng cross encoder trong [reranking/cross_encoder.py](reranking/cross_encoder.py)
- DPR không dùng vector DB trong [DPR/dense_passage_retrieval_no_vector_db.py](DPR/dense_passage_retrieval_no_vector_db.py)
- DPR có dùng vector DB trong [DPR/dense_passage_retrieval_with_vector_db.py](DPR/dense_passage_retrieval_with_vector_db.py)

## Cấu trúc thư mục

- `expansion_with_answers/`: demo mở rộng câu hỏi bằng một câu trả lời giả định hoặc diễn giải giàu ngữ nghĩa hơn.
- `expansion_with_queries/`: demo sinh nhiều truy vấn mở rộng để tăng khả năng tìm đúng chunk.
- `reranking/`: demo lấy ứng viên ban đầu rồi chấm điểm lại bằng cross encoder.
- `DPR/`: demo Dense Passage Retrieval, gồm cả bản không dùng vector DB và bản có dùng vector DB.

## Cài đặt

Tạo môi trường Python rồi cài dependencies theo đúng technique bạn muốn chạy:

```bash
pip install -r expansion_with_answers/docker/requirement.txt
pip install -r expansion_with_queries/docker/requirement.txt
pip install -r reranking/docker/requirement.txt
pip install -r DPR/docker/requirements.txt
```

Nếu bạn chỉ chạy một demo, chỉ cần cài file requirement tương ứng của demo đó. Ví dụ:

- Answer expansion: [expansion_with_answers/docker/requirement.txt](expansion_with_answers/docker/requirement.txt)
- Query expansion: [expansion_with_queries/docker/requirement.txt](expansion_with_queries/docker/requirement.txt)
- Reranking: [reranking/docker/requirement.txt](reranking/docker/requirement.txt)
- DPR: [DPR/docker/requirements.txt](DPR/docker/requirements.txt)

Nếu bạn dùng Ollama, hãy đảm bảo đã cấu hình đúng biến môi trường trong `.env` hoặc môi trường shell, đặc biệt là `OLLAMA_HOST`.

## Cách chạy từng technique

Chạy từ thư mục root `d:\RAG`.

### 1. Answer expansion

```bash
python expansion_with_answers/expansion_answers.py
```

Script này sẽ đọc PDF trong `expansion_with_answers/data/`, sinh phần mở rộng cho câu hỏi, rồi truy xuất và tạo câu trả lời cuối cùng.

### 2. Query expansion

```bash
python expansion_with_queries/expansion_queries.py
```

Script này tạo nhiều truy vấn mở rộng, truy xuất theo từng query rồi hợp nhất kết quả để tăng recall.

### 3. Reranking bằng cross encoder

```bash
python reranking/cross_encoder.py
```

Script này lấy top ứng viên bằng similarity search trước, sau đó dùng cross encoder để sắp xếp lại và chọn context tốt hơn.

### 4. DPR không dùng vector DB

```bash
python DPR/dense_passage_retrieval_no_vector_db.py
```

Đây là bản DPR cơ bản, encode query và passage rồi so sánh trực tiếp bằng cosine similarity.

### 5. DPR có dùng vector DB

```bash
python DPR/dense_passage_retrieval_with_vector_db.py
```

Đây là bản DPR có lưu embedding passage vào TurboQuantIndex để tìm kiếm nhanh hơn.

## Dữ liệu

Các demo đều làm việc với file PDF `_The Last Lecture_` trong thư mục `data/` tương ứng của từng technique. Nếu script báo lỗi không tìm thấy file, hãy kiểm tra lại tên và vị trí của PDF trong đúng folder.

## Ghi chú

- Một số script sẽ yêu cầu nhập câu hỏi trực tiếp trong terminal.
- Một số script dùng thư viện `turbovec`, `sentence-transformers`, `transformers`, `ollama`, và `pypdf`.
- Nếu muốn hiểu chi tiết từng technique, xem các file `theory.md` trong từng thư mục tương ứng.
