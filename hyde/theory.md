# HyDE trong RAG

HyDE (Hypothetical Document Embedding) là kỹ thuật nâng cao chất lượng truy xuất trong RAG bằng cách tạo ra tài liệu giả định trả lời câu hỏi, sau đó sử dụng tài liệu giả định này để tìm kiếm các đoạn văn thực sự liên quan trong corpus.

## Ý tưởng chính

Trong RAG truyền thống, ta embed trực tiếp câu hỏi của người dùng và tìm kiếm các đoạn văn tương tự. Tuy nhiên, câu hỏi thường ngắn và sử dụng từ vựng khác biệt so với tài liệu trong corpus, dẫn đến hiệu suất retrieval kém.

HyDE giải quyết bài toán này bằng cách:

1. Sử dụng LLM để tạo ra một tài liệu giả định trả lời câu hỏi của người dùng
2. Nhúng tài liệu giả định này (có khả năng sử dụng từ vựng và phong cách tương tự như tài liệu thực trong corpus)
3. Tìm kiếm các đoạn văn thực trong corpus có độ tương đồng cao nhất với nhúng của tài liệu giả định
4. Sử dụng các đoạn văn thực được truy xuất để tạo ra câu trả lời cuối cùng

## Cách hoạt động trong code

Trong file `hyde_technique.py`, luồng xử lý đi theo các bước sau:

1. Đọc tài liệu PDF và trích xuất toàn bộ văn bản.
2. Chia văn bản thành các chunk nhỏ bằng `RecursiveCharacterTextSplitter`.
3. Khởi tạo `DPRQuestionEncoder` và `DPRContextEncoder` từ model `facebook/dpr-question_encoder-single-nq-base` và `facebook/dpr-ctx_encoder-single-nq-base`.
4. Nhận câu hỏi từ người dùng.
5. Sử dụng LLM (qwen3:0.6b qua Ollama) để tạo ra tài liệu giả định trả lời câu hỏi.
6. Tokenize và encode tài liệu giả định để lấy hypothetical document embedding.
7. Tokenize từng chunk và encode thành context embedding.
8. Dùng `cosine_similarity` để so sánh hypothetical document embedding với toàn bộ context embeddings.
9. Sắp xếp kết quả và lấy ra top 5 chunk giống nhất.
10. Sử dụng các chunk được truy xuất như là ngữ cảnh để tạo ra câu trả lời cuối cùng qua LLM.

## Vì sao HyDE hiệu quả

HyDE cải thiện chất lượng retrieval bằng cách giảm bù khoảng cách về từ vựng và phong cách giữa câu hỏi và tài liệu trong corpus:

- Tài liệu giả định được tạo ra bởi LLM có xu hướng sử dụng từ vựng và phong cách tương tự như các tài liệu trong corpus hơn so với câu hỏi gốc ngắn gọn
- Điều này giúp embedding của tài liệu giả định có độ tương đồng cao hơn với các embedding của tài liệu thực trong corpus
- HyDE đặc biệt hiệu quả khi câu hỏi của người dùng ngắn, mơ hồ hoặc sử dụng từ vựng khác biệt so với tài liệu

## Lợi ích

- Cải thiện chất lượng retrieval mà không cần huấn lại mô hình embedding
- Hữu ích khi có khoảng cách về từ vựng giữa câu hỏi và tài liệu
- Tận dụng kiến thức của LLM để tạo ra nội dung giả định hợp lý
- Dễ triển khai và không thay đổi được kiến trúc retrieval hiện có

## Hạn chế

- Phụ thuộc vào chất lượng của LLM được sử dụng để tạo tài liệu giả định
- Thêm bước xử lý (tạo tài liệu giả định) nên có thể chậm hơn so với truy xuất trực tiếp
- Nếu LLM tạo ra tài liệu giả định không chính xác, có thể dẫn tới retrieval kết quả kém
- Cần tiêu thụ thêm tài nguyên để chạy LLM cho bước tạo tài liệu giả định

## Tóm tắt

HyDE trong RAG là kỹ thuật dùng LLM để tạo ra tài liệu giả định trả lời câu hỏi, sau đó embed tài liệu giả định và tìm kiếm các đoạn văn thực có độ tương đồng cao nhất. Cách làm này giúp hệ thống truy xuất theo ngữ nghĩa tốt hơn, đặc biệt khi câu hỏi và tài liệu không trùng khớp từ ngữ hoặc khi câu hỏi quá ngắn để chứa đủ thông tin để matching hiệu quả.