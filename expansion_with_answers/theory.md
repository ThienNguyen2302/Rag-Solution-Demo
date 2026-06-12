# Query Expansion trong RAG

Query expansion là kỹ thuật mở rộng câu hỏi người dùng thành nhiều biến thể truy vấn khác nhau trước khi tìm kiếm trong vector database. Mục tiêu là tăng khả năng khớp ngữ nghĩa với các đoạn văn liên quan, đặc biệt khi câu hỏi ban đầu quá ngắn, mơ hồ hoặc dùng từ không giống cách tài liệu gốc diễn đạt.

## Ý tưởng chính

Trong RAG, retriever thường lấy các embedding của câu hỏi để tìm những chunk gần nhất trong không gian vector. Vấn đề là một câu hỏi ngắn có thể không chứa đủ tín hiệu để truy xuất đúng tài liệu. Query expansion giải quyết bằng cách:

1. Giữ nguyên truy vấn gốc của người dùng.
2. Dùng LLM tạo ra 3 đến 5 truy vấn mở rộng dựa trên ý định cốt lõi.
3. Thêm từ đồng nghĩa, khái niệm liên quan, tên riêng, chủ đề đặc trưng, hoặc cách diễn đạt khác.
4. Ghép truy vấn gốc và phần mở rộng lại trước khi embedding và retrieval.

![Sơ đồ](data/query_expansion_techniques.png)

## Cách hoạt động trong code

Trong file `expansion_answers.py`, luồng xử lý đi theo các bước sau:

1. Đọc tài liệu PDF và chia nhỏ thành các chunk.
2. Tạo embedding cho từng chunk và lưu vào ChromaDB.
3. Nhận câu hỏi từ người dùng.
4. Gọi hàm `generate_augment_query()` để LLM sinh ra danh sách truy vấn mở rộng.
5. Ghép truy vấn gốc với kết quả mở rộng để tạo thành `joint_query`.
6. Dùng `joint_query` để truy xuất các chunk liên quan hơn.
7. Đưa các chunk đã truy xuất vào hàm sinh câu trả lời cuối cùng.

Điểm quan trọng là mô hình không trả lời ngay ở bước mở rộng. Nó chỉ tạo ra các phiên bản truy vấn khác nhau để tăng “diện tích tìm kiếm” trong không gian ngữ nghĩa.

## Vì sao query expansion hiệu quả

Một câu hỏi như “What did he say about failure?” có thể quá chung chung. Nếu mở rộng thành các biến thể như:

- “What did Randy Pausch say about failure and making mistakes?”
- “brick walls are there to show how badly we want something”
- “learning from setbacks and overcoming obstacles in The Last Lecture”

thì hệ thống có thêm nhiều dấu hiệu ngữ nghĩa để tìm đúng đoạn văn hơn. Nói cách khác, thay vì chỉ tìm theo một vector duy nhất, ta tìm theo một biểu diễn giàu thông tin hơn.

## Lợi ích

- Tăng recall, tức khả năng tìm được chunk liên quan.
- Hữu ích với truy vấn ngắn, thiếu ngữ cảnh, hoặc dùng từ khác với tài liệu.
- Có thể tận dụng kiến thức của LLM để suy ra các cách diễn đạt gần nghĩa.

## Hạn chế

- Có thể làm truy vấn bị “lệch ý” nếu LLM mở rộng sai hướng.
- Nếu mở rộng quá rộng, retriever có thể kéo về nhiều chunk nhiễu hơn.
- Tốn thêm thời gian và chi phí vì phải gọi LLM trước khi truy xuất.

## Ghi chú về cách đặt tên trong code

Biến `hypothetical_answer` trong code thực tế đang chứa các truy vấn mở rộng, không phải câu trả lời. Tên này có thể gây nhầm lẫn. Về mặt khái niệm, đây là query expansion, không phải answer generation.

## Tóm tắt

Query expansion trong RAG là bước biến một câu hỏi gốc thành nhiều phiên bản truy vấn giàu ngữ nghĩa hơn, rồi dùng các truy vấn đó để tìm chunk liên quan chính xác hơn. Kỹ thuật này đặc biệt hữu ích khi dữ liệu cần tìm nằm rải rác hoặc được diễn đạt khác với cách người dùng đặt câu hỏi.

![Use case](data/use_case.png)
