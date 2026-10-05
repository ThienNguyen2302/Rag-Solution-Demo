# Phân giải truy vấn trong RAG

Phân giải truy vấn (Query Decomposition) là kỹ thuật trong hệ thống Retrieval-Augmented Generation (RAG) mà một câu hỏi phức tạp được tách thành nhiều câu hỏi con đơn giản, độc lập để cải thiện hiệu suất truy xuất và chất lượng câu trả lời.

## Ý tưởng chính

Trong RAG truyền thống, một câu hỏi phức tạp thường dẫn đến truy xuất không hiệu quả vì:
- Câu hỏi có thể chứa nhiều khía cạnh cần tìm kiếm thông tin khác nhau
- Vector embedding của câu hỏi phức tạp可能 không tốt représenter từng khía cạnh cụ thể
- Một passage có thể chỉ trả lời được một phần của câu hỏi

Phân giải truy vấn giải quyết vấn đề này bằng cách:
1. Phân tích câu hỏi gốc để xác định các khía cạnh thành phần
2. Tạo ra 2-4 câu hỏi con, mỗi câu hỏi tập trung vào một khía cạnh cụ thể
3. Thực hiện truy xuất riêng cho mỗi câu hỏi con
4. Tổng hợp kết quả từ tất cả các truy xuất để tạo câu trả lời cuối cùng

![Sơ đồ](images/query_decomposition.png)

## Cách hoạt động trong code

Trong file `decomposition.py`, luồng xử lý đi theo các bước sau:

1. Đọc tài liệu PDF và trích xuất toàn bộ văn bản.
2. Chia văn bản thành các chunk nhỏ bằng `RecursiveCharacterTextSplitter` và `SentenceTransformersTokenTextSplitter`.
3. Tạo vector embeddings cho tất cả các chunk menggunakan mô hình `sentence-transformers/all-MiniLM-L6-v2`.
4. Lưu trữ embeddings vào vector store (`TurboQuantVectorStore`).
5. Nhận câu hỏi phức tạp từ người dùng.
6. Sử dụng LLM (qwen3:0.6b) để phân giải câu hỏi thành 2-4 câu hỏi con đơn giản.
7. Với mỗi câu hỏi con, thực hiện similarity search để lấy top 3 chunk liên quan nhất.
8. Loại bỏ các chunk trùng lặp từ kết quả của các câu hỏi con khác.
9. Tổng hợp tất cả các chunk độc lập để tạo context.
10. Sử dụng LLM để tạo câu trả lời cuối cùng dựa trên context và câu hỏi gốc.

## Vì sao Phân giải truy vấn hiệu quả

Khác với việc truy xuất trực tiếp từ câu hỏi gốc, phân giải truy vấn giúp hệ thống:
- Tập trung vào từng khía cạnh cụ thể của câu hỏi thay vì пытаясь охватить wszystko naraz
- Tăng khả năng tìm thấy thông tin liên quan cho mỗi phần của câu hỏi
- Giảm nhiễu trong kết quả truy xuất vì mỗi truy xuất cụ thể hơn
- Cho phép hệ thống sử dụng các chiến lược truy xuất khác nhau cho các loại câu hỏi con khác nhau

Ví dụ, câu hỏi "What did Randy Pausch say is the specific purpose of 'brick walls', and how did he apply this concept to his courtship of his wife, Jai?" có thể được phân giải thành:
1. What is the specific purpose of 'brick walls' according to Randy Pausch?
2. How did Randy Pausch apply the 'brick walls' concept to his courtship with Jai?

Mỗi câu hỏi con này đơn giản hơn và có thể được truy xuất hiệu quả hơn.

## Phân giải truy vấn khác gì truy xuất chuẩn

Phân giải truy vấn và truy xuất chuẩn đều là thành phần trong pipeline RAG, nhưng chúng giải quyết các vấn đề khác nhau:

- Truy xuất chuẩn: Nhận một câu hỏi và tìm các passage liên quan nhất dựa trên độ similarity vector
- Phân giải truy vấn: Lưu lý trước khi truy xuất - biến một câu hỏi phức tạp thành nhiều câu hỏi đơn giản để làm cho truy xuất hiệu quả hơn

- Truy xuất estándar tập trung vào: "Làm sao để tìm passages tốt nhất cho một câu hỏi đã cho?"
- Phân giải truy vấn tập trung vào: "Làm sao để biến câu hỏi này thành các câu hỏi đơn giản để tìm passages tốt hơn?"

- Phân giải truy vấn cải thiện chất lượng đầu vào cho bước truy xuất
- Các cải tiến trong truy xuất (như vector database tốt hơn) vẫn có lợi cho cả hai cách tiếp cận

## Lợi ích

- Cải thiện chất lượng truy xuất bằng cách tập trung vào các khía cạnh cụ thể của câu hỏi
- Giảm khả năng bỏ sót thông tin quan trọng vì mỗi khía cạnh được xử lý riêng
- Tích hợp tốt với các kỹ thuật truy xuất nâng cao (re-ranking, etc.)
- Dễ dàng thực hiện và không cần thay đổi kiến trúc RAG cơ bản
- Hoạt động tốt với các câu hỏi đa khía cạnh, cần synthesis thông tin từ nhiều nguồn

## Hạn chế

- Phụ thuộc vào chất lượng của LLM dùng để sinh ra các câu hỏi con
- Thêm bước xử lý trước truy xuất, tăng latency tổng thể
- Nguy cơ sinh ra các câu hỏi con không相关 hoặc redundant nếu LLM không tốt
- Cần cuidadosa thiết kế prompt để đảm bảo các câu hỏi con thực sự hữu ích
- Có thể dẫn đến truy xuất quá nhiều thông tin nếu không có cơ chế lọc tốt

## Ghi chú về cách đặt tên trong code

File `decomposition.py` thể hiện đúng bản chất của kỹ thuật: sử dụng LLM để phân giải câu hỏi trước khi thực hiện truy xuất vector. Tên accurately reflects the core technique being demonstrated - query decomposition as a preprocessing step for retrieval in RAG systems.

## Tóm tắt

Phân giải truy vấn trong RAG là kỹ thuật dùng LLM để biến một câu hỏi phức tạp thành nhiều câu hỏi con đơn giản, độc lập, rồi thực hiện truy xuất riêng cho từng câu hỏi con trước khi tổng hợp kết quả. Cách làm này giúp hệ thống tập trung vào từng khía cạnh cụ thể của câu hỏi gốc, cải thiện chất lượng truy xuất và uiteindelijk dẫn đến câu trả lời tốt hơn, đặc biệt hiệu quả với các câu hỏi có nhiều khía cạnh hoặc cần synthesis thông tin từ nhiều nguồn khác nhau.