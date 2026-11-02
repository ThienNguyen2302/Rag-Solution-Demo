# RAPTOR (Xử lý trừu tượng đệ quy cho truy xuất được tổ chức theo cây) trong hệ thống RAG

## Tổng quan

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) là một kỹ thuật truy xuất nâng cao, xây dựng cấu trúc cây phân cấp bằng cách đệ quy tóm tắt các đoạn văn bản ở những mức độ trừu tượng tăng dần. Phương pháp này cho phép truy xuất ở nhiều mức độ chi tiết khác nhau và cải thiện hiệu năng đối với các truy vấn cần hiểu ngữ cảnh tổng thể.

## Vấn đề cốt lõi

Các phương pháp truy xuất truyền thống phải đối mặt với sự đánh đổi cơ bản:
- **Độ chính xác cục bộ**: Các đoạn ngắn nắm bắt được chi tiết chính xác nhưng bỏ sót ngữ cảnh rộng hơn
- **Độ bao phủ toàn cục**: Các đoạn dài hoặc toàn bộ tài liệu nắm bắt được ngữ cảnh nhưng làm loãng các kết quả khớp cụ thể

RAPTOR giải quyết vấn đề này bằng cách tạo ra biểu diễn cơ sở tri thức ở nhiều độ phân giải, cho phép hệ thống truy xuất thông tin ở mức độ trừu tượng phù hợp với từng truy vấn.

## Quy trình RAPTOR

### 1. Chia nhỏ tài liệu (mức nút lá)
- Tài liệu được chia thành các đoạn nhỏ (thường từ 100–200 token)
- Mỗi đoạn trở thành một nút lá trong cây
- Embedding được tạo cho từng nút lá

### 2. Phân cụm và tóm tắt đệ quy
- Các nút lá được phân cụm dựa trên độ tương đồng ngữ nghĩa
- Mỗi cụm được tóm tắt để tạo thành một nút cha (biểu diễn trừu tượng hơn)
- Các nút cha được tạo embedding và trở thành đầu vào cho cấp tiếp theo
- Quy trình tiếp tục cho đến khi đạt chiều cao cây mong muốn hoặc hội tụ

### 3. Cấu trúc cây
- **Cấp 0**: Các nút lá (các đoạn ban đầu)
- **Cấp 1**: Bản tóm tắt của các cụm nút lá
- **Cấp 2**: Bản tóm tắt của các cụm cấp 1
- ...
- **Cấp N**: Các nút gốc biểu diễn những khái niệm trừu tượng nhất

### 4. Cơ chế truy xuất
- Truy vấn được tạo embedding và so sánh với các nút ở mọi cấp
- Kết quả có thể được trả về từ bất kỳ cấp nào, tùy thuộc vào độ cụ thể của truy vấn
- Có thể cung cấp ngữ cảnh phân cấp (hiển thị quan hệ nút cha/nút con)
- Hỗ trợ cả truy xuất chi tiết chính xác lẫn đối sánh khái niệm toàn cục

## Ưu điểm

### 1. Đối sánh đa mức độ chi tiết
- Các truy vấn cụ thể khớp với những nút lá chi tiết
- Các truy vấn khái quát khớp với những nút tóm tắt trừu tượng
- Không cần chọn một kích thước đoạn duy nhất

### 2. Bảo toàn ngữ cảnh
- Các nút cha giữ lại thông tin ngữ nghĩa từ các nút con
- Cho phép truy xuất với khả năng hiểu ngữ cảnh
- Giảm sự phân mảnh của các thông tin liên quan

### 3. Suy luận toàn cục hiệu quả
- Các nút trừu tượng cho phép đánh giá nhanh mức độ liên quan của tài liệu
- Cho phép lọc trước khi thực hiện các phép so sánh chi tiết tốn kém
- Hỗ trợ điều hướng tri thức theo cấu trúc phân cấp

### 4. Khả năng chống nhiễu
- Việc tóm tắt có thể lọc bỏ các chi tiết không liên quan
- Các cấp độ trừu tượng ít bị ảnh hưởng hơn bởi những biến thể nhỏ
- Cung cấp góc nhìn đã được khử nhiễu về các tài liệu phức tạp

## Chi tiết triển khai

### Chiến lược phân cụm
- Sử dụng độ tương đồng cosine giữa các embedding của nút
- Gán tham lam vào tâm cụm gần nhất
- Bị giới hạn bởi hệ số phân nhánh để kiểm soát độ rộng của cây

### Tóm tắt
- Tóm tắt trích xuất (chọn các câu then chốt)
- Trong môi trường thực tế: Thay thế bằng các mô hình tóm tắt trừu tượng (PEGASUS, BART, T5, v.v.)
- Kiểm soát độ dài để duy trì mức độ trừu tượng phù hợp

### Tạo embedding
- Sử dụng hàm embedding đơn giản để minh họa
- Trong môi trường thực tế: Sử dụng sentence transformer (SBERT, v.v.) hoặc embedding theo ngữ cảnh
- Không gian embedding nhất quán ở mọi cấp cho phép so sánh công bằng

### Tính linh hoạt khi tìm kiếm
- Tìm kiếm theo cấp: Nhắm đến các cấp độ trừu tượng cụ thể
- Tìm kiếm đa cấp: Tìm kiếm đồng thời trên mọi cấp
- Kết quả theo ngữ cảnh: Bao gồm quan hệ nút cha/nút con trong đầu ra
- Chuẩn hóa điểm số: Cho phép so sánh công bằng giữa các cấp

## Khi nào nên sử dụng RAPTOR

RAPTOR đặc biệt hữu ích khi:
- Tập hợp tài liệu chứa các tài liệu có độ dài không đồng nhất
- Truy vấn của người dùng đa dạng, từ câu hỏi thực tế cụ thể đến câu hỏi khái niệm rộng
- Cần tổng hợp tri thức xuyên suốt các phần của tài liệu
- Hiểu ngữ cảnh là yếu tố quan trọng để truy xuất chính xác
- Kho dữ liệu phù hợp với việc phân tích đa độ phân giải (tài liệu pháp lý, y tế, kỹ thuật)

## So sánh với các phương pháp thay thế

| Kỹ thuật | Mức độ chi tiết | Ngữ cảnh | Phù hợp nhất cho |
|-----------|-----------------|----------|------------------|
| Chia đoạn phẳng | Cố định | Hạn chế | Truy xuất đơn giản, tài liệu đồng nhất |
| Đa biểu diễn | Nhiều góc nhìn | Theo từng góc nhìn | Các khía cạnh khác nhau của cùng một nội dung |
| **RAPTOR** | **Cây phân cấp** | **Đường dẫn cha–con** | **Truy vấn đa mức độ chi tiết, bảo toàn ngữ cảnh** |
| Phân cụm phân cấp | Các cụm phẳng | Chỉ ở mức cụm | Mô hình hóa chủ đề không tập trung vào truy xuất |

## Ghi chú triển khai

Phần triển khai này cung cấp:
- Thiết kế mô-đun với các thành phần có thể tách rời (bộ tóm tắt, bộ tạo embedding, bộ phân cụm)
- Cấu trúc cây rõ ràng với các quan hệ giữa các nút
- Tìm kiếm linh hoạt với đầu ra theo ngữ cảnh
- Theo dõi thống kê để giám sát đặc điểm của cây
- Ví dụ sử dụng minh họa chức năng đầu-cuối

Trong các hệ thống thực tế, `SimpleSummarizer` và `SimpleColBERTEmbedder` sẽ được thay thế bằng các mô hình đã được huấn luyện trước từ Hugging Face hoặc các nguồn tương tự, qua đó cải thiện đáng kể chất lượng tóm tắt và hiệu quả của embedding.