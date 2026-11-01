# Các kỹ thuật lập chỉ mục cơ bản cho hệ thống RAG

## Tổng quan

Module này triển khai các kỹ thuật lập chỉ mục nền tảng cần thiết cho hệ thống Retrieval-Augmented Generation (RAG). Việc lập chỉ mục đúng cách chuyển đổi tài liệu thô thành các cấu trúc có thể tìm kiếm, cho phép truy xuất nhanh và phù hợp — yếu tố quan trọng đối với hiệu năng của RAG.

## Các khái niệm cốt lõi

### 1. Chỉ mục đảo
- **Mục đích**: Truy xuất dựa trên từ khóa bằng cách ánh xạ thuật ngữ với tài liệu
- **Cách hoạt động**: Ánh xạ mỗi thuật ngữ duy nhất tới các tài liệu/chunk chứa thuật ngữ đó, cùng với tần suất xuất hiện
- **Phù hợp nhất cho**: Khớp từ khóa chính xác, truy vấn Boolean, xếp hạng dựa trên TF-IDF
- **Hạn chế**: Khả năng hiểu ngữ nghĩa kém, gặp khó khăn với từ đồng nghĩa và các biến thể hình thái

### 2. Chỉ mục vector
- **Mục đích**: Tìm kiếm ngữ nghĩa bằng dense embedding
- **Cách hoạt động**: Chuyển đổi văn bản thành các vector có độ dài cố định bằng mô hình embedding, sau đó truy xuất thông qua tìm kiếm tương đồng (cosine/tích vô hướng)
- **Phù hợp nhất cho**: Tương đồng ngữ nghĩa, khớp theo khái niệm, truy xuất đa ngôn ngữ
- **Hạn chế**: Cần có mô hình embedding, khó diễn giải hơn, có thể bỏ sót các kết quả khớp chính xác theo từ khóa

### 3. Chỉ mục lai
- **Mục đích**: Kết hợp ưu điểm của chỉ mục đảo và chỉ mục vector
- **Cách hoạt động**: Duy trì riêng cả hai chỉ mục, sau đó kết hợp điểm số tại thời điểm truy vấn
- **Phù hợp nhất cho**: Truy xuất cân bằng giữa từ khóa và ngữ nghĩa, hiệu năng ổn định với nhiều loại truy vấn
- **Đánh đổi**: Chi phí lưu trữ và tính toán cao hơn, cần có chiến lược hợp nhất điểm số

### 4. Các chiến lược chia nhỏ tài liệu
Việc chia nhỏ đúng cách rất quan trọng để lập chỉ mục hiệu quả:

- **Kích thước cố định**: Chia theo số ký tự/token (đơn giản nhưng có thể làm tách rời các đơn vị ngữ nghĩa)
- **Theo câu**: Chia tại ranh giới câu (bảo toàn các đơn vị ngôn ngữ)
- **Theo đoạn**: Chia tại ranh giới đoạn văn (bảo toàn các đơn vị chủ đề)
- **Ngữ nghĩa**: Chia theo sự mạch lạc về ngữ nghĩa (cần mô hình, mang lại ý nghĩa tốt nhất)
- **Đệ quy**: Chia theo các dấu phân cách theo thứ tự quan trọng (Markdown, cấu trúc mã)
- **Cửa sổ trượt**: Các chunk chồng lấn lên nhau (bảo toàn ngữ cảnh tại ranh giới)

## Ghi chú triển khai

Triển khai này cung cấp:
- Lớp cơ sở trừu tượng (`BaseIndex`) định nghĩa giao diện lập chỉ mục
- Các triển khai cụ thể cho từng loại chỉ mục
- Pipeline thống nhất (`IndexingPipeline`) giúp dễ dàng chuyển đổi giữa các kỹ thuật
- Các cấu trúc dữ liệu được chuẩn hóa (`DocumentChunk`, `IndexStats`)
- Các hàm tokenization và embedding cơ bản (placeholder cho các mô hình dùng trong môi trường production)

## Khi nào nên sử dụng từng kỹ thuật

- **Chỉ mục đảo**: Khi độ chính xác của từ khóa là yếu tố quan trọng nhất (tài liệu pháp lý, tài liệu kỹ thuật)
- **Chỉ mục vector**: Khi khả năng hiểu ngữ nghĩa quan trọng hơn việc khớp chính xác (QA hội thoại)
- **Chỉ mục lai**: Cho các hệ thống RAG đa mục đích cần cả hai khả năng
- **Chiến lược chia nhỏ**: Lựa chọn phù hợp với cấu trúc tài liệu và mục tiêu truy xuất