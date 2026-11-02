# Lập chỉ mục đa biểu diễn cho hệ thống RAG

## Tổng quan

Lập chỉ mục đa biểu diễn là một kỹ thuật nâng cao, lưu trữ nhiều biểu diễn của cùng một tài liệu để nắm bắt các khía cạnh khác nhau và cải thiện hiệu quả truy xuất trong hệ thống Retrieval-Augmented Generation (RAG). Phương pháp này thừa nhận rằng các truy vấn khác nhau có thể khớp với những khía cạnh khác nhau của tài liệu. Bằng cách duy trì nhiều biểu diễn, ta có thể cải thiện độ bao phủ mà không làm giảm độ chính xác.

## Khái niệm cốt lõi

Ý tưởng cơ bản là tạo nhiều góc nhìn hoặc biểu diễn cho mỗi tài liệu, lập chỉ mục phù hợp cho từng biểu diễn (có thể dùng các chiến lược khác nhau), rồi hợp nhất kết quả tại thời điểm truy vấn. Cách làm này đặc biệt hiệu quả vì:

1. **Các truy vấn khác nhau khớp với các khía cạnh khác nhau**: Truy vấn về “lập trình Python” có thể khớp với toàn văn, trong khi “Ai tạo ra Python?” có thể khớp tốt hơn với biểu diễn hỏi–đáp
2. **Các chiến lược lập chỉ mục khác nhau phù hợp với các loại nội dung khác nhau**: Lập chỉ mục từ khóa phù hợp với thuật ngữ chính xác, còn lập chỉ mục vector nổi bật ở tương đồng ngữ nghĩa
3. **Bù đắp hạn chế của từng biểu diễn**: Không một biểu diễn đơn lẻ nào nắm bắt được mọi khía cạnh liên quan của tài liệu

## Các loại biểu diễn phổ biến

### 1. Biểu diễn toàn văn
- **Nội dung**: Toàn bộ văn bản của tài liệu
- **Phù hợp nhất cho**: Bao phủ toàn diện, truy xuất thông tin chi tiết
- **Lập chỉ mục**: Thường dùng cả chỉ mục đảo (từ khóa) và chỉ mục vector (ngữ nghĩa)

### 2. Biểu diễn tóm tắt
- **Nội dung**: Phiên bản cô đọng của tài liệu
- **Phù hợp nhất cho**: Đối sánh chủ đề nhanh, lọc tài liệu không liên quan
- **Lập chỉ mục**: Lập chỉ mục vector thường phù hợp với đối sánh tóm tắt theo ngữ nghĩa

### 3. Biểu diễn tiêu đề–tóm tắt
- **Nội dung**: Chỉ gồm tiêu đề và phần tóm tắt
- **Phù hợp nhất cho**: Lọc chủ đề nhanh, truy xuất bài viết học thuật
- **Lý do**: Tiêu đề và phần tóm tắt thường chứa thông tin nổi bật nhất

### 4. Biểu diễn dựa trên câu hỏi
- **Nội dung**: Các câu hỏi mà tài liệu trả lời (được sinh ra hoặc trích xuất)
- **Phù hợp nhất cho**: Đối sánh truy vấn trong hệ thống hỏi–đáp, truy xuất hội thoại
- **Lý do**: Căn chỉnh trực tiếp với định dạng truy vấn của người dùng

### 5. Biểu diễn dựa trên thực thể
- **Nội dung**: Thực thể có tên (người, địa điểm, tổ chức, v.v.)
- **Phù hợp nhất cho**: Truy vấn tập trung vào thực thể, kiểm chứng thông tin, phân giải mơ hồ
- **Lập chỉ mục**: Thường phù hợp với chỉ mục đảo để đối sánh thực thể chính xác

### 6. Biểu diễn cụm từ khóa
- **Nội dung**: Các cụm từ hoặc từ khóa quan trọng được trích xuất từ tài liệu
- **Phù hợp nhất cho**: Mô hình hóa chủ đề, đối sánh khái niệm, truy xuất dựa trên thẻ
- **Lý do**: Nắm bắt cốt lõi nội dung mà không bị nhiễu bởi toàn bộ văn bản

## Chiến lược hợp nhất

Sau khi truy xuất kết quả từ từng biểu diễn, cần kết hợp chúng một cách hiệu quả:

### 1. Tổng có trọng số
- **Cách làm**: Chuẩn hóa điểm từ từng biểu diễn, áp dụng trọng số rồi cộng lại
- **Phù hợp nhất khi**: Các biểu diễn có ưu điểm bổ trợ cho nhau
- **Tính linh hoạt**: Dễ điều chỉnh trọng số dựa trên mức độ quan trọng của từng biểu diễn

### 2. Điểm cao nhất
- **Cách làm**: Lấy điểm cao nhất của mọi biểu diễn cho mỗi tài liệu
- **Phù hợp nhất khi**: Chỉ cần một biểu diễn khớp mạnh là đủ
- **Tính bảo thủ**: Có xu hướng ưu tiên tài liệu có ít nhất một kết quả khớp rất mạnh

### 3. Hợp nhất thứ hạng (RRF)
- **Cách làm**: Chuyển thứ hạng thành điểm bằng công thức 1/(k + thứ hạng), sau đó cộng lại
- **Phù hợp nhất khi**: Thứ hạng đáng tin cậy hơn điểm số thô
- **Tính bền vững**: Ít nhạy cảm hơn với khác biệt về phân phối điểm giữa các biểu diễn

### 4. Dựa trên phân phối (điểm Z)
- **Cách làm**: Chuẩn hóa điểm bằng điểm Z cho từng biểu diễn rồi kết hợp
- **Phù hợp nhất khi**: Phân phối điểm giữa các biểu diễn khác nhau đáng kể
- **Tính thống kê**: Xử lý sự khác nhau về khoảng điểm và phương sai

## Cách tiếp cận triển khai

Phần triển khai này tuân theo thiết kế mô-đun:

1. **RepresentationConfig**: Xác định cách lập chỉ mục cho từng biểu diễn
2. **BaseRepresentationIndex**: Giao diện trừu tượng cho chỉ mục theo từng biểu diễn
3. **RepresentationIndexWrapper**: Điều chỉnh các chỉ mục hiện có (đảo, vector, lai) theo giao diện biểu diễn
4. **MultiRepresentationIndex**: Lớp chính quản lý nhiều biểu diễn và xử lý việc hợp nhất
5. **Các hàm hỗ trợ**: Cấu hình mặc định và hàm biến đổi (trích xuất tiêu đề, tạo tóm tắt, tạo câu hỏi)

## Ưu điểm

- **Cải thiện độ bao phủ**: Truy vấn khớp với bất kỳ biểu diễn nào cũng có thể truy xuất tài liệu liên quan
- **Linh hoạt**: Mỗi biểu diễn có thể sử dụng một chiến lược lập chỉ mục khác nhau
- **Dễ diễn giải**: Có thể thấy biểu diễn nào đóng góp vào kết quả khớp
- **Có thể tùy chỉnh**: Dễ dàng thêm loại biểu diễn hoặc chiến lược hợp nhất mới

## Khi nào nên sử dụng

Lập chỉ mục đa biểu diễn đặc biệt hữu ích khi:
- Tập hợp tài liệu không đồng nhất về nội dung hoặc phong cách
- Truy vấn của người dùng rất đa dạng về định dạng và mục đích
- Độ chính xác quan trọng nhưng cần cải thiện độ bao phủ
- Tài nguyên cho phép duy trì nhiều chỉ mục
- Nhật ký truy vấn cho thấy các mẫu mà một số biểu diễn cụ thể có thể bắt được

Kỹ thuật này làm tăng chi phí lưu trữ và tính toán tại thời điểm truy vấn, nhưng thường mang lại cải thiện đáng kể về chất lượng truy xuất, đủ để biện minh cho chi phí đó.
