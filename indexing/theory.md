# Trực quan và lý thuyết về kỹ thuật lập chỉ mục trong hệ thống RAG

## 1. Giới thiệu chung

Trong hệ thống Retrieval-Augmented Generation (RAG), lập chỉ mục là thành phần then chốt, chịu trách nhiệm biến đổi dữ liệu văn bản thô thành cấu trúc có thể tìm kiếm nhanh và hiệu quả. Mục tiêu chính là giảm thời gian tìm kiếm từ O(n) (phải quét toàn bộ n tài liệu) xuống mức dưới tuyến tính như O(log n) hoặc tốt hơn, đồng thời duy trì mức độ liên quan cao giữa truy vấn và kết quả.

## 2. Trực quan về lập chỉ mục

Hãy hình dung lập chỉ mục trong RAG giống như mục lục của một thư viện khổng lồ chứa hàng triệu trang văn bản. Thay vì đọc từng trang mỗi khi có câu hỏi, ta tạo một hệ thống chỉ mục để nhanh chóng định vị những đoạn văn bản có khả năng chứa câu trả lời.

Ví dụ, khi tìm một khái niệm trong sách giáo khoa 500 trang, bạn không đọc từ trang 1 đến trang 500 mà tra mục lục hoặc chỉ mục ở cuối sách. Lập chỉ mục trong RAG cũng làm điều tương tự: biến văn bản thô thành cấu trúc tìm kiếm nhanh như cây, bảng băm hoặc không gian vector.

Lập chỉ mục giải quyết các vấn đề cốt lõi:
1. **Tốc độ**: Tìm kiếm trong hàng triệu hoặc hàng tỷ đoạn văn bản mà không quét từng đoạn
2. **Khả năng mở rộng**: Hệ thống vẫn hoạt động hiệu quả khi dữ liệu tăng gấp 10, 100 hoặc 1.000 lần
3. **Mức độ liên quan**: Không chỉ tìm nhanh mà còn tìm đúng, dựa trên ngữ nghĩa chứ không chỉ từ khóa
4. **Tiết kiệm tài nguyên**: Giảm lượng tính toán cho mỗi truy vấn, cho phép xử lý nhiều truy vấn đồng thời

## 3. Lý thuyết cơ bản về kỹ thuật lập chỉ mục

### 3.1. Mục tiêu tối ưu

Một hệ thống lập chỉ mục tốt cần đạt được:
- **Thời gian tìm kiếm dưới tuyến tính**: Thay vì O(n), hướng tới O(log n), O(1) hoặc O(√n)
- **Đo lường tương đồng có ý nghĩa**: Phản ánh đúng mức độ liên quan về ngữ nghĩa giữa truy vấn và tài liệu
- **Cân bằng độ bao phủ và độ chính xác**:
  - Độ bao phủ (recall): Tìm đủ các kết quả thực sự liên quan, không bỏ sót kết quả tốt
  - Độ chính xác (precision): Không trả về quá nhiều kết quả không liên quan
- **Hiệu quả về không gian**: Bộ nhớ bổ sung cho chỉ mục hợp lý so với kích thước dữ liệu gốc

### 3.2. Hai nguyên tắc cơ bản

#### Nguyên tắc 1: Ánh xạ từ không gian văn bản sang không gian tìm kiếm
- **Không gian văn bản**: Dữ liệu văn bản thô, phức tạp và có thể mơ hồ
- **Không gian tìm kiếm**: Cấu trúc dữ liệu được tối ưu cho truy vấn nhanh (bảng băm, cây hoặc không gian vector)
- **Hàm biến đổi**: Chuyển văn bản thành biểu diễn tìm kiếm (thuật ngữ, vector, cây, v.v.)

#### Nguyên tắc 2: Đo lường tương đồng trong không gian tìm kiếm

Sau khi biến đổi, cần một hàm đo lường tương đồng:
- **Dựa trên thuật ngữ**: TF-IDF, BM25 (độ quan trọng của từ trong tài liệu so với toàn bộ kho dữ liệu)
- **Dựa trên vector**: Độ tương đồng cosine, tích vô hướng (góc hoặc tích giữa vector truy vấn và tài liệu)
- **Dựa trên cây**: Khoảng cách trong cấu trúc cây (RAPTOR)
- **Tương tác trễ**: MaxSim (ColBERT – độ tương đồng tốt nhất giữa các cặp token)

### 3.3. Các đánh đổi cơ bản

| Yếu tố đánh đổi | Mô tả | Ảnh hưởng đến lựa chọn kỹ thuật |
|----------------|-------|--------------------------------|
| **Xây dựng chỉ mục và truy vấn thời gian thực** | Xây dựng chỉ mục có thể tốn thời gian một lần, nhưng truy vấn phải nhanh (dưới 100 ms) | Kỹ thuật phức tạp thường xây dựng lâu hơn nhưng truy vấn nhanh hơn |
| **Mức độ chi tiết và hiệu quả** | Giữ nhiều thông tin giúp chính xác hơn nhưng chậm hơn; giữ ít thông tin thì nhanh nhưng kém chính xác | Chỉ mục vector nhanh hơn; ColBERT giữ nhiều thông tin token hơn |
| **Bộ nhớ và hiệu suất** | Chỉ mục lớn thường cho hiệu suất tốt hơn nhưng tốn RAM hơn | Chỉ mục đảo ít tốn bộ nhớ; chỉ mục vector tốt hơn về ngữ nghĩa |
| **Khớp chính xác và khớp ngữ nghĩa** | Tìm đúng từ khóa khác với tìm theo ý nghĩa | Chỉ mục đảo mạnh về từ khóa; chỉ mục vector mạnh về ngữ nghĩa |

## 4. Trực quan và lý thuyết của các kỹ thuật cụ thể

### 4.1. Kỹ thuật lập chỉ mục cơ bản

#### 4.1.1. Chỉ mục đảo

**Trực quan:** Giống chỉ mục trong sách: mỗi từ được ánh xạ tới danh sách các trang chứa từ đó. Khi tìm “machine learning”, ta lấy danh sách của “machine” và “learning”, rồi lấy giao của hai danh sách.

**Lý thuyết:**
- **Cấu trúc**: Thuật ngữ → danh sách tài liệu chứa thuật ngữ đó (danh sách xuất hiện)
- **Xử lý truy vấn**: Tách truy vấn thành thuật ngữ, lấy các danh sách tương ứng, rồi hợp nhất bằng phép giao (AND) hoặc hợp (OR)
- **Đo tương đồng**: TF-IDF hoặc BM25
  - TF (tần suất thuật ngữ): Số lần từ xuất hiện trong tài liệu
  - IDF (tần suất tài liệu nghịch đảo): log(tổng số tài liệu / số tài liệu chứa thuật ngữ)
  - Điểm = tổng TF(t,d) × IDF(t) với mọi thuật ngữ t trong truy vấn
- **Ưu điểm**: Rất nhanh khi khớp chính xác, dễ giải thích, ít tốn bộ nhớ với dữ liệu thưa
- **Nhược điểm**: Không hiểu ngữ nghĩa, gặp khó khăn với từ đồng nghĩa và từ đa nghĩa

#### 4.1.2. Chỉ mục vector

**Trực quan:** Mỗi tài liệu trở thành một điểm trong không gian vector nhiều chiều; các điểm gần nhau có ý nghĩa tương tự. Truy vấn cũng được biểu diễn thành một điểm, sau đó tìm các điểm tài liệu gần nhất.

**Lý thuyết:**
- **Biến đổi**: Tài liệu → vector trong R^d (d là số chiều embedding)
  - Sử dụng mô hình embedding như BERT hoặc SBERT
  - Mỗi chiều biểu diễn một khía cạnh ngữ nghĩa
- **Truy vấn**: Tạo vector q cho truy vấn, tính tương đồng với các vector tài liệu dᵢ, rồi trả về k tài liệu có điểm cao nhất
- **Đo tương đồng**:
  - Cosine: (q·d) / (||q|| × ||d||)
  - Tích vô hướng: q·d (khi vector đã được chuẩn hóa)
- **Ưu điểm**: Nắm bắt ý nghĩa, xử lý từ đồng nghĩa, phù hợp với truy vấn mô tả
- **Nhược điểm**: Có thể mất thông tin từ khóa cụ thể và phụ thuộc vào chất lượng mô hình embedding

#### 4.1.3. Chỉ mục lai

**Trực quan:** Kết hợp ưu điểm của cả hai phương pháp: dùng chỉ mục đảo để lọc nhanh tài liệu có từ khóa chính xác, sau đó dùng chỉ mục vector để xếp hạng theo mức độ liên quan ngữ nghĩa.

**Lý thuyết:**
1. **Lọc**: Dùng chỉ mục đảo để tìm tập tài liệu chứa một số thuật ngữ trong truy vấn
2. **Xếp hạng**: Dùng chỉ mục vector để tính cosine và xếp hạng tập đã lọc
3. **Công thức hợp nhất**: `Score = α × Score_text + (1 - α) × Score_vector`
   - `Score_text`: Điểm từ TF-IDF hoặc BM25
   - `Score_vector`: Điểm cosine
   - `α`: Trọng số cân bằng, thường từ 0,2 đến 0,8 tùy ứng dụng
- **Ưu điểm**: Kết hợp được khớp chính xác và khớp ngữ nghĩa
- **Nhược điểm**: Triển khai phức tạp hơn và cần tinh chỉnh α

### 4.2. Lập chỉ mục đa biểu diễn

**Trực quan:** Thay vì chỉ có một cách nhìn tài liệu, ta tạo nhiều “ống kính”: toàn văn để xem chi tiết, tóm tắt để xem tổng quan, câu hỏi để khớp truy vấn hỏi–đáp, và tiêu đề–tóm tắt để khớp tài liệu học thuật. Mỗi ống kính được lập chỉ mục bằng phương pháp phù hợp.

**Lý thuyết:**
- Các biểu diễn thường dùng: toàn văn, tóm tắt, tiêu đề–tóm tắt, câu hỏi, thực thể và cụm từ khóa
- Chiến lược hợp nhất:
  1. **Tổng có trọng số**: Σ(wᵢ × Scoreᵢ)
  2. **Điểm cao nhất**: Max(Scoreᵢ)
  3. **Hợp nhất thứ hạng (RRF)**: Σ(1/(k + rankᵢ))
  4. **Dựa trên phân phối (điểm Z)**: Chuẩn hóa điểm trước khi cộng
- **Ưu điểm**: Tăng độ bao phủ, linh hoạt với nhiều dạng truy vấn, cho phép tối ưu từng biểu diễn
- **Nhược điểm**: Tăng dung lượng lưu trữ, thời gian tiền xử lý và độ phức tạp của chiến lược hợp nhất

### 4.3. RAPTOR

**Trực quan:** Xây dựng cây tri thức: lá là các đoạn văn bản gốc, nút giữa là bản tóm tắt của nút con, còn gốc là bản tóm tắt khái quát nhất của toàn bộ cơ sở tri thức. Truy vấn cụ thể tìm ở gần lá; truy vấn tổng quát tìm ở gần gốc.

**Lý thuyết:**
1. **Chia đoạn**: Chia tài liệu thành các đoạn cơ bản (cấp 0, các nút lá)
2. **Phân cụm**: Nhóm các nút tương tự ở cấp hiện tại
3. **Tóm tắt**: Tóm tắt mỗi cụm thành một nút cha (cấp tiếp theo)
4. **Lặp lại**: Tiếp tục với các nút cha cho đến khi còn một nút gốc hoặc đạt độ sâu mong muốn

Các truy vấn cụ thể thường tìm ở cấp sâu để lấy chi tiết; truy vấn tổng quát tìm ở cấp cao để lấy bối cảnh. Có thể kết hợp kết quả từ nhiều cấp.

- **Ưu điểm**: Truy xuất được cả chi tiết và bối cảnh, hiệu quả với tài liệu dài, phù hợp với tri thức có cấu trúc phân cấp
- **Nhược điểm**: Xây dựng tốn thời gian, phụ thuộc chất lượng tóm tắt và cần chọn độ sâu tối ưu

### 4.4. ColBERT

**Trực quan:** Thay vì nén toàn bộ tài liệu thành một vector, ColBERT giữ vector cho từng token. Khi truy vấn, hệ thống tìm độ tương đồng cao nhất giữa mỗi token truy vấn và mọi token tài liệu (MaxSim), rồi cộng các điểm cao nhất.

**Lý thuyết:**
- Mỗi token của tài liệu và truy vấn có một embedding riêng; tài liệu n token tạo ma trận n × d, truy vấn m token tạo ma trận m × d
- Tương tác được thực hiện ở thời điểm truy vấn, tránh mất thông tin do nén sớm
- MaxSim tìm token tài liệu khớp nhất cho từng token truy vấn rồi cộng các điểm tương đồng
- **Ưu điểm**: Giữ chi tiết cấp token, xử lý tốt từ đa nghĩa và từ đồng nghĩa, hiệu quả hơn cross-encoder
- **Nhược điểm**: Tốn bộ nhớ hơn chỉ mục vector, truy vấn chậm hơn và cần tối ưu MaxSim

## 5. Khi nào nên sử dụng kỹ thuật nào?

| Kỹ thuật | Khi nào sử dụng | Lưu ý thực tế |
|----------|----------------|---------------|
| **Chỉ mục đảo cơ bản** | Truy vấn cần khớp từ khóa; tài liệu pháp lý, kỹ thuật hoặc mã nguồn; tài nguyên hạn chế | Dùng stemming, loại stop word và nén danh sách xuất hiện |
| **Chỉ mục vector cơ bản** | Ngữ nghĩa quan trọng hơn khớp chính xác; hỏi đáp, trò chuyện, gợi ý | Chọn mô hình theo miền, chuẩn hóa vector và cân nhắc lượng tử hóa |
| **Chỉ mục lai** | Cần cả độ chính xác từ khóa và khả năng hiểu ngữ nghĩa | Tinh chỉnh α và chuẩn hóa điểm giữa hai chỉ mục |
| **Đa biểu diễn** | Tài liệu có nhiều góc nhìn, truy vấn đa dạng và có thể duy trì nhiều chỉ mục | Tập trung vào 3–5 biểu diễn hữu ích, thử nghiệm các chiến lược hợp nhất |
| **RAPTOR** | Tài liệu dài, cần cả chi tiết lẫn bối cảnh, tri thức có cấu trúc phân cấp | Đánh giá độ sâu bằng tập kiểm thử và dùng mô hình tóm tắt tốt |
| **ColBERT** | Cần độ chính xác cao, truy vấn mơ hồ hoặc nhiều sắc thái, có đủ tài nguyên | Dùng mô hình BERT thực tế, tối ưu MaxSim bằng ANN và cân nhắc giảm chiều vector |

## 6. Nguyên tắc thiết kế tổng quan

### 6.1. Biểu diễn không mất mát và có mất mát

- **Biểu diễn không mất mát**: Giữ nguyên toàn bộ văn bản gốc; phù hợp khi cần tái tạo chính xác ngữ cảnh sau truy xuất
- **Biểu diễn có mất mát**: Chấp nhận bỏ một phần thông tin để đạt hiệu quả, như embedding hoặc nén có mất mát trong chỉ mục đảo; phù hợp cho truy xuất ban đầu
- **Phương pháp hiện đại**: Kết hợp cả hai, dùng biểu diễn có mất mát để tìm nhanh và biểu diễn không mất mát để tái tạo ngữ cảnh

### 6.2. Vòng lặp tối ưu độ bao phủ và độ chính xác

Một hệ thống tốt cần liên tục tối ưu:
1. **Đo lường**: Đánh giá độ bao phủ và độ chính xác trên tập kiểm thử
2. **Phân tích**: Tìm truy vấn bị bỏ sót hoặc trả về kết quả kém liên quan
3. **Điều chỉnh**:
   - Độ bao phủ thấp: Thêm biểu diễn, giảm ngưỡng tương đồng hoặc tối ưu phân cụm
   - Độ chính xác thấp: Tăng ngưỡng tương đồng, cải thiện biểu diễn hoặc tối ưu hợp nhất
4. **Lặp lại**: Tiếp tục đến khi đạt mức cân bằng mong muốn

### 6.3. Cân nhắc khi triển khai thực tế

- **Thời gian xây dựng chỉ mục**: Có thể chậm hơn vì thường chỉ thực hiện một lần
- **Thời gian truy vấn**: Phải nhanh, thường dưới 100 ms cho ứng dụng tương tác
- **Dấu chân bộ nhớ**: Phù hợp với RAM sẵn có; cân nhắc chỉ mục trên đĩa nếu cần
- **Tần suất cập nhật**: Dữ liệu tĩnh có thể xây dựng một lần; dữ liệu động cần cập nhật gia tăng hoặc xây dựng lại định kỳ
- **Mở rộng ngang**: Phân vùng theo mã tài liệu hoặc hàm băm, đồng thời nhân bản để tăng khả năng chịu lỗi và thông lượng

## 7. Kết luận

Lập chỉ mục không chỉ là bước chuẩn bị trong hệ thống RAG mà còn là thành phần chiến lược quyết định chất lượng và hiệu quả của toàn hệ thống. Việc lựa chọn kỹ thuật phù hợp cần dựa trên:

1. **Tính chất của miền tri thức**: Độ dài tài liệu, cấu trúc phân cấp và mức độ chuyên ngành
2. **Loại truy vấn**: Câu hỏi thực tế, câu hỏi khái niệm, câu hỏi so sánh hoặc câu hỏi khám phá
3. **Yêu cầu hiệu suất**: Độ trễ tối đa, thông lượng và tài nguyên CPU/GPU/RAM
4. **Đánh đổi giữa tính toán và độ chính xác**: Có thể hy sinh một phần độ chính xác để tăng tốc hoặc đầu tư thời gian xây dựng để giảm thời gian truy vấn

Các kỹ thuật từ cơ bản (chỉ mục đảo, vector và lai) đến nâng cao (đa biểu diễn, RAPTOR và ColBERT) tạo thành một bộ công cụ phong phú để tối ưu hệ thống RAG. Chìa khóa là hiểu rõ bài toán tìm kiếm, chọn kỹ thuật phù hợp và liên tục điều chỉnh dựa trên dữ liệu thực tế.

---

*Tài liệu này trình bày trực quan và lý thuyết về các kỹ thuật lập chỉ mục trong hệ thống RAG bằng tiếng Việt.*
