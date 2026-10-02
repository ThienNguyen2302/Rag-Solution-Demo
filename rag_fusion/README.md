# RAG Fusion

## RAG Fusion là gì?

RAG Fusion là kỹ thuật cải thiện retrieval trong hệ thống Retrieval-Augmented Generation (RAG) bằng cách tạo ra nhiều truy vấn từ một câu hỏi gốc, thực hiện tìm kiếm cho từng truy vấn, sau đó hợp nhất các danh sách kết quả bằng thông tin về thứ hạng.

Thay vì chỉ tìm kiếm một lần với một câu hỏi, hệ thống nhìn câu hỏi từ nhiều góc độ khác nhau. Một đoạn văn xuất hiện tốt trong nhiều danh sách kết quả sẽ có khả năng liên quan cao hơn.

Trong demo này, LLM tạo ra các truy vấn mở rộng cho cuốn sách _The Last Lecture_. Mỗi truy vấn được dùng để tìm các chunk trong vector store. Các chunk sau đó được hợp nhất bằng Reciprocal Rank Fusion (RRF) trước khi gửi cho LLM tạo câu trả lời cuối cùng.

## Vì sao cần RAG Fusion?

Một câu hỏi ngắn có thể không chứa đủ từ khóa hoặc ngữ cảnh để tìm đúng chunk. Cùng một ý tưởng có thể được diễn đạt theo nhiều cách khác nhau trong tài liệu.

Ví dụ, câu hỏi:

```text
What did he say about failure?
```

có thể được mở rộng thành:

- What did Randy Pausch say about failure and making mistakes?
- How does _The Last Lecture_ describe learning from setbacks?
- What is the meaning of the First Penguin Award?

Mỗi truy vấn có thể tìm được các chunk khác nhau. RAG Fusion kết hợp các kết quả này để tăng khả năng thu thập được thông tin liên quan.

## Quy trình hoạt động

```text
Câu hỏi gốc
    |
    v
LLM tạo nhiều truy vấn mở rộng
    |
    v
Tìm kiếm vector cho từng truy vấn
    |
    v
Nhiều danh sách kết quả có thứ hạng
    |
    v
Reciprocal Rank Fusion (RRF)
    |
    v
Context đã hợp nhất
    |
    v
LLM tạo câu trả lời cuối cùng
```

Trong file `rag_fusion.py`, quy trình chính gồm các bước:

1. Đọc file PDF và tách nội dung thành các chunk.
2. Chia chunk thành các đoạn phù hợp với tokenizer.
3. Tạo embedding và nạp các đoạn văn vào `TurboQuantVectorStore`.
4. Nhận câu hỏi từ người dùng.
5. Dùng Ollama để tạo các truy vấn mở rộng.
6. Tìm kiếm top-k chunk cho câu hỏi gốc và từng truy vấn mở rộng.
7. Hợp nhất các danh sách kết quả bằng RRF.
8. Đưa các chunk sau khi fusion vào prompt tạo câu trả lời.

## Reciprocal Rank Fusion

RRF không cần so sánh trực tiếp điểm embedding giữa các retriever khác nhau. Nó chỉ sử dụng vị trí của tài liệu trong mỗi danh sách kết quả.

Công thức:

$$
RRF(d) = \sum_{q \in Q} \frac{1}{k + rank(d, q)}
$$

Trong đó:

- `d` là một document hoặc chunk.
- `Q` là tập các truy vấn.
- `rank(d, q)` là vị trí của document trong kết quả của truy vấn `q`.
- `k` là hằng số làm giảm ảnh hưởng của thứ hạng đầu tiên, thường dùng giá trị `60`.

Ví dụ với `k = 0`:

```text
Query 1: A, B, C
Query 2: B, D, A
```

Điểm của các document:

```text
A = 1/1 + 1/3 = 1.33
B = 1/2 + 1/1 = 1.50
C = 1/3       = 0.33
D = 1/2       = 0.50
```

Thứ tự sau khi fusion là:

```text
B, A, D, C
```

`B` được xếp đầu vì nó xuất hiện ở vị trí cao trong cả hai danh sách.

Hàm `fuse_results_rrf()` trong `rrf.py` cũng tính điểm cho từng chunk, bỏ qua duplicate trong cùng một danh sách, sau đó sắp xếp theo điểm giảm dần.

## So sánh với Query Expansion

Query Expansion và RAG Fusion có liên quan nhưng không phải là một kỹ thuật.

### Query Expansion

Query Expansion tập trung vào việc tạo ra nhiều cách diễn đạt khác nhau cho câu hỏi gốc.

```text
Câu hỏi gốc
    |
    v
Nhiều truy vấn mở rộng
    |
    v
Tìm kiếm
```

Nó trả lời câu hỏi: “Nên tìm kiếm bằng những truy vấn nào?”

Trong repo, `expansion_with_queries/` tạo các query mở rộng, tìm kiếm với từng query, sau đó dùng `set()` để loại bỏ duplicate.

### RAG Fusion

RAG Fusion bao gồm bước tạo nhiều truy vấn, nhưng điểm quan trọng là cách hợp nhất kết quả.

```text
Câu hỏi gốc
    |
    v
Nhiều truy vấn mở rộng
    |
    v
Tìm kiếm từng truy vấn
    |
    v
RRF giữ lại thông tin thứ hạng
```

Nó trả lời câu hỏi: “Nên kết hợp các danh sách kết quả như thế nào?”

Nếu chỉ dùng `set()`, hệ thống biết document nào đã xuất hiện nhưng không biết document nào có thứ hạng tốt hơn. RRF giữ lại thông tin này.

## So sánh với Reranking

Reranking là bước sắp xếp lại một danh sách ứng viên bằng một mô hình mạnh hơn, thường là cross encoder.

```text
Truy vấn + danh sách ứng viên
    |
    v
Cross encoder chấm điểm từng cặp query-document
    |
    v
Danh sách mới được sắp xếp lại
```

Reranking trả lời câu hỏi: “Trong các document ứng viên, document nào phù hợp nhất với câu hỏi này?”

RAG Fusion trả lời câu hỏi: “Khi có nhiều danh sách kết quả từ nhiều truy vấn, document nào được ủng hộ nhiều nhất?”

| Tiêu chí        | Query Expansion                  | RAG Fusion                             | Reranking                          |
| --------------- | -------------------------------- | -------------------------------------- | ---------------------------------- |
| Mục đích        | Tạo thêm cách diễn đạt cho query | Hợp nhất nhiều danh sách kết quả       | Sắp xếp lại các ứng viên           |
| Đầu vào         | Một query                        | Nhiều query và nhiều ranked list       | Một query và các document ứng viên |
| Kỹ thuật chính  | LLM sinh query                   | RRF hoặc một hàm fusion khác           | Cross encoder hoặc reranker        |
| Tác động chính  | Tăng recall                      | Kết hợp tín hiệu từ nhiều lần tìm kiếm | Tăng precision                     |
| Chi phí         | Thêm một lần gọi LLM             | Thêm nhiều lần vector search           | Chấm điểm từng cặp query-document  |
| Có thể kết hợp? | Có                               | Có                                     | Có                                 |

## Có thể dùng chung cả ba kỹ thuật

Một pipeline đầy đủ có thể là:

```text
Câu hỏi người dùng
    |
    v
Query Expansion: tạo nhiều query
    |
    v
Vector Retrieval: tìm top-k cho mỗi query
    |
    v
RAG Fusion: hợp nhất bằng RRF
    |
    v
Reranking: cross encoder sắp xếp lại top candidates
    |
    v
LLM: tạo câu trả lời
```

Vai trò của từng bước:

- Query Expansion mở rộng phạm vi tìm kiếm.
- RAG Fusion tổng hợp các kết quả có thứ hạng.
- Reranking lọc và sắp xếp lại các chunk quan trọng nhất.
- LLM tạo câu trả lời dựa trên context cuối cùng.

Không nhất thiết lúc nào cũng phải dùng cả ba. Với dữ liệu nhỏ, Query Expansion kết hợp RRF có thể đã đủ. Reranking phù hợp khi kết quả fusion vẫn còn nhiều chunk nhiễu hoặc cần độ chính xác cao hơn.

## Ưu điểm

- Tăng khả năng tìm thấy thông tin khi query ban đầu ngắn hoặc mơ hồ.
- Kết hợp được nhiều góc nhìn của cùng một câu hỏi.
- Không phụ thuộc vào việc các retriever có cùng thang điểm.
- RRF đơn giản, dễ giải thích và ít tham số.
- Có thể kết hợp với reranking để làm sạch context trước khi tạo câu trả lời.

## Hạn chế

- Phải thực hiện nhiều lần vector search.
- Phụ thuộc vào chất lượng các query mở rộng do LLM tạo ra.
- Query mở rộng sai có thể đưa thêm nhiều kết quả nhiễu.
- RRF chỉ dựa vào thứ hạng, không hiểu nội dung document sâu như cross encoder.
- Nếu không giới hạn số lượng chunk, context có thể quá dài và tốn thêm chi phí LLM.

## Tóm tắt

Query Expansion tạo ra nhiều truy vấn.

RAG Fusion hợp nhất kết quả của nhiều truy vấn bằng thông tin thứ hạng, thường là RRF.

Reranking dùng mô hình mạnh hơn để chấm điểm và sắp xếp lại các document ứng viên.

Một hệ thống RAG có thể dùng cả ba theo thứ tự:

```text
Expansion -> Retrieval -> Fusion -> Reranking -> Generation
```
