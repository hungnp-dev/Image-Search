# Tìm kiếm Hình ảnh Dựa trên Văn bản (Text To Image Search)

## **Giới thiệu:**

Đây là một hệ thống tìm kiếm Văn bản sang Hình ảnh được xây dựng trên cơ sở dữ liệu vector Qdrant, có khả năng tìm kiếm hình ảnh tương tự dựa trên các truy vấn văn bản.

Dữ liệu sử dụng cho việc truy xuất là các poster quảng cáo từ Google, được cung cấp tại các đường dẫn sau:
[phần thứ nhất](https://storage.googleapis.com/ads-dataset/subfolder-0.zip), [phần thứ hai](https://storage.googleapis.com/ads-dataset/subfolder-0.zip). Không cần cài bộ dữ liệu này vì đã được tích hợp sẵn trong source code.
## **Cấu trúc dự án:**

```
├───templates                   <- chứa các template giao diện Jinja được sử dụng trong dự án
│   ├───data_report_template.py <- template báo cáo phân tích dữ liệu
│   └───images_template.py      <- template hiển thị kết quả tìm kiếm hình ảnh
├───src                         <- mã nguồn chính
│   └───schemas.py              <- định nghĩa các schema cho FastAPI
├───utils                       <- các tiện ích
│   ├───data.py                 <- tiện ích xử lý dữ liệu
│   ├───search.py               <- tiện ích tìm kiếm
│   └───utils.py                <- tiện ích chung
├───evaluate.py                 <- endpoint đánh giá hiệu suất thuật toán
├───service.py                  <- endpoint khởi chạy ứng dụng FastAPI
└───prepare.py                  <- endpoint tạo báo cáo dữ liệu và cập nhật vector DB hình ảnh
```

## **Cài đặt:**

1. Cài đặt [Docker](https://docs.docker.com/engine/install/) trên máy tính

2. Cài đặt tất cả thư viện cần thiết:

Môi trường đã được kiểm nghiệm với Python 3.10.

```bash
pip install -r requirements.txt
```

3. Tải hình ảnh Qdrant từ Docker Hub:

```bash
docker pull qdrant/qdrant
```

4. Khởi chạy Qdrant trong Docker:

- Linux:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

<<<<<<< HEAD
- Windows CMD & PowerShell:
=======
- Windows CMD hoặc PowerShell:
>>>>>>> 8a6a3cefc7456b76739f16462a9b464db54540b6

CMD

```bash
docker run -p 6333:6333 -v %cd%/qdrant_storage:/qdrant/storage qdrant/qdrant
```

PowerShell

```bash
docker run -p 6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

## **Kiến trúc hệ thống:**

![](docs/architecture.jpg)

**Chi tiết**:

- Cả biểu diễn vector (embedding) hình ảnh và văn bản (chiều 512) đều được tạo bằng mô hình `CLIP-ViT-B-32`
- Sử dụng các mẫu Jinja để tạo báo cáo dữ liệu với các biểu đồ được tạo cục bộ
- Sử dụng mẫu Jinja để hiển thị biểu mẫu tìm kiếm và hình ảnh trong ứng dụng FastAPI
- Biểu mẫu tìm kiếm được xác thực để chỉ chứa từ (chữ cái) và khoảng trắng
- Sau khi nhập văn bản vào biểu mẫu tìm kiếm, hệ thống thực hiện yêu cầu POST đến API, quy trình như sau:
  - Gửi dữ liệu biểu mẫu đến công cụ tìm kiếm, tạo vector biểu diễn cho văn bản và thực hiện tìm kiếm láng giềng gần nhất xấp xỉ (ANN)
  - Công cụ tìm kiếm trả về payload của 5 hình ảnh liên quan nhất. Payload chứa đường dẫn đến các hình ảnh
  - Đường dẫn hình ảnh cục bộ được gửi đến mẫu Jinja để hiển thị trong cùng giao diện với biểu mẫu tìm kiếm

## **Sử dụng:**

Sử dụng lệnh sau để:

- Tải xuống, tổ chức và trích xuất thông tin về dữ liệu vào `resources/data_info.csv`
- Tạo báo cáo dữ liệu trong `resources/data_report.html`
- Xây dựng vector biểu diễn hình ảnh và lưu trữ trong `resources/image_embeddings.parquet`
- Tạo và điền vào bộ sưu tập Qdrant với các vector biểu diễn hình ảnh

```bash
python prepare.py
```

Để khởi chạy ứng dụng FastAPI, thực hiện lệnh sau:

```bash
python service.py
```

Để đánh giá thuật toán theo phần [Đánh giá](#eval), thực hiện lệnh sau:

```bash
python evaluate.py <PATH_TO_LABELS_TXT>
```

**LƯU Ý QUAN TRỌNG**: Các truy vấn cần được định nghĩa trong tệp `<PATH_TO_LABELS_TXT>`, mỗi truy vấn trên một dòng.
Một ví dụ đã được cung cấp sẵn tại tệp `docs/labels.txt`.

## **Kết quả:**

#### **Ví dụ tốt:**

- Từ khóa tìm kiếm: **beer** (bia)
  ![](docs/beer.jpg)
- Từ khóa tìm kiếm: **astronaut** (phi hành gia)
  ![](docs/astronaut.jpg)
- Từ khóa tìm kiếm: **fries with ketchup** (khoai tây chiên với tương cà)
  ![](docs/fries_with_ketchup.jpg)
- Từ khóa tìm kiếm: **bad weather** (thời tiết xấu) -> ta nhận thấy hệ thống đang xét đến các hình ảnh chứa từ `bad`, như `badminton`
  ![](docs/bad_weather.jpg)

#### **Ví dụ chưa tốt:**

- Từ khóa tìm kiếm: **black cat** (mèo đen) -> kết quả chưa được tốt
  ![](docs/black_cat.jpg)
- Từ khóa tìm kiếm: **happiness** (hạnh phúc) -> ..."sô-cô-la và ăn nhậu mang lại hạnh phúc cho chúng tôi".
  ![](docs/happiness.jpg)

## <a name="eval"></a> **Đánh giá độ chính xác truy xuất:**

Theo các phương pháp tốt nhất được Qdrant đề xuất [tại đây](https://qdrant.tech/documentation/tutorials/retrieval-quality/),
tôi đã đánh giá thuật toán ANN bằng cách sử dụng `precision@k` với `k=30`, bằng cách so sánh với tìm kiếm kNN đầy đủ và
xem xét mức độ xấp xỉ của thuật toán ANN so với tìm kiếm chính xác. Đối với danh sách truy vấn nhỏ được định nghĩa trong `docs/label.txt`, `precision=1.0`.

Để đánh giá định lượng độ chính xác truy xuất, chúng ta cần một tập hợp dữ liệu chuẩn (ground truth), được gán nhãn thủ công.
Tập hợp này sẽ chứa các truy vấn văn bản và K hình ảnh liên quan nhất tương ứng (thứ tự không quan trọng).
Chúng ta sẽ định nghĩa một tập kiểm tra riêng biệt gồm các truy vấn văn bản và chạy chúng qua thuật toán kNN đầy đủ để lấy K hình ảnh liên quan nhất.
Chúng ta có thể sử dụng mean Average Precision làm thước đo giữa K ground truth và K hình ảnh dự đoán.

## Contributors
HBH - LMHTX
