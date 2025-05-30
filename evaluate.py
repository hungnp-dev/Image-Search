import argparse
from utils.search import Text2Img
from utils.utils import read_txt

def get_cli_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_file", help="Path to labels file", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_cli_arg()
    labels_filepath = args.labels_file
    text2img = Text2Img()
    test_dataset = read_txt(path=labels_filepath)
    accuracy, mapping = text2img.avg_precision_at_k(test_dataset, k=5)
    print(f"Đối với các truy vấn đã cung cấp, độ chính xác của hệ thống là {accuracy}.")
    print("Ánh xạ giữa các lớp được định nghĩa trong tệp nhãn và các hình ảnh phổ biến được truy xuất bởi cả ANN và kNN")
    for class_name, images in mapping.items():
        print(f"{class_name}: \n {images}")
