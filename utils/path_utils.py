import os
import platform

def get_platform():
    """Trả về hệ điều hành hiện tại"""
    return platform.system().lower()

def normalize_path(path):
    """Chuẩn hóa đường dẫn cho phù hợp với hệ điều hành"""
    return os.path.normpath(path)

def get_relative_path(path):
    """Chuyển đổi đường dẫn tuyệt đối thành đường dẫn tương đối"""
    return os.path.relpath(path)

def join_paths(*paths):
    """Kết hợp các phần của đường dẫn một cách an toàn"""
    return normalize_path(os.path.join(*paths))

def get_file_name(path):
    """Lấy tên file từ đường dẫn"""
    return os.path.basename(path)

def ensure_dir(path):
    """Đảm bảo thư mục tồn tại, tạo nếu chưa có"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def convert_path_for_url(path):
    """Chuyển đổi đường dẫn để sử dụng trong URL"""
    # Luôn sử dụng forward slash cho URL
    return path.replace(os.sep, '/') 