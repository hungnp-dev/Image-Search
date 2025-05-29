import io
import os
import zipfile
import requests
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from tqdm import tqdm
from typing import List, Dict
from fastapi import Response
from PIL import Image
from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from zipfile import ZipFile
from .path_utils import join_paths, ensure_dir, normalize_path, get_file_name

tqdm.pandas()

def read_txt(path: str) -> List[str]:
    data = []
    with open(normalize_path(path), 'r') as file:
        for line in file:
            data.append(line.strip())
    return data

def download_and_extract(url: str, extract_to: str = '.'):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    with ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(normalize_path(extract_to))
    os.remove(local_filename)

def specs(x, **kwargs):
    plt.axvline(x.mean(), c='k', ls='-', lw=1.5, label='mean')
    plt.axvline(x.median(), c='orange', ls='--', lw=1.5, label='median')

def zip_files(filenames: List[Dict[str, str]]) -> Response:
    zip_filename = "images.zip"
    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    for entry in filenames:
        fpath = normalize_path(entry['path'])
        fname = get_file_name(fpath)
        zf.write(fpath, fname)

    zf.close()

    resp = Response(s.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })

    return resp

def calculate_embedding(model, image_path: str) -> Optional[List[float]]:
    try:
        image = Image.open(normalize_path(image_path))
        encoded_im = model.encode(image).tolist()
        image.close()
        return encoded_im
    
    except Exception:
        print(f"Lỗi khi tạo vector đặc trưng cho hình ảnh {image_path}")
        return None

def build_image_embeddings(df: pd.DataFrame, save_path: str = 'resources'):
    embeddings_file = "images_embeddings.parquet"
    embeddings_path = join_paths(save_path, embeddings_file)

    ensure_dir(save_path)
    if os.path.isfile(embeddings_path):
        print("Vector đặc trưng đã được tạo")
        return

    print("Đang xây dựng vector đặc trưng cho hình ảnh...")

    image_model = SentenceTransformer("clip-ViT-B-32")

    try:
        import torch

        if torch.cuda.is_available():
            image_model = image_model.to('cuda')
        else:
            print("CUDA không khả dụng. Sử dụng CPU thay thế. Quá trình này có thể mất nhiều thời gian...")
    except ModuleNotFoundError:
        print("Torch chưa được cài đặt")

    df["embedding"] = df["path"].progress_apply(lambda x: calculate_embedding(image_model, x))
    df["embedding"] = df["embedding"].replace({None: np.nan})
    df = df.dropna(subset=["embedding"])

    df.to_parquet(embeddings_path)

    print("Hoàn thành")

def update_db_collection(collection_name: str = 'images',
                         vectors_dir_path: str = 'resources',
                         m: int = 16,
                         ef_construct: int = 100):
    embeddings_path = join_paths(vectors_dir_path, "images_embeddings.parquet")
    if not os.path.isfile(embeddings_path):
        print("Vector đặc trưng không có trong thư mục bạn đã chỉ định hoặc chưa được tạo!")
        return

    qdrant_client = QdrantClient("http://localhost:6333")

    im_df = pd.read_parquet(embeddings_path)
    print(f"Có {len(im_df)} hình ảnh.")

    # Chuyển dữ liệu thành list để dễ xử lý
    paths = [normalize_path(p) for p in im_df['path'].values.tolist()]
    vectors = list(map(list, im_df["embedding"].tolist()))
    
    print("Đang đưa vector đặc trưng vào collection Qdrant...")

    # Xóa collection cũ nếu tồn tại
    try:
        qdrant_client.delete_collection(collection_name=collection_name)
        print("Đã xóa collection cũ")
    except:
        print("Không có collection cũ để xóa")

    # Tạo collection mới với cấu hình tối ưu
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=512,
            distance=Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(
            m=m,
            ef_construct=ef_construct
        ),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=2
        ),
        on_disk_payload=True  # Lưu payload trên đĩa để tiết kiệm RAM
    )
    print("Đã tạo collection mới")

    # Upload vectors theo batch nhỏ
    batch_size = 100  # Batch size nhỏ hơn để tránh quá tải
    total_batches = len(vectors) // batch_size + (1 if len(vectors) % batch_size != 0 else 0)
    
    print(f"Bắt đầu upload {len(vectors)} vectors theo {total_batches} batches...")
    
    for i in range(0, len(vectors), batch_size):
        batch_vectors = vectors[i:i + batch_size]
        batch_payloads = [{'path': p} for p in paths[i:i + batch_size]]
        batch_ids = list(range(i, min(i + batch_size, len(vectors))))
        
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=batch_ids,
                    vectors=batch_vectors,
                    payloads=batch_payloads
                )
            )
            print(f"Đã upload batch {i//batch_size + 1}/{total_batches}")
        except Exception as e:
            print(f"Lỗi khi upload batch {i//batch_size + 1}: {str(e)}")
            continue

    # Kiểm tra kết quả
    try:
        count_info = qdrant_client.count(collection_name=collection_name)
        print(f"Hoàn thành! Đã tạo collection '{collection_name}' với {count_info.count} vectors")
    except Exception as e:
        print(f"Không thể kiểm tra số lượng vectors: {str(e)}")
        print("Nhưng quá trình upload đã hoàn tất")
