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

tqdm.pandas()

def read_txt(path: str) -> List[str]:
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(line.strip())
    return data

def download_and_extract(url: str, extract_to: str = '.'):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    with ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(local_filename)

def specs(x, **kwargs):
    plt.axvline(x.mean(), c='k', ls='-', lw=1.5, label='mean')
    plt.axvline(x.median(), c='orange', ls='--', lw=1.5, label='median')

def zip_files(filenames: List[Dict[str, str]]) -> Response:
    zip_filename = "images.zip"
    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    for entry in filenames:
        fpath = entry['path']
        fdir, fname = os.path.split(fpath)
        zf.write(fpath, fname)

    zf.close()

    resp = Response(s.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })

    return resp

def calculate_embedding(model, image_path: str) -> Optional[List[float]]:
    try:
        image = Image.open(image_path)
        encoded_im = model.encode(image).tolist()
        image.close()
        return encoded_im
    
    except Exception:
        print(f"Lỗi khi tạo vector đặc trưng cho hình ảnh {image_path}")
        return None

def build_image_embeddings(df: pd.DataFrame, save_path: str = 'resources'):
    embeddings_file = "images_embeddings.parquet"
    embeddings_path = os.path.join(save_path, embeddings_file)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
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
    embeddings_path = os.path.join(vectors_dir_path, "images_embeddings.parquet")
    if not os.path.isfile(embeddings_path):
        print("Vector đặc trưng không có trong thư mục bạn đã chỉ định hoặc chưa được tạo!")
        return

    qdrant_client = QdrantClient("http://localhost:6333")

    im_df = pd.read_parquet(embeddings_path)
    print(f"Có {len(im_df)} hình ảnh.")

    paths = im_df['path'].values
    payloads = iter([{'path': p} for p in paths])
    vectors = iter(list(map(list, im_df["embedding"].tolist())))

    print("Đang đưa vector đặc trưng vào collection Qdrant...")

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    qdrant_client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=payloads,
        ids=None,
        batch_size=256
    )

    while True:
        collection_info = qdrant_client.get_collection(collection_name=collection_name)
        if collection_info.status == models.CollectionStatus.GREEN:
            break

    print(f"Có {qdrant_client.count(collection_name)} điểm đã được tạo "
          f"trong collection Qdrant có tên '{collection_name}'")
