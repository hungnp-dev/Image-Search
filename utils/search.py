import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Set
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from .path_utils import normalize_path

class Text2Img:
    def __init__(self, collection_name: str = 'images'):
        self.collection_name = collection_name
        self.text_encoder = SentenceTransformer("clip-ViT-B-32", device="cpu")
        self.qdrant_client = QdrantClient("http://localhost:6333")

    def search(self, text: str) -> List[Dict[str, str]]:
        vector = self.text_encoder.encode(text).tolist()
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            with_payload=True,
            limit=5,
        )
        payloads = [{'path': normalize_path(hit.payload['path'])} for hit in search_result]
        return payloads

    def avg_precision_at_k(self, test_dataset: List[str], k: int = 5) -> Tuple[float, Dict[str, Set[str]]]:
        precisions = []
        common_images_mapping = {}

        print("Đang đánh giá tập dữ liệu tùy chỉnh...")
        for item in tqdm(test_dataset):
            vector = self.text_encoder.encode(item).tolist()

            ann_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=k,
            )

            knn_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=k,
                search_params=models.SearchParams(
                    exact=True,
                ),
            )

            ann_ids = set(item.id for item in ann_result)
            knn_ids = set(item.id for item in knn_result)

            common_indexes = ann_ids.intersection(knn_ids)

            mask = np.isin(list(common_indexes), list(knn_ids))
            common_results = np.array(knn_result)[mask]
            common_images = [normalize_path(res.payload['path']) for res in common_results]
            common_images_mapping[item] = set(common_images)

            precision = len(common_indexes) / k
            precisions.append(precision)

        return sum(precisions) / len(precisions), common_images_mapping
