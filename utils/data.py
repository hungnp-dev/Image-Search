import shutil
import os
import io
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from jinja2 import Environment, FileSystemLoader
from .utils import build_image_embeddings, update_db_collection, download_and_extract, specs
from .path_utils import join_paths, ensure_dir, normalize_path

class Preparator:
    def __init__(self,
                 imgs_path: str = 'images',
                 docs_path: str = 'resources',
                 collection_name: str = 'images',
                 m: int = 16,
                 ef_construct: int = 100
                 ):
        self.imgs_path = normalize_path(imgs_path)
        self.docs_path = normalize_path(docs_path)
        self.collection_name = collection_name
        self.m = m
        self.ef_construct = ef_construct
        self.im_df = None

    def run(self):
        self.get_data()
        self.im_df = self.store_image_info()
        self.create_report()
        self.build_embeddings()
        self.update_collection()

    def get_data(self):
        if os.path.isdir(self.imgs_path):
            images = os.listdir(self.imgs_path)
            if len(images) > 0:
                print("Dữ liệu đã được tải về trước đó.")
                return

        urls = ['https://storage.googleapis.com/ads-dataset/subfolder-0.zip',
                'https://storage.googleapis.com/ads-dataset/subfolder-1.zip']

        ensure_dir(self.imgs_path)

        print("Đang tải dữ liệu...")
        for url in tqdm(urls):
            download_and_extract(url, extract_to=self.imgs_path)
        print("Hoàn thành")

        print("Đang tổ chức dữ liệu...")
        for folder in ['0', '1']:
            src_path = join_paths(self.imgs_path, folder)
            for filename in os.listdir(src_path):
                shutil.move(join_paths(src_path, filename), self.imgs_path)

            shutil.rmtree(src_path)
        print("Hoàn thành")

    def store_image_info(self) -> pd.DataFrame:
        ensure_dir(self.docs_path)
        data_info_path = join_paths(self.docs_path, 'data_info.csv')
        
        if os.path.isfile(data_info_path):
            print("Thông tin hình ảnh đã tồn tại. Đang đọc...")
            results_df = pd.read_csv(data_info_path)
            print("Hoàn thành.")
            return results_df

        all_imgs = os.listdir(self.imgs_path)
        columns_df = ['path', 'width', 'height', 'area', 'aspect_ratio']
        imgs_df = pd.DataFrame(columns=columns_df)

        print("Đang lưu thông tin hình ảnh...")
        for im_name in tqdm(all_imgs):
            im_path = join_paths(self.imgs_path, im_name)
            im = Image.open(im_path)
            w, h = im.size
            new_df = pd.DataFrame([[im_path, w, h, w * h, w / h]], columns=columns_df)
            
            if imgs_df.empty:
                imgs_df = new_df.copy()
            else:
                imgs_df = pd.concat([imgs_df, new_df])

        result_df = imgs_df.reset_index(drop=True)
        result_df.to_csv(data_info_path, index=False)
        print("Hoàn thành.")
        return result_df

    def create_report(self):
        if not os.path.exists(self.docs_path):
            os.makedirs(self.docs_path)

        data_report_path = join_paths(self.docs_path, 'data_report.html')
        if os.path.isfile(data_report_path):
            print("Báo cáo đã được tạo trước đó.")
            return

        print("Đang tạo báo cáo phân tích dữ liệu...")

        areas_sorted = self.im_df.sort_values(by='area', ascending=False)
        highest_resolution = areas_sorted.iloc[0][['width', 'height']].values
        lowest_resolution = areas_sorted.iloc[-1][['width', 'height']].values

        plot_path1 = join_paths(self.docs_path, 'area_distrib.png')
        sns_plot = sns.displot(self.im_df, x="aspect_ratio", kde=True).set(title="Phân phối tỷ lệ khung hình")
        sns_plot.map(specs, 'aspect_ratio')
        plt.legend()
        sns_plot.savefig(plot_path1)
        plt.close(sns_plot.fig)

        plot_path2 = join_paths(self.docs_path, 'width_height_distrib.png')
        sns_plot = sns.jointplot(x='width', y='height', data=self.im_df)
        plt.suptitle("Phân phối chiều rộng và chiều cao")
        sns_plot.savefig(plot_path2)
        plt.close(sns_plot.fig)

        random_imgs = np.random.choice(self.im_df['path'], size=5, replace=False)

        plot_path3 = join_paths(self.docs_path, 'images.png')
        fig, axes = plt.subplots(1, 5, figsize=(12, 6))
        for i, im_path in enumerate(random_imgs):
            im = Image.open(im_path)
            axes[i].imshow(im)
            axes[i].axis('off')
        fig.savefig(plot_path3)
        plt.close()

        data = {
            'nr_imgs': len(self.im_df),
            'high_res': f"{highest_resolution[0]} x {highest_resolution[1]}",
            'low_res': f"{lowest_resolution[0]} x {lowest_resolution[1]}",
            'plot_paths': [os.path.join('..', plot_path1), os.path.join('..', plot_path2)],
            'images': os.path.join('..', plot_path3)
        }

        env = Environment(loader=FileSystemLoader('templates'))
        env.charset = 'utf-8'
        template = env.get_template('data_report_template.html')

        html_output = template.render(**data)

        try:
            with io.open(data_report_path, 'w', encoding='utf-8') as file:
                file.write(html_output)
        except UnicodeEncodeError:
            with open(data_report_path, 'wb') as file:
                file.write(html_output.encode('utf-8'))

        print("Hoàn thành.")

    def build_embeddings(self):
        build_image_embeddings(self.im_df, self.docs_path)

    def update_collection(self):
        update_db_collection(self.collection_name, self.docs_path, self.m, self.ef_construct)