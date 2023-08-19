import torch
import matplotlib
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import cv2 as cv

import platform
import os
import zipfile
import glob
import math
import wget
import random

def download_bar(current, total, width=30):
    downloaded = int(math.floor(width * (float(current) / total)))
    progress = '[' + ('*' * downloaded) + ('-' * (width - downloaded)) + '] ' + f' {current/total*100:.2f}%'
    return progress


def download_coco(root_dir='./', year='2017', remove_compressed_file=True):
    coco_train_url = f'http://images.cocodataset.org/zips/train{year}.zip'
    coco_val_url = f'http://images.cocodataset.org/zips/val{year}.zip'
    coco_test_url = f'http://images.cocodataset.org/zips/test{year}.zip'
    coco_trainval_anno_url = f'http://images.cocodataset.org/annotations/annotations_trainval{year}.zip'

    os.makedirs(root_dir, exist_ok=True)
    anno_dir = os.path.join(root_dir, 'annotations')
    os.makedirs(anno_dir, exist_ok=True)

    # train, val, test 데이터셋 다운로드
    if (os.path.exists(os.path.join(root_dir, 'train' + year))
                       and os.path.exists(os.path.join(root_dir, 'val' + year))):
        print('train, val이 이미 존재합니다.')
        return

    print("다운로드 중...")

    # wget을 통해 데이터셋을 다운로드합니다.
    wget.download(coco_train_url, root_dir, bar=download_bar)
    wget.download(coco_val_url, root_dir, bar=download_bar)
    wget.download(coco_test_url, root_dir, bar=download_bar)
    wget.download(coco_trainval_anno_url, root_dir, bar=download_bar)

    with zipfile.ZipFile(os.path.join(root_dir, 'train' + year + '.zip')) as unzip:
        unzip.extractall(root_dir)
    with zipfile.ZipFile(os.path.join(root_dir, 'val' + year + '.zip')) as unzip:
        unzip.extractall(root_dir)
    with zipfile.ZipFile(os.path.join(root_dir, 'annotations_trainval' + year + '.zip')) as unzip:
        unzip.extractall(root_dir)

    # 다운로드한 zip 파일 삭제
    if remove_compressed_file:
        root_zip_list = glob.glob(os.path.join(root_dir, '*.zip'))
        for zip in root_zip_list:
            os.remove(zip)

    print("다운로드 완료!")



'''
cocoapi를 이용해 COCO 데이터셋을 불러오는 코드입니다.
:param root_dir: COCO 데이터셋이 저장된 경로
:param name: COCO 데이터셋의 이름 (ex. train2017, val2017)
:param split: train, val 중 어떤 데이터셋인지를 나타냅니다. (TRAIN, VAL)
'''
class COCODataset(Dataset):
    def __init__(self,
                 root_dir='./coco',
                 name='train',
                 year='2017',
                 split='train',
                 resize=None,
                 download=True,
                 transform=None,
                 visualization=False):

        super().__init__()

        if platform.system() == 'Windows':
            matplotlib.use('TkAgg')

        assert split in ['train', 'val', 'test']

        self.root_dir = root_dir
        self.set_name = name + year
        self.split = split
        self.resize = resize
        if self.resize is None:
            self.resize = 600

        self.download = download
        if self.download:
            download_coco(root_dir=root_dir, year=year)

        self.transform = transform
        self.visualization = visualization

        self.img_path = glob.glob(os.path.join(self.root_dir, self.set_name, '*.jpg'))

        # COCO를 이용해 데이터셋을 불러옵니다.
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.img_id = list(self.coco.imgToAnns.keys())
        self.coco_ids = sorted(self.coco.getCatIds()) # 1 ~ 90 (len = 80)
        self.coco_ids_to_continuous_ids = {coco_id: i for i, coco_id in enumerate(self.coco_ids)} # 0~79로 매핑
        self.coco_ids_to_class_names = {category['id']: category['name'] for category in
                                        self.coco.loadCats(self.coco_ids)} # { 1: 'person', 2: 'bicycle', ... }
        # print(self.img_id)
        # print(self.coco_ids)
        # print(self.coco_ids_to_continuous_ids)
        # print(self.coco_ids_to_class_names)


    def __getitem__(self, idx):
        img_id = self.img_id[idx]
        image = self._load_image(img_id)
        annotation = self._load_annotations(img_id)

        boxes, labels = self._parse_coco(annotation)
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.visualization:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # tensor to img
            img_vis = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            # img_vis = np.array(image.permute(1,2,0), np.float32)
            # img_vis = img_vis * std + mean
            # img_vis = np.clip(img_vis, 0, 1)

            plt.figure('input')
            plt.imshow(img_vis)

            for i in range(len(boxes)):
                new_h_scale = new_w_scale = 1
                #new_h_scale, new_w_scale = image.shape[:2]

                x1 = boxes[i][0] * new_w_scale
                y1 = boxes[i][1] * new_h_scale
                x2 = boxes[i][2] * new_w_scale
                y2 = boxes[i][3] * new_h_scale

                color = np.random.randint(0, 255, size=3) / 255

                plt.text(x=x1-5, y=y1-5, s=str(self.coco_ids_to_class_names[self.coco_ids[labels[i]]]),
                         bbox=dict(boxstyle='round4',
                                   facecolor=color,
                                   alpha=0.9))
                plt.gca().add_patch(plt.Rectangle(xy=(x1, y1), width=x2-x1, height=y2-y1,
                                                  linewidth=1, edgecolor=color, facecolor='none'))
            plt.show()

            return image, boxes, labels


    def _load_image(self, img_id):
        file_name = self.coco.loadImgs(img_id)[0]["file_name"]
        print("root: ", os.path.join(self.root_dir, self.set_name, file_name))
        return cv.imread(os.path.join(self.root_dir, self.set_name, file_name))


    def _load_annotations(self, img_id):
        annotation = self.coco.loadAnns(ids=self.coco.getAnnIds(imgIds=img_id))
        return annotation

    def _parse_coco(self, anno, type='bbox'):
        if type == 'segm':
            return -1
        annotations = np.zeros((0, 5))
        # width, height가 1 이하인 annotation은 제외합니다.
        for idx, anno_dict in enumerate(anno):
            if anno_dict['bbox'][2] < 1 or anno_dict['bbox'][3] < 1:
                continue
            annotation = np.zeros((1, 5))
            annotation[0, :4] = anno_dict['bbox']
            annotation[0, 4] = self.coco_ids_to_continuous_ids[anno_dict['category_id']]
            annotations = np.append(annotations, annotation, axis=0)
        # [x, y, w, h] -> [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        # print('bbox and labels: ', annotations)

        return annotations[:, :4], annotations[:, 4]




