# utils/voc_dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms

class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, input_size, classes, train=True):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.input_size = input_size
        self.classes = classes
        self.train = train
        self.image_list = os.listdir(image_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_id = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_id)
        annotation_path = os.path.join(self.annotation_dir, image_id.replace('.jpg', '.xml'))

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        boxes = []
        labels = []

        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label not in self.classes:
                continue
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[label])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        targets = torch.zeros((len(boxes), 8))
        targets[:, 1:5] = boxes
        targets[:, 6] = labels

        return image, targets, image_id