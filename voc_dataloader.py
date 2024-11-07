import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VOC2012Dataset(Dataset):
    def __init__(self, data_root, image_set='train', transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])):
        self.data_root = data_root
        self.image_set = image_set
        self.transform = transform

        self.image_list = self.load_image_set()

    def load_image_set(self):
        # 加载图像集文件路径
        image_set_file = os.path.join(self.data_root, 'ImageSets', 'Main', f'{self.image_set}.txt')
        with open(image_set_file, 'r') as f:
            return [x.strip() for x in f.readlines()]

    def load_annotations(self, index):
        # 加载对应索引的标注
        image_id = self.image_list[index]
        annotation_file = os.path.join(self.data_root, 'Annotations', f'{image_id}.xml')

        boxes = []
        labels = []
        tree = ET.parse(annotation_file)
        for obj in tree.findall('object'):
            label = obj.find('name').text
            labels.append(label)
            bndbox = obj.find('bndbox')
            boxes.append([
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            ])
        return boxes, labels

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_id = self.image_list[index]
        image_path = os.path.join(self.data_root, 'JPEGImages', f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')  # 加载图像
        image = self.transform(image)  # 应用转换

        boxes, labels = self.load_annotations(index)
        return image, boxes, labels


# 示例代码用于测试数据加载器
if __name__ == '__main__':
    data_root = 'path_to_voc_dataset'  # 替换为你的数据集路径
    dataset = VOC2012Dataset(data_root, image_set='train')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for images, boxes, labels in data_loader:
        print(images.shape)  # 打印图像的形状
        print(boxes)  # 打印标注框
        print(labels)  # 打印标签
