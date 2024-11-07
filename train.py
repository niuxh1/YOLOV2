import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from voc_dataloader import VOC2012Dataset  # 引用 VOC 数据加载器
from yolov2_d19 import yolo_v2_d19  # 引用 YOLOv2 模型
import tools_for_yolov2 as tools  # 引用工具模块


def train(model, dataloader, optimizer, device, num_epochs):
    for param in model.parameters():
        param.requires_grad = True
    for epoch in range(num_epochs):
        total_conf_loss, total_cls_loss, total_box_loss, total_iou_loss = 0, 0, 0, 0
        for images, boxes, labels in dataloader:  # 修改这里
            images = images.to(device)

            # 清除梯度
            optimizer.zero_grad()

            # 前向传播
            conf_loss, cls_loss, box_loss = model(images, targets=(boxes, labels))  # 修改这里以使用 boxes 和 labels

            # 反向传播
            loss = conf_loss.float().mean() + cls_loss.float().mean() + box_loss.float().mean()
            assert conf_loss.requires_grad, "conf_loss does not require grad"
            assert cls_loss.requires_grad, "cls_loss does not require grad"
            assert box_loss.requires_grad, "box_loss does not require grad"
            loss.backward()
            optimizer.step()

            total_conf_loss += conf_loss.item()
            total_cls_loss += cls_loss.item()
            total_box_loss += box_loss.item()


        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Conf Loss: {total_conf_loss:.4f}, "
              f"Cls Loss: {total_cls_loss:.4f}, "
              f"Box Loss: {total_box_loss:.4f}, ")


if __name__ == "__main__":
    # 设置超参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_size = 416  # 根据模型设置
    num_classes = 20  # 根据数据集设置
    anchors_size = [
        [116, 90],
        [156, 198],
        [373, 326],
        [30, 61],
        [62, 45],
        [59, 119],
        [10, 13],
        [45, 60],
        [30, 30]
    ]  # 根据 YOLOv2 的锚框设置
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001

    # 数据增强和数据加载
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    dataloader = DataLoader(
        VOC2012Dataset(data_root=r'C:\yolov2\voc\VOCdevkit\VOC2012', transform=transform),  # 替换为你的数据集路径
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=tools.collate_fn  # 确保使用自定义的 collate_fn
    )

    # 初始化模型、优化器
    model = yolo_v2_d19(device=device, input_size=input_size, num_classes=num_classes, anchors_size=anchors_size).to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    print("开始训练...")
    train(model, dataloader, optimizer, device, num_epochs)
