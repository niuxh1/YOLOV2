import os
import requests
import zipfile
import tarfile


def download_file(url, save_path):
    """下载文件的函数，显示下载进度"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))  # 获取文件总大小
    downloaded_size = 0  # 已下载大小

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                # 计算并打印下载进度
                progress = (downloaded_size / total_size) * 100
                print(f"下载进度: {progress:.2f}%", end='\r')

    print(f"\n下载完成: {save_path}")


def extract_tar(tar_path, extract_to):
    """解压 tar 文件的函数"""
    with tarfile.open(tar_path, 'r') as tar_ref:
        tar_ref.extractall(extract_to)


def download_voc2012(data_dir):
    """下载 VOC 2012 数据集"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # VOC 2012 数据集 URL
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    file_name = "VOCtrainval_11-May-2012"
    tar_path = os.path.join(data_dir, f"{file_name}.tar")

    print(f"正在下载 {file_name} ...")
    download_file(url, tar_path)

    # 解压文件
    print(f"正在解压 {file_name} ...")
    extract_tar(tar_path, data_dir)

    # 删除 tar 文件
    os.remove(tar_path)
    print(f"{file_name} 下载和解压完成！")


if __name__ == "__main__":
    download_voc2012(data_dir=r'C:\yolov2\voc')
