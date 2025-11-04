
import cv2
import numpy as np
import os

def resize_and_grayscale(image, size=(16, 16)):
    """
    将图像调整为指定尺寸并转为灰度。
    """
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def calculate_difference(img1, img2):
    """
    计算两张灰度图像的差异。
    """
    # 求像素差的绝对值并取平均
    difference = np.abs(img1 - img2)
    avg_difference = np.mean(difference)
    return avg_difference

def is_duplicate(img1, img2, threshold=10):
    """
    判断两张图像是否为重复图像。
    如果图像差异小于阈值，则认为是重复图像。
    """
    difference = calculate_difference(img1, img2)
    return difference < threshold

def compare_with_folder(new_image, folder_path, threshold=10):
    """
    比较新文件与文件夹中所有图像，找出重复图像。
    """
    duplicates = []
    new_image_resized = resize_and_grayscale(new_image)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not filename.endswith(('.jpg', '.png', '.jpeg')):  # 支持的文件类型
            continue
        folder_image = cv2.imread(file_path)
        folder_image_resized = resize_and_grayscale(folder_image)

        if is_duplicate(new_image_resized, folder_image_resized, threshold):
            duplicates.append(file_path)

    return duplicates

