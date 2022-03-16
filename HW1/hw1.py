import os
import cv2
import copy
from tqdm import tqdm
import numpy as np

SRC_FOLDER = '/Users/jimyang/Documents/相片/Wall_Paper'
SRC_FILE = 'DSC_3019.JPG'

def gaussianBlur(ori_img, size=3):
    img = copy.deepcopy(ori_img)
    img1 = copy.deepcopy(ori_img)
    x, y = np.mgrid[-int(size/2):int(size/2)+1, -int(size/2):int(size/2)+1]
    kernel = np.exp(-(x**2+y**2))
    kernel /= kernel.sum()

    border = cv2.copyMakeBorder(
        img,
        top     = int(size/2),
        bottom  = int(size/2),
        left    = int(size/2),
        right   = int(size/2),
        borderType = cv2.BORDER_REPLICATE)

    for h in tqdm(range(img.shape[0])):
        for w in range(img.shape[1]):
            img[h, w] = np.sum(border[h:h+size, w:w+size] * kernel)
    
    return img, cv2.filter2D(img1, -1, kernel)

def sharpen(ori_img):
    size=3
    img = copy.deepcopy(ori_img)
    img1 = copy.deepcopy(ori_img)
    kernel = np.array([ [0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

    border = cv2.copyMakeBorder(
        img,
        top     = int(size/2),
        bottom  = int(size/2),
        left    = int(size/2),
        right   = int(size/2),
        borderType = cv2.BORDER_REPLICATE)

    for h in tqdm(range(img.shape[0])):
        for w in range(img.shape[1]):
            img[h, w] = np.clip(np.sum(border[h:h+size, w:w+size] * kernel), 0, 255)
    
    return img, cv2.filter2D(img1, -1, kernel)

def outline(ori_img):
    size=3
    img = copy.deepcopy(ori_img)
    img1 = copy.deepcopy(ori_img)
    kernel = np.array([ [-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])

    border = cv2.copyMakeBorder(
        img,
        top     = int(size/2),
        bottom  = int(size/2),
        left    = int(size/2),
        right   = int(size/2),
        borderType = cv2.BORDER_REPLICATE)

    for h in tqdm(range(img.shape[0])):
        for w in range(img.shape[1]):
            img[h, w] = np.clip(np.sum(border[h:h+size, w:w+size] * kernel), 0, 255)
    
    return img, cv2.filter2D(img1, -1, kernel)

def main():
    # Read image
    input_img = cv2.imread(os.path.join(SRC_FOLDER, SRC_FILE), cv2.IMREAD_GRAYSCALE)
    input_img = cv2.resize(input_img, (int(input_img.shape[1]*0.25), int(input_img.shape[0]*0.25)))
    cv2.imwrite("./Input.JPG", input_img)
    print(f"Image Size: {input_img.shape}")

    # GaussianBlur
    my_GaussianBlur_img, GaussianBlur_img = gaussianBlur(input_img, size=9)
    cv2.imwrite("./Blur.JPG", my_GaussianBlur_img)
    cv2.imwrite("./Blur_filter2d.JPG", GaussianBlur_img)

    # Sharpen
    my_Sharpen_img, Sharpen_img = sharpen(input_img)
    cv2.imwrite("./Sharpen.JPG", my_Sharpen_img)
    cv2.imwrite("./Sharpen_filter2d.JPG", Sharpen_img)

    # Outline
    my_Outline_img, Outline_img = outline(input_img)
    cv2.imwrite("./Outline.JPG", my_Outline_img)
    cv2.imwrite("./Outline_filter2d.JPG", Outline_img)

if __name__ == '__main__':
    main()