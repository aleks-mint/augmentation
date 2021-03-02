# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 19:59:39 2021

@author: Admin
"""
"""
RandomRotate90, +
Flip, +
Transpose, +-
GaussNoise, +
MedianBlur, +-
ShiftScaleRotate, 
RandomBrightness,
HueSaturationValue
"""
import cv2 as cv
import numpy as np
import os


#методы аугментации
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    image_center = (w / 2, h / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

def flip_image(image, param):
    result = cv.flip(image, param)
    return result

def perspective_image(image):
    rows, cols = image.shape[:2]
    pts1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
    pts2 = np.float32([[0,200], [cols, 0], [0, rows], [cols, rows+200]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(image, M, (cols, rows+200))
    return dst


def elastic_transform_image(image):
    img = image

    A = img.shape[1] / 3.0
    w = 2.0 / img.shape[0]

    shift = lambda x: 0.1 * A * np.sin(2*np.pi*x * w)

    for i in range(img.shape[1]):
        img[:, i] = np.roll(img[:, i], int(shift(i)))
    
    return img

"""
def emboss_image(image):
import cv2
import numpy as np

img_emboss_input = cv2.imread('input.jpg')

# generating the kernels
kernel_emboss_1 = np.array([[0,-1,-1],
                            [1,0,-1],
                            [1,1,0]])
kernel_emboss_2 = np.array([[-1,-1,0],
                            [-1,0,1],
                            [0,1,1]])
kernel_emboss_3 = np.array([[1,0,0],
                            [0,0,0],
   ...
"""

def gauss_noise_image(image, mean, var):
    A = np.double(image)
    out = np.zeros(A.shape, np.double)
    cv.normalize(A, out, 1.0, 0.0, cv.NORM_MINMAX)    
    image = out       
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise            
    result = np.clip(out, 0, 1.0)        
    return result

def bilateral_filter_image(image):
    result = cv.bilateralFilter(image, 15, 75, 75)
    return result

def gauss_blur_image(image):
    result = cv.GaussianBlur(image, (5, 5), 0)
    return result

def median_blur_image(image):
    result = cv.medianBlur(image, 5)
    return result

def blur_image(image):
    result = cv.blur(image, (5, 5))
    return result
    





#путь к папкам исходных и аугментированных фото
input_path = "C:/Users/Admin/Desktop/images/"
output_path = "C:/Users/Admin/Desktop/aug images/"

path, dirs, files = next(os.walk(input_path))
file_count = len(files)
print(file_count)


for i in range(1, file_count):
    
    if i <= 9:
        ima = cv.imread(input_path + "0" + str(i) + ".jpg")
    else:
        ima = cv.imread(input_path + str(i) + ".jpg")
        
    ima2 = gauss_noise_image(ima, 0.1, 0.01)
    ima2 = np.uint8(ima2*255)
        
    ima3 = ima2
        
    cv.imwrite(output_path + "1-" + str(i)+".jpg", flip_image(ima3, -1))
    cv.imwrite(output_path + "2-" + str(i)+".jpg", flip_image(ima3, 0))
    cv.imwrite(output_path + "3-" + str(i)+".jpg", flip_image(ima3, 1))
    cv.imwrite(output_path + "4-" + str(i)+".jpg", rotate_image(ima3, 45))
    cv.imwrite(output_path + "5-" + str(i)+".jpg", rotate_image(ima3, 75))
    cv.imwrite(output_path + "6-" + str(i)+".jpg", ima2)
    cv.imwrite(output_path + "7-" + str(i)+".jpg", rotate_image(flip_image(ima3, 1), 45))
    cv.imwrite(output_path + "8-" + str(i)+".jpg", rotate_image(flip_image(ima3, 1), 75))
    cv.imwrite(output_path + "9-" + str(i)+".jpg", rotate_image(flip_image(ima3, 0), 45))
    cv.imwrite(output_path + "10-" + str(i)+".jpg", rotate_image(flip_image(ima3, 0), 75))
   #cv.imwrite("C:/Users/Admin/Desktop/aug image/" + "11-" + str(i)+".jpg", elastic_transform_image(ima3))
   #cv.imwrite("C:/Users/Admin/Desktop/aug image/" + "7-" + str(i)+".jpg", bilateral_filter_image(ima2))
   #cv.imwrite("C:/Users/Admin/Desktop/aug image/" + "8-" + str(i)+".jpg", gauss_blur_image(ima2))
        
        
    cv.waitKey(0)
    cv.destroyAllWindows()
    
