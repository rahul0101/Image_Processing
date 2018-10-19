import numpy as np
import cv2
import matplotlib.pyplot as plt

imgpath = "E:\\FM\\Images\\lena_color_512.tif"
img = cv2.imread(imgpath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def max_rgb(image):
    (B, G, R) = cv2.split(image)

    M = np.maximum(np.maximum(R, G), B)
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    X = cv2.merge([B, G, R])

    return X

x = max_rgb(img)
plt.imshow(x)
plt.show()