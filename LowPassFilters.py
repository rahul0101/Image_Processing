import matplotlib.pyplot as plt
import cv2

imgpath = "E:\\FM\\Images\\lena_color_512.tif"
img = cv2.imread(imgpath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

box = cv2.boxFilter(img, -1, (53, 53))

blur = cv2.blur(img, (13, 13))

gaussian = cv2.GaussianBlur(img, (37, 37), 0)

median = cv2.medianBlur(img, 3)

bil = cv2.bilateralFilter(img, 9, 15, 80, 80)

outputs = [img, box, blur, gaussian, median, bil]

for i in range(6):
    plt.subplot(3, 2, i + 1)
    plt.imshow(outputs[i])
plt.show()
