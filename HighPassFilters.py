import matplotlib.pyplot as plt
import cv2

imgpath = "E:\\FM\\misc2.jpg"
img = cv2.imread(imgpath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

laplacian = cv2.Laplacian(img,cv2.CV_8U)

sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3)

sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)

outputs = [img, laplacian, sobelx, sobely]

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(outputs[i])
plt.show()
