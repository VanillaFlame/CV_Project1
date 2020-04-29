import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def color_pools(img, base, mask):
    _, markers = cv.connectedComponents(base)
    markers = markers + 1
    markers[mask == 255] = 0
    markers = cv.watershed(img, markers)

    # result
    count = markers.max() - 1
    # background
    img[markers == 1] = [0, 0, 0]

    for marker in range(2, markers.max()):
        color = np.random.choice(range(256), size=3)
        img[markers == marker] = color

    return img, count


img = cv.imread('Coins.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (25, 25), 0)
canny = cv.Canny(blur, 21, 41)

kernel = np.ones((3, 3), np.uint8)
dilate = cv.dilate(canny, kernel, iterations=1)
erode = cv.erode(dilate, kernel, iterations=1)

img, count = color_pools(img, dilate, erode)
res_title = "Coins Count = " + str(count)

# plot
plt.subplot(221), plt.imshow(gray, cmap='gray')
plt.title('Input'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(canny, cmap='gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(dilate, cmap='gray')
plt.title('Dilate'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img, cmap='gray')
plt.title(res_title), plt.xticks([]), plt.yticks([])

plt.show()
