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
    # contour
    img[markers == -1] = [255, 255, 255]
    # background
    img[markers == 1] = [0, 0, 0]

    for marker in range(2, markers.max() + 1):
        color = np.random.choice(range(256), size=3)
        img[markers == marker] = color

    return img, count


img = cv.imread('LicensePlates.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (45, 45), 0)
thresh = cv.threshold(blur, 180, 255, cv.THRESH_BINARY)[1]

binary = cv.bitwise_not(thresh)

kernel = np.ones((7, 7), np.uint8)
erode = cv.erode(binary, kernel, iterations=1)

img, count = color_pools(img, binary, erode)
res_title = "Plates Count = " + str(count)

contours, _ = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
for c in contours:
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.15 * peri, True)
    cv.drawContours(img, [c], -1, (255, 255, 255), 2)

# plot
plt.subplot(221), plt.imshow(gray, cmap='gray')
plt.title('Input'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(blur, cmap='gray')
plt.title('Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(binary, cmap='gray')
plt.title('Binary'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img, cmap='gray')
plt.title(res_title), plt.xticks([]), plt.yticks([])

plt.show()
