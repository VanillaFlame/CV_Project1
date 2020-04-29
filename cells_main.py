import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def color_pools(img, base, mask):
    _, markers = cv.connectedComponents(base)
    markers[mask == 255] = 0
    markers = cv.watershed(img, markers)

    # result
    count = markers.max() - 1
    # contour
    img[markers == -1] = [255, 255, 255]
    # background
    img[markers == 1] = [0, 0, 0]

    for marker in range(2, markers.max()):
        color = np.random.choice(range(256), size=3)
        img[markers == marker] = color

    return img, count


img = cv.imread('Cells.png')

kernel_ones_3x3 = np.ones((7, 7), np.uint8)
kernel_ellipse_7x7 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
kernel_ellipse_3x3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

# binarize
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (17, 17), 0)
binary = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10)
blur1 = cv.GaussianBlur(binary, (9, 9), 0)
binary = cv.adaptiveThreshold(blur1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 10)

median_filter = cv.medianBlur(binary, 11)

closing = cv.morphologyEx(median_filter, cv.MORPH_CLOSE, kernel_ellipse_3x3, iterations=3)
opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel_ones_3x3, iterations=4)

dilate = cv.erode(opening, kernel_ellipse_7x7, iterations=3)
erode = cv.dilate(dilate, kernel_ellipse_3x3, iterations=4)

binary2 = cv.bitwise_not(erode)

img, count = color_pools(img, erode, binary2)
res_title = "Cells Count = " + str(count)

# plot
plt.subplot(251), plt.imshow(gray, cmap='gray')
plt.title('Input'), plt.xticks([]), plt.yticks([])
plt.subplot(252), plt.imshow(blur, cmap='gray')
plt.title('Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(253), plt.imshow(binary, cmap='gray')
plt.title('Binary'), plt.xticks([]), plt.yticks([])
plt.subplot(254), plt.imshow(median_filter, cmap='gray')
plt.title('Median Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(255), plt.imshow(closing, cmap='gray')
plt.title('Closing'), plt.xticks([]), plt.yticks([])

plt.subplot(256), plt.imshow(opening, cmap='gray')
plt.title('Opening'), plt.xticks([]), plt.yticks([])
plt.subplot(257), plt.imshow(dilate, cmap='gray')
plt.title('Dilate'), plt.xticks([]), plt.yticks([])
plt.subplot(258), plt.imshow(erode, cmap='gray')
plt.title('Erode'), plt.xticks([]), plt.yticks([])
plt.subplot(259), plt.imshow(binary2, cmap='gray')
plt.title('Inverted'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 5, 10), plt.imshow(img, cmap='gray')
plt.title(res_title), plt.xticks([]), plt.yticks([])

plt.show()
