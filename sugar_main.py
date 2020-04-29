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


img = cv.imread('Sugar.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (21, 21), 0)
thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)[1]

kernel = np.ones((5, 5), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
erode = cv.erode(opening, kernel)

img, count = color_pools(img, opening, erode)
res_title = "Sugar + Spoons Count = " + str(count)

# считаем ложки и сахар отдельно
img_sugar = img.copy()
img_spoons = img.copy()
sugar_count = 0
spoons_count = 0
contours, _ = cv.findContours(opening.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
for c in contours:
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.15 * peri, True)
    if len(approx) == 4:
        cv.drawContours(img_sugar, [c], -1, (255, 255, 255), 2)
        sugar_count += 1
    else:
        cv.drawContours(img_spoons, [c], -1, (255, 255, 255), 2)
        spoons_count += 1

res_sugar_title = "Sugar Count = " + str(sugar_count)
res_spoons_title = "Spoons Count = " + str(spoons_count)

# plot
plt.subplot(231), plt.imshow(gray, cmap='gray')
plt.title('Input'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(thresh, cmap='gray')
plt.title('Binary'), plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(opening, cmap='gray')
plt.title('Opening'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(img, cmap='gray')
plt.title(res_title), plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(img_sugar, cmap='gray')
plt.title(res_sugar_title), plt.xticks([]), plt.yticks([])

plt.subplot(236), plt.imshow(img_spoons, cmap='gray')
plt.title(res_spoons_title), plt.xticks([]), plt.yticks([])
plt.show()
