#%%
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pdf2image import convert_from_path


def pdf_to_jpeg(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    jpeg_path = pdf_path[:-3] + "jpg"
    for page in pages:
        page.save(jpeg_path, "JPEG")

    return jpeg_path


#%% Load in the PDF and convert to JPEG format so that
# OpenCV can read in the file:
filename = "/Users/francis/Dropbox/eg_scan.pdf"

jpeg_path = pdf_to_jpeg(filename)

# Read the converted jpeg:
im = cv.imread(jpeg_path)
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
# Plot the jpeg:
# Matplotlib has RGB channels but OpenCV uses BGR channels,
# so reverse during plotting:
plt.figure()
# plt.imshow(im[:, :, ::-1])
plt.imshow(im)

# %% apply binary thresholding
ret, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)

# visualize the binary image
cv.imshow("Binary image", thresh)
cv.waitKey(0)
cv.imwrite("image_thres1.jpg", thresh)
cv.destroyAllWindows()

# %% detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv.findContours(
    image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE
)

# contour on top of copy of original:
image_copy = im.copy()
# draw contours
cv.drawContours(
    image=image_copy,
    contours=contours,
    contourIdx=-1,
    color=(0, 255, 0),
    thickness=2,
    lineType=cv.LINE_AA,
)

# visualize the results
cv.imshow("CHAIN_APPROX_NONE", image_copy)
cv.waitKey(0)
cv.imwrite("contours_none_image1.jpg", image_copy)
cv.destroyAllWindows()

# %% Find the bounding box for each 'cluster' of bubbles
# invert thresholding:
thresh = 255 - thresh

# apply morphological close with square kernel
kernel_size = 50
kernel = np.ones((kernel_size, kernel_size), np.uint8)
morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

# find the external contours:
contours = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# draw contours
result = im.copy()
for cntr in contours:
    # get bounding boxes
    pad = 10
    x, y, w, h = cv.boundingRect(cntr)
    cv.rectangle(result, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 0, 255), 4)

# save result
cv.imwrite("john_bbox.png", result)

# visualize:  result
cv.imshow("inverted threshold", thresh)
cv.imshow("morphological close", morph)
cv.imshow("bounding box approximation", result)
cv.waitKey(0)
cv.destroyAllWindows()
# %%
