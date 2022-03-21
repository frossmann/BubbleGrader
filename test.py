#%%
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pdf2image import convert_from_path
from StringUtils import tprint, get_alpha
import string
import yaml


def pdf_to_jpeg(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    jpeg_path = pdf_path[:-3] + "jpg"
    for page in pages:
        page.save(jpeg_path, "JPEG")

    return jpeg_path


class BubbleDispatcher:
    def load_jpeg(self):
        # Read the converted jpeg:
        print(f"{self.filename=}")
        im = cv.imread(self.jpeg_path)
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        # apply binary thresholding
        _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
        # Find the bounding box for each 'cluster' of bubbles
        # invert thresholding:
        thresh = 255 - thresh

        return im, thresh

    def find_bboxes(self):
        # apply morphological close with square kernel
        kernel_size = 50
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        morph = cv.morphologyEx(self.thresh, cv.MORPH_CLOSE, kernel)

        #  find the external contours:
        contours = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        bboxes = np.array([cv.boundingRect(contour) for contour in contours])
        bbox_area = bboxes[:, 2] * bboxes[:, 3]
        #    n_boxes = input(f"Input number of bounding boxes to search for:")

        idx = np.argsort(bbox_area)[::-1][: int(self.n_boxes)]
        candidate_bboxes = bboxes[idx, :]
        return candidate_bboxes

    def set_template(self):
        cv.imshow("Template", self.im)
        cv.waitKey(1)

        n_boxes = ""
        while not n_boxes.isnumeric():
            n_boxes = input(
                "How many bounding boxes should BubbleDispatcher search for? "
            )
        self.n_boxes = n_boxes

        candidate_bboxes = self.find_bboxes()

        im_copy = self.im.copy()
        for bb in candidate_bboxes:
            pad = 10
            x, y, w, h = bb
            cv.rectangle(
                im_copy, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 0, 255), 4
            )
        cv.imshow("Bounding box set-up", im_copy)
        cv.waitKey(1)
        cv.destroyAllWindows()

        to_keep = []
        n_kept = 0
        for iter, bb in enumerate(candidate_bboxes):
            pad = 10
            x, y, w, h = bb
            im_copy = self.im.copy()
            cv.rectangle(
                im_copy, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 0, 255), 4
            )
            cv.imshow("Bounding box set-up", im_copy)
            cv.waitKey(1)
            cv.destroyAllWindows()

            bbox_params = {}
            param_keys = ["direction", "n_cols", "n_rows", "format"]
            while (res := str(input(f"Keep bounding box? [y/n] ").lower())) not in {
                "y",
                "n",
            }:
                pass

            if res == "y":
                direction = input("Row-major or column-major? [Enter r/c]")
                n_cols = input("How many columns?    [Enter integer")
                n_rows = input("How many rows? [Enter integer] ")
                format = input("Alphanumeric or numeric?    [Enter a/n]")
                bbox_params[n_kept] = dict(
                    zip(param_keys, [direction, n_cols, n_rows, format])
                )
                to_keep.append(iter)
                n_kept += 1
            else:
                pass
        self.bbox_params = bbox_params
        self.bboxes = candidate_bboxes[to_keep, :]
        self.n_boxes_kept = n_kept

    def __init__(self, im_filename):
        print("Initializing BubbleDispatcher...")
        # im_filename = input("Input filepath to template bubble sheet:  ")

        self.filename = im_filename
        jpeg_path = pdf_to_jpeg(im_filename)
        self.jpeg_path = jpeg_path
        self.im, self.thresh = self.load_jpeg()


class BubbleGrader:
    pass


# filename = "/Users/francis/Dropbox/bad_bubbles.pdf"
# /Users/francis/Dropbox/eg_scan.pdf
if __name__ == "__main__":
    template = BubbleDispatcher(im_filename="/Users/francis/Dropbox/eg_scan.pdf")
    template.set_template()

    new = BubbleDispatcher(
        "/Users/francis/Desktop/eosc110v01/data/20220318131548963.pdf"
    )
    # for student in students...
    """Load each student's PDF scan and find the bounding
    boxes given template: 
    - n_boxes_kept
    - bboxes

    
    For each student scan find n_boxes_kept boundary boxes
    using template.bboxes as a target to make the decision about
    what bboxes to keep / discard."""
    for target_bbox in template.bboxes:
        """Then crop each binary image inside of the target_bbox
        and grade with:
        - solutions
        - alpha
        - num_questions
        - num_options"""
        pass


#%%
print("")
#%%


def grade_bbox(alpha, solutions, im, thresh, bb, num_options, num_questions):
    result = im.copy()
    pad = 10
    x, y, w, h = bb
    cv.rectangle(result, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 0, 255), 4)
    im_crop = im.copy()[y - pad : y + h + pad, x - pad : x + pad + w]
    thresh_crop = thresh.copy()[y - pad : y + h + pad, x - pad : x + pad + w]

    # # show the cropped image:
    # fig, ax = plt.subplots(2,1, figsize=(10,10))
    # ax[0].imshow(im_crop)
    # ax[1].imshow(thresh_crop)
    image_copy = im_crop.copy()

    contours = cv.findContours(
        image=thresh_crop, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE
    )
    contours = contours[0] if len(contours) == 2 else contours[1]

    # # visualize the results
    # cv.imshow("contours :: CHAIN_APPROX_NONE", image_copy)
    # cv.waitKey(0)
    # # cv.imwrite("contours.jpg", image_copy)
    # cv.destroyAllWindows()
    bubble_contours = []
    areas = []
    for contour in contours:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / float(h)

        area = cv.contourArea(contour)
        # print(f'{w=}, {h=}')
        # print(f'(w x h)= {w*h}')
        # print(f'{area=}')
        areas.append(area)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if aspect_ratio >= 3 / 4 and aspect_ratio <= 5 / 4 and area > 1000:
            # print('Contour passed:')
            # tprint(f'{w=}, {h=}')
            # tprint(f'(w x h)= {w*h}')
            # tprint(f'{area=}')
            bubble_contours.append(contour)

    # sort the contours from top to bottom:
    sorted_contours = sorted(
        bubble_contours, key=lambda contour: cv.boundingRect(contour)[1], reverse=False
    )

    ###
    num_bubbles = num_questions * num_options

    if not len(sorted_contours) == num_bubbles:
        Warning(
            f"Failed autograde. {len(sorted_contours)} bubbles found, expected{num_bubbles=}. "
        )
        return False

    # sort the question contours top-to-bottom
    responses = np.zeros(num_questions)
    # I'm using num_bubbles as the stopping point so I don't get dict lookup errors
    # if I find too many contours
    for (q, i) in enumerate(np.arange(0, num_bubbles, 5)):
        if q > num_questions:
            continue
        # sort the contours for the current question from
        # left to right, then initialize the index of the bubbled answer
        # default sort for sort_contours function is 'left-to-right'. [0] returns the contours
        sorted_question = sorted(
            sorted_contours[i : i + 5], key=lambda contour: cv.boundingRect(contour)[0]
        )
        sorted_question = sorted_question
        # cnts = contours.sort_contours(sorted_contours[i : i + 5])[0]

        pixel_counts = np.zeros(num_options)
        bubble_sizes = np.zeros(num_options)
        # loop over the sorted contours
        for iter, question_contour in enumerate(sorted_question):
            # construct a mask that reveals only the current "bubble" for the question
            # Return a new array, mask, of given shape and type, filled with zeros
            mask = np.zeros(thresh_crop.shape, dtype="uint8")

            # drawContours args: source image [mask], contours passed as a list [[c]],
            # index of contours (to draw all contours pass -1), color, thickness (-1 is filled in I think)
            # given mask, draw all contours on the mask of zeros, filling those pixels in with 255
            cv.drawContours(mask, [question_contour], -1, 255, -1)

            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the bubble area
            masked = cv.bitwise_and(thresh_crop, thresh_crop, mask=mask)

            # cv.imshow('sequentially masked bubble',masked)
            # cv.waitKey(0)

            # countNonZero returns the number of non-zero pixels in the bubble area
            total = cv.countNonZero(masked)

            bubble_sizes[iter] = np.size(np.where(mask == 255)[0])  # area of bubble
            pixel_counts[iter] = total  # number of 'filled in' pixels
        # Uncomment to look at masks

        bubble_fill = pixel_counts / bubble_sizes
        max_filled = np.argmax(bubble_fill)
        answer = None

        if (
            np.allclose(np.max(bubble_fill), np.min(bubble_fill), rtol=0.5)
            and np.max(bubble_fill) < 0.8
        ):
            tprint("No answer selected")

        else:
            tprint(f"Answer given: {alpha[max_filled]}")
            answer = max_filled

            if alpha[max_filled] == solutions[q]:
                tprint("Correct!")
                color = [0, 255, 0]

            else:
                tprint("Incorrect.")
                color = [0, 0, 255]
        # Unfilled bubble ~ 33% filled.
        # Filled bubble ~ 95% filled

        # if answer:
        #     image_copy = im_crop.copy()
        #     # draw contours
        #     cv.drawContours(
        #         image=image_copy,
        #         contours=sorted_question[answer],
        #         contourIdx=-1,
        #         color=color,
        #         thickness=2,
        #         lineType=cv.LINE_AA,
        #     )
        #     cv.imshow("Testing", image_copy)
        #     cv.waitKey(0)


# Load in the PDF and convert to JPEG format so that
# OpenCV can read in the file:
filename = "/Users/francis/Dropbox/eg_scan.pdf"

alpha = dict(zip(np.arange(0, 6), ["a", "b", "c", "d", "e"]))

solutions = [
    "b",
    "c",
    "c",
    "e",
    "c",
    "d",
    "a",
    "b",
    "d",
    "c",
    "e",
    "e",
    "b",
    "c",
    "e",
    "e",
    "c",
    "b",
]

# a bad scan to try and break the code:
filename = "/Users/francis/Dropbox/bad_bubbles.pdf"

num_questions = 30
num_options = 5

# jpeg_path = pdf_to_jpeg(filename)

# im, thresh = load_jpeg(jpeg_path)

# bubble_bboxs = find_bounding_boxes(im, thresh)

alpha = get_alpha(num_options=num_options)

grade_bbox(alpha, solutions, im, thresh, bubble_bboxs[0], num_options, num_questions)

#%%# %%
