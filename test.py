#%%
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pdf2image import convert_from_path
from StringUtils import tprint, get_alpha, timestamp
import string
import yaml
import os


def pdf_to_jpeg(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    jpeg_path = pdf_path[:-3] + "jpg"
    for page in pages:
        page.save(jpeg_path, "JPEG")

    return jpeg_path


class BubbleDispatcher:
    def load_jpeg(self):
        # Read the converted jpeg:
        im = cv.imread(self.jpeg_path)
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        # apply binary thresholding
        _, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
        # Find the bounding box for each 'cluster' of bubbles
        # invert thresholding:
        thresh = 255 - thresh

        return im, thresh

    def find_n_bboxes(self):
        # apply morphological close with square kernel
        bboxes = self.find_bboxes()
        bbox_area = bboxes[:, 2] * bboxes[:, 3]
        #    n_boxes = input(f"Input number of bounding boxes to search for:")
        idx = np.argsort(bbox_area)[::-1][: int(self.n_boxes)]
        # candidate_bboxes = bboxes[idx, :]
        self.bboxes = bboxes[idx, :]

    def find_bboxes(self):
        kernel_size = 50
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        morph = cv.morphologyEx(self.thresh, cv.MORPH_CLOSE, kernel)

        #  find the external contours:
        contours = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        bboxes = np.array([cv.boundingRect(contour) for contour in contours])
        return bboxes

    def set_template(self):
        cv.imshow("Template", self.im)
        cv.waitKey(1)

        n_boxes = ""
        while not n_boxes.isnumeric():
            n_boxes = input(
                "How many bounding boxes should BubbleDispatcher search for? "
            )
        self.n_boxes = n_boxes

        self.find_n_bboxes()

        im_copy = self.im.copy()
        for bb in self.bboxes:
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
        for iter, bb in enumerate(self.bboxes):
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
                n_cols = int(input("How many columns?    [Enter integer"))
                n_rows = int(input("How many rows? [Enter integer] "))
                format = input("Alphanumeric or numeric?    [Enter a/n]")
                bbox_params[n_kept] = dict(
                    zip(param_keys, [direction, n_cols, n_rows, format])
                )
                to_keep.append(iter)
                n_kept += 1
            else:
                pass
        self.bbox_params = bbox_params
        self.bboxes = self.bboxes[to_keep, :]
        self.n_boxes_kept = n_kept

    def find_bbox_from_target(self, targets):
        if len(targets) == 1:
            targets = [targets]

        output_bboxes = []
        for target in targets:
            bboxes = self.find_bboxes()
            rmse = [np.sqrt(np.mean(bbox - target) ** 2) for bbox in bboxes]
            closest_bbox = np.argmin(rmse)
            output_bboxes.append(bboxes[closest_bbox])
        self.bboxes = np.array(output_bboxes)

    def __init__(self, jpeg_path):
        # im_filename = input("Input filepath to template bubble sheet:  ")
        self.jpeg_path = jpeg_path
        self.im, self.thresh = self.load_jpeg()
        self.n_boxes = None


# filename = "/Users/francis/Dropbox/bad_bubbles.pdf"
# /Users/francis/Dropbox/eg_scan.pdf
if __name__ == "__main__":
    fullfile = (
        "/Users/francis/Desktop/eosc110v01/eosc110v01_midterm1_individual_with_key.pdf"
    )
    folder, file = os.path.split(os.path.abspath(fullfile))
    # pages = convert_from_path(fullfile, dpi=300)

    today = timestamp()
    jpeg_folder = folder + "/" + today + "_pdf2jpg/"
    # os.makedirs(jpeg_folder, exist_ok=True)

    print("Reading PDF and converting to JPEG...")
    print(f"Converted pages will be saved to {jpeg_folder}")
    for page_num, page in enumerate(pages[:2]):
        jpeg_path = jpeg_folder + file[:-4] + "_page" + str(page_num + 1) + ".jpg"
        page.save(jpeg_path, "JPEG")

        if page_num == 0:
            template = BubbleDispatcher(jpeg_path)
            template.set_template()

        else:
            bubblesheet = BubbleDispatcher(jpeg_path)
            bubblesheet.find_bbox_from_target(template.bboxes)
            """For each student scan find n_boxes_kept boundary boxes
            using template.bboxes as a target to make the decision about
            what bboxes to keep / discard."""

            for target_bbox in template.bboxes:
                pass
                # Select the correct bounding box in the bubblesheet
                # based of of the target bounding box from the template (key).


#%%
print("")
#%%


def find_bubble_contours(contours):
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
    return bubble_contours


def read_bubbles_in_bbox(im, thresh, bb, n_rows, n_cols, direction, format):

    num_bubbles = n_rows * n_cols
    print(f"{num_bubbles=}")
    num_questions = n_rows if direction == "r" else n_cols
    print(f"{num_questions=}")
    num_options = n_cols if direction == "r" else n_rows
    print(f"{num_options=}")

    if format == "a":
        alpha = get_alpha(num_options)
    else:
        alpha = np.arange(num_options)
    print(f"{alpha=}")

    # Copy the image for plotting later:
    im_copy = im.copy()
    # Draw the bounding box rectange (plus some padding)
    # and crop both the image and the thresholded image.
    pad = 10
    x, y, w, h = bb
    cv.rectangle(
        im_copy, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 0, 255), 4
    )
    im_crop = im_copy[y - pad : y + h + pad, x - pad : x + pad + w]
    thresh_crop = thresh.copy()[y - pad : y + h + pad, x - pad : x + pad + w]

    # # show the cropped image:
    # fig, ax = plt.subplots(2,1, figsize=(10,10))
    # ax[0].imshow(im_crop)
    # ax[1].imshow(thresh_crop)

    # Find the external contours in the cropped and thresholded
    # image:
    contours = cv.findContours(
        image=thresh_crop, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE
    )
    contours = contours[0] if len(contours) == 2 else contours[1]

    bubble_contours = find_bubble_contours(contours)
    print(f"{bubble_contours=}")
    if not len(bubble_contours) == num_bubbles:
        Warning(
            f"Failed autograde. {len(bubble_contours)} bubbles found, expected{num_bubbles=}. "
        )
        return False

    # If row-major:
    if direction == "r":
        # Sort the bubble contours from top to bottom of the page:
        sorted_contours = sorted(
            bubble_contours,
            key=lambda contour: cv.boundingRect(contour)[1],
            reverse=False,
        )
    # Else if column-major:
    elif direction == "c":
        # Sort the bubble contours from left to right of the page:
        sorted_contours = sorted(
            bubble_contours,
            key=lambda contour: cv.boundingRect(contour)[0],
            reverse=False,
        )

    print(f"{sorted_contours=}")
    # Initiate an empty array to record answers:
    responses = []
    # Loop through the sorted bubble contours from range(num_question)
    # in steps of num_options:
    for q, i in enumerate(np.arange(0, num_bubbles, num_options)):
        print(f"{q=}")
        print(f"{i=}")
        if direction == "r":
            # sort the contours for the current question from left to right
            sorted_question = sorted(
                sorted_contours[i : i + num_options],
                key=lambda contour: cv.boundingRect(contour)[0],
            )
        if direction == "c":
            # sort the contours for the current question from top to bottom
            sorted_question = sorted(
                sorted_contours[i : i + num_options],
                key=lambda contour: cv.boundingRect(contour)[1],
            )

        # Initialize empty arrays to hold # pixels filled per
        # bubble and number of pixels total per bubble:
        pixel_counts = np.zeros(num_options)
        bubble_sizes = np.zeros(num_options)
        # For each question, loop through bubbles:
        for iter, question_contour in enumerate(sorted_question):
            print(f"{iter=}")
            print(f"{question_contour=}")
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

            # countNonZero returns the number of non-zero pixels in the bubble area
            total = cv.countNonZero(masked)

            bubble_sizes[iter] = np.size(np.where(mask == 255)[0])  # area of bubble
            pixel_counts[iter] = total  # number of 'filled in' pixels

        # Calculate relative proportion of bubble filled in:
        bubble_fill = pixel_counts / bubble_sizes
        # Find the most filled in bubble:
        max_filled = np.argmax(bubble_fill)

        # Test to see if no bubble was selected:
        if (
            np.allclose(np.max(bubble_fill), np.min(bubble_fill), rtol=0.5)
            and np.max(bubble_fill) < 0.8
        ):
            tprint("No answer selected")
            answer = np.nan
        # Otherwise, choose the most-filled in option.
        # FIXME: add capability to read > 1 filled in option, flag, and allow manual marking.
        else:
            print(alpha)
            print(max_filled)
            answer = alpha[max_filled]

        responses.append(answer)
    return responses

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


# # Load in the PDF and convert to JPEG format so that
# # OpenCV can read in the file:
# filename = "/Users/francis/Dropbox/eg_scan.pdf"

# alpha = dict(zip(np.arange(0, 6), ["a", "b", "c", "d", "e"]))

# solutions = [
#     "b",
#     "c",
#     "c",
#     "e",
#     "c",
#     "d",
#     "a",
#     "b",
#     "d",
#     "c",
#     "e",
#     "e",
#     "b",
#     "c",
#     "e",
#     "e",
#     "c",
#     "b",
# ]


responses = read_bubbles_in_bbox(
    bubblesheet.im, bubblesheet.thresh, bubblesheet.bboxes[0], **template.bbox_params[0]
)

#%%
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
