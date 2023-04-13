# events.py
#%%
import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from pdf2image import convert_from_path
import scipy.signal as signal
import cv2 as cv
import string


def show(im):
    """
    Plotter function with params:
     - figsize: 8.5x11''
     - colorbar
    """
    _, ax = plt.subplots(figsize=(8.5, 11))
    im = ax.imshow(im, cmap="gray_r")
    plt.colorbar(mappable=im, ax=ax, shrink=0.5)
    plt.show()
    return ax


def load_image():
    pdf_path = "/Users/francis/Desktop/midterm2/individual_single.pdf"
    pages = convert_from_path(pdf_path, dpi=300)
    return pages


def threshold(im):
    im = np.array(im)
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    thresh = 255 - thresh
    return thresh


def print_sep():
    print("\n" + "*" * 20)


class BubbleBox:
    def __init__(self, im):
        self.im = im
        self.label = input(f"\nInput label for bounding box: ")
        self.extent = self.select_bounding_box()
        self.crop = self.crop_image()
        self.shape = np.shape(self.crop)
        self.xedges, self.yedges = self.partition_bubbles()
        self.nx = len(self.xedges) - 1
        self.ny = len(self.yedges) - 1
        self.axis = self.set_axis()
        self.is_alpha = self.set_format()
        self.merge_results = self.set_merge()
        self.field_labels = self.set_field_labels()
        # self.show_bins()

    def show(self, img, label=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        if label:
            ax.set_title(label)
        plt.show()

    def select_bounding_box(self):
        def onselect(eclick, erelease):
            pass

        _, ax = plt.subplots(figsize=(8.5, 11))
        ny, nx = np.shape(self.im)
        ax.imshow(self.im, origin="upper", cmap="gray_r")
        ax.set_title(f"Select Region for : '{self.label}'\nPress Q to continue.")

        props = dict(facecolor="blue", alpha=0.2)
        RS = RectangleSelector(ax, onselect, interactive=True, props=props)
        plt.show()

        extent = [int(pt) for pt in RS.extents]
        return extent

    def crop_image(self):
        x1, x2, y1, y2 = self.extent
        i1 = int(x1)
        i2 = int(x2) + 1
        i3 = int(y1)
        i4 = int(y2) + 1
        crop = self.im.copy()[i3:i4, i1:i2]
        return crop

    def savgol_filter(self, x, window_size, poly_order):
        xhat = signal.savgol_filter(x, window_size, poly_order)
        return xhat / xhat.max()

    def partition_bubbles(self):
        """Rule based method for delineating inter-bubble space in a
        regular grid by isolating periodic whitespace"""

        xsums = 1 - np.sum(self.crop, axis=0) / np.sum(self.crop, axis=0).max()
        ysums = 1 - np.sum(self.crop, axis=1) / np.sum(self.crop, axis=1).max()

        window_size = 50
        poly_order = 5
        xhat = self.savgol_filter(xsums, window_size, poly_order)  #
        yhat = self.savgol_filter(ysums, window_size, poly_order)

        fx, Pxx = signal.welch(xhat)
        fy, Pyy = signal.welch(yhat)
        ix = np.argmax(Pxx)
        iy = np.argmax(Pyy)

        x_window_size = int(1.2 / fx[ix])
        y_window_size = int(1.2 / fy[iy])
        xhat = self.savgol_filter(xsums, x_window_size, poly_order)  #
        yhat = self.savgol_filter(ysums, y_window_size, poly_order)

        xpeaks, _ = signal.find_peaks(xhat, height=0.5, distance=0.8 / fx[ix])
        ypeaks, _ = signal.find_peaks(yhat, height=0.5, distance=0.8 / fy[iy])

        xedges = [0, *xpeaks, self.shape[1]]
        # print(f"{xedges=}, {np.diff(xeådges)=}")
        yedges = [0, *ypeaks, self.shape[0]]
        # print(f"{yedges=}, , {np.diff(yedges)=}")

        # squash and under-sized bins at the edges:
        xdiff = np.diff(xedges)
        ydiff = np.diff(yedges)

        x_min_width = np.median(xdiff) * 0.7
        y_min_width = np.median(ydiff) * 0.7
        if xdiff[0] < x_min_width:
            xedges.pop(0)
        if xdiff[-1] < x_min_width:
            xedges.pop(-1)
        if ydiff[0] < y_min_width:
            yedges.pop(0)
        if ydiff[-1] < y_min_width:
            yedges.pop(-1)

        # fig, ax = plt.subplots(3)
        # ax[0].plot(xhat)
        # ax[0].set_title("Filtered x-sums")
        # yl0 = ax[0].get_ylim()

        # ax[1].plot(yhat)
        # ax[1].set_title("Filtered y-sums")
        # yl1 = ax[1].get_ylim()

        # ax[2].plot(fx, Pxx, label="Pxx")
        # ax[2].plot(fy, Pyy, label="Pyy")
        # plt.legend()

        # plt.suptitle(
        #     f"{1/fx[ix] = }  {1/fy[iy] = }\n{x_window_size=}, {y_window_size=}"
        # )

        # for i in xedges:
        #     ax[0].vlines(i, *yl0, color="r", alpha=0.6)
        # for i in yedges:
        #     ax[1].vlines(i, *yl1, color="r", alpha=0.6)

        # plt.show()

        return xedges, yedges

    def show_bins(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.crop)
        for i in self.yedges:
            ax.hlines(
                i, xmin=self.xedges[0], xmax=self.xedges[-1], color="w", alpha=0.6
            )
        for i in self.xedges:
            ax.vlines(
                i, ymin=self.yedges[0], ymax=self.yedges[-1], color="w", alpha=0.6
            )
        ax.set_title(f"{self.label}\nAutomatically extracted bins ")
        plt.show()

    def set_axis(self):
        AXIS_SET = False
        while not AXIS_SET:
            user_input = input("Read along rows or columns? [r/c]: ")
            if user_input.lower() == "r":
                return 0
            elif user_input.lower() == "c":
                return 1
            else:
                print("Invalid input. Valid inputs are [r/c].")

    def set_format(self):
        FORMAT_SET = False
        while not FORMAT_SET:
            user_input = input("Alphabetical or numeric data type? [a/n]: ")
            if user_input.lower() == "a":
                return 1
            elif user_input.lower() == "n":
                return 0
            else:
                print("Invalid input. Valid inputs are [a/n].")

    def set_merge(self):
        MERGE_SET = False
        while not MERGE_SET:
            user_input = input("Continuous or discrete data type? [c/d]: ")
            if user_input.lower() == "c":
                return 1
            elif user_input.lower() == "d":
                return 0
            else:
                print("Invalid input. Valid inputs are [c/d].")

    def set_field_labels(self):
        if self.merge_results:
            return (self.label,)
        else:
            if self.axis == 0:
                return [self.label + f"{ii + 1}" for ii in range(self.ny)]
            if self.axis == 1:
                return [self.label + f"{ii + 1}" for ii in range(self.nx)]


def get_alpha(num_options):
    alphabet = list(string.ascii_lowercase)
    alpha = dict(zip(np.arange(num_options), alphabet[:num_options]))
    return alpha


def bin_and_count_bbox(bbox):
    H = np.zeros((bbox.ny, bbox.nx))

    xe, ye = bbox.xedges, bbox.yedges
    for ix in range(len(xe) - 1):
        for iy in range(len(ye) - 1):
            i1 = xe[ix]
            i2 = xe[ix + 1] + 1
            i3 = ye[iy]
            i4 = ye[iy + 1] + 1
            H[iy, ix] = np.sum(bbox.crop[i3:i4, i1:i2])

    return H


def bin_and_count_image(image, bbox):
    H = np.zeros((bbox.ny, bbox.nx))

    xe, ye = bbox.xedges, bbox.yedges
    for ix in range(len(xe) - 1):
        for iy in range(len(ye) - 1):
            i1 = xe[ix]
            i2 = xe[ix + 1] + 1
            i3 = ye[iy]
            i4 = ye[iy + 1] + 1
            H[iy, ix] = np.sum(image[i3:i4, i1:i2])
    return H


def parse_filled_bubbles(bbox, H):
    background_value = np.median(H.ravel())

    results = []
    # count over rows
    if bbox.axis == 0:
        for ii in range(bbox.ny):
            row = H[ii, :]
            filled = [
                idx for idx, val in enumerate(row) if val > 1.5 * np.percentile(row, 66)
            ]
            if len(filled) > 0:
                results.append(filled)
            else:
                results.append(
                    [
                        None,
                    ]
                )

    # count over columns
    if bbox.axis == 1:
        for ii in range(bbox.nx):
            col = H[:, ii]
            filled = [
                idx for idx, val in enumerate(col) if val > 1.5 * np.percentile(col, 66)
            ]
            if len(filled) > 0:
                results.append(filled)
            else:
                results.append(
                    [
                        None,
                    ]
                )

    # convert numeric to alpha
    if bbox.is_alpha:
        to_alpha = get_alpha(26)
        alpha_results = []

        for sublist in results:
            sub = [to_alpha[val] if val is not None else val for val in sublist]
            alpha_results.append(sub)

        results = alpha_results

    if bbox.merge_results:
        merged_results = ""
        for item in results:
            if len(item) == 1:
                if item[0] is not None:
                    merged_results += f"{item[0]}"
            elif len(item) > 1:
                compound_results = "("
                for val in item:
                    compound_results += f"{val},"
                compound_results += ")"
                merged_results += compound_results
        cleaned_results = merged_results
    else:
        cleaned_results = []
        for item in results:
            if len(item) == 1:
                cleaned_results.append(f"{item[0]}")
            if len(item) > 1:
                compound_item = "("
                for val in item:
                    compound_item += f"{val},"
                compound_item += ")"
                cleaned_results.append(compound_item)

    return cleaned_results


def select_regions(image):

    REGIONS_FINALIZED = False

    regions = dict()

    num_regions = 0
    while not REGIONS_FINALIZED:
        # initialize a new BubbleBox instance and wait for user-input
        num_regions += 1
        print_sep()
        print(f"Bounding box {num_regions}:")
        bbox = BubbleBox(image)

        regions[bbox.label] = bbox

        if not click.confirm("\nDraw another region?", default=True):
            REGIONS_FINALIZED = True

    return regions


def plot_annotated(image, regions):
    # PLOT THE ANNOTATED DOCUMENT
    fig, ax = plt.subplots(figsize=(10, 10))
    xl = [0, bbox.im.shape[1]]
    yl = [0, bbox.im.shape[0]]
    ax.imshow(bbox.im, extent=[*xl, *yl], cmap="gray_r")
    ax.set_xlim(xl)
    ax.set_ylim(yl)

    # with gridlines

    for key, bbox in regions.items():

        H = bin_and_count_bbox(bbox)
        result = parse_filled_bubbles(bbox, H)

        im = ax.imshow(
            H,
            extent=[
                bbox.extent[0] + bbox.xedges[0] - 1,
                bbox.extent[0] + bbox.xedges[-1] - 1,
                yl[1] - bbox.extent[3] + bbox.yedges[0] + 1,
                yl[1] - bbox.extent[3] + bbox.yedges[-1] + 1,
            ],
            alpha=0.5,
            zorder=11,
        )

    plt.show()


if __name__ == "__main__":
    im = threshold(load_image()[0])
    regions = select_regions()


# %%
