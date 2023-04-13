#%%
import copy
import pickle
from datetime import datetime
from pathlib import Path

import click
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from tqdm import tqdm

from events import (
    bin_and_count_image,
    estimate_filled_threshold,
    parse_filled_bubbles_with_corrections,
    print_sep,
    select_regions,
)


class FeatureExtraction:
    # def threshold(self, im):
    #     # im = np.array(im)
    #     # gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    #     _, thresh = cv.threshold(gray, 255 // 2, 255, cv.THRESH_BINARY)
    #     thresh = 255 - thresh
    #     return thresh

    def __init__(self, img):
        orb = cv.ORB_create(
            nfeatures=5000, scaleFactor=1.2, scoreType=cv.ORB_HARRIS_SCORE
        )

        self.img = copy.copy(img)
        self.gray_img = self.img
        self.kps, self.des = orb.detectAndCompute(self.gray_img, None)
        self.img_kps = cv.drawKeypoints(
            self.img, self.kps, 0, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        self.matched_pts = []


def crop(im, extent):
    x1, x2, y1, y2 = extent
    i1 = int(np.floor(x1))
    i2 = int(np.ceil(x2)) + 1
    i3 = int(np.floor(y1))
    i4 = int(np.ceil(y2)) + 1
    cropped = im.copy()[i3:i4, i1:i2]
    return cropped


def binary_threshold(im):
    im = np.array(im)
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 255 // 2, 255, cv.THRESH_BINARY)
    thresh = 255 - thresh
    return thresh


def feature_matching(features0, features1):

    LOWES_RATIO = 0.7
    MIN_MATCHES = 50
    index_params = dict(
        algorithm=6, table_number=6, key_size=10, multi_probe_level=2  # FLANN_INDEX_LSH
    )
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = []  # good matches as per Lowe's ratio test
    if features0.des is not None and len(features0.des) > 2:
        all_matches = flann.knnMatch(features0.des, features1.des, k=2)
        try:
            for m, n in all_matches:
                if m.distance < LOWES_RATIO * n.distance:
                    matches.append(m)
        except ValueError:
            pass
        if len(matches) > MIN_MATCHES:
            features0.matched_pts = np.float32(
                [features0.kps[m.queryIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
            features1.matched_pts = np.float32(
                [features1.kps[m.trainIdx].pt for m in matches]
            ).reshape(-1, 1, 2)
    return matches


def load_pdfs(pdf_path):
    print(f"\nLoading PDFs from {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=300)

    return pages


def register_images(pages):
    print("\nRegistering images with ORB keypoints.")
    ref_image = binary_threshold(np.array(pages[0]))
    h, w = ref_image.shape
    ref_features = FeatureExtraction(ref_image)

    registered_images = np.zeros([*ref_image.shape, len(pages)])
    registered_images[:, :, 0] = ref_image

    fig_params = {"figsize": (8.5, 11)}

    for ii in tqdm(range(1, len(pages))):
        this_img = binary_threshold(np.array(pages[ii]))
        features = FeatureExtraction(this_img)

        matches = feature_matching(ref_features, features)

        H, _ = cv.findHomography(
            features.matched_pts, ref_features.matched_pts, cv.RANSAC, 5.0
        )

        warped = cv.warpPerspective(this_img, H, (w, h), borderMode=cv.BORDER_CONSTANT)

        registered_images[:, :, ii] = warped
        # print(f"Finished {ii}/{len(pages) - 1}")

    return registered_images


def plot_annotated_document(
    bboxes,
    annotated_image_filename,
    image,
    xl,
    yl,
    annotation_data,
    image_destination=None,
):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.imshow(image, origin="upper", cmap="gray_r")
    ax.set_xlim(xl)
    ax.set_ylim(yl)

    for key, bbox in bboxes.items():
        H = annotation_data[key]["H"]
        H_mask = annotation_data[key]["H_mask"]
        bbox_result = annotation_data[key]["bbox_result"]
        ax.imshow(
            H_mask * H,
            extent=[
                bbox.extent[0] + bbox.xedges[0],
                bbox.extent[0] + bbox.xedges[-1],
                bbox.extent[2] + bbox.yedges[-1],
                bbox.extent[2] + bbox.yedges[0],
            ],
            alpha=0.4,
            zorder=111,
            # cmap="gray_r",
        )

        if not bbox.merge_results:
            for jj, result in enumerate(bbox_result):
                if bbox.axis == 0:
                    diff = np.diff(bbox.yedges)
                    ax.annotate(
                        result,
                        (
                            bbox.extent[0] + bbox.xedges[-1] * 1.1,
                            bbox.extent[2] + bbox.yedges[jj] + diff[jj] / 2,
                        ),
                        color="red",
                    )
                if bbox.axis == 1:
                    pass

    ax.set_title("Reference image")
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_title(annotated_image_filename)

    if image_destination:
        plt.savefig(str(image_destination), bbox_inches="tight")

    # plt.show()
    plt.close()


@click.command()
@click.option("--pdf_path", "-p", type=str, default=None)
@click.option("--recover", "-r", type=str, default=None)
def main(pdf_path, recover):

    # -------------------------------------------------------------------------------
    # STARTUP TEXT:
    # -------------------------------------------------------------------------------
    with open("logo.txt", "r") as f:
        for line in f:
            print(line.rstrip())

    # -------------------------------------------------------------------------------
    # SET PATHS FOR OUTPUT
    # -------------------------------------------------------------------------------
    session_name = input("Input a name for this session: ")
    full_session_name = (
        f"{session_name}_{datetime.now().now().strftime('%Y_%m_%d_%H%M')}"
    )

    destination_folder = Path.cwd() / "output" / full_session_name
    destination_folder.mkdir(exist_ok=True, parents=True)

    image_folder = destination_folder / "images"
    excel_folder = destination_folder / "excel"
    cache_folder = destination_folder / "cache"
    image_folder.mkdir(exist_ok=True, parents=True)
    excel_folder.mkdir(exist_ok=True, parents=True)
    cache_folder.mkdir(exist_ok=True, parents=True)

    # -------------------------------------------------------------------------------
    # LOAD PDF AND PREPROCESS
    # -------------------------------------------------------------------------------
    if pdf_path:
        pages = load_pdfs(pdf_path)

        registered_images = register_images(pages)

        # -------------------------------------------------------------------------------
        # SET BOUNDING BOXES
        # -------------------------------------------------------------------------------
        # select target bounding boxes off of the first image
        print("\nBounding box set-up:")
        bboxes = select_regions(registered_images[:, :, 0])
        print("\nFinished bounding box set-up.\n")

        # -------------------------------------------------------------------------------
        # CACHE / PICKLE REGISTERED IMAGES AND BOUNDING BOXES
        # -------------------------------------------------------------------------------
        print_sep()
        print(f"\nCacheing session data to {str(cache_folder)}")
        print(
            f"Use --recover [-r] output/{full_session_name} to restart session from cache-file.\n"
        )

        image_cache_path = cache_folder / "registered_images.npy"
        np.save(str(image_cache_path), registered_images)

        bbox_cache_path = cache_folder / "bounding_boxes.pkl"
        with open(str(bbox_cache_path), "wb") as file:
            pickle.dump(bboxes, file)

    # -------------------------------------------------------------------------------
    # LOAD REGISTERED IMAGES AND BBOXES FROM CACHE:
    # -------------------------------------------------------------------------------
    if recover:
        image_cache_path = Path.cwd() / recover / "cache" / "registered_images.npy"
        bbox_cache_path = Path.cwd() / recover / "cache" / "bounding_boxes.pkl"

        registered_images = np.load(str(image_cache_path), allow_pickle=True)

        with open(str(bbox_cache_path), "rb") as file:
            bboxes = pickle.load(file)

    # -------------------------------------------------------------------------------
    # SET UP OUTPUT VARIABLES
    # -------------------------------------------------------------------------------
    all_field_labels = [
        label for _, bbox in bboxes.items() for label in bbox.field_labels
    ]

    combined_results = pd.DataFrame(columns=all_field_labels, index=None)

    # -------------------------------------------------------------------------------
    # MAIN LOOP:
    # -------------------------------------------------------------------------------
    # loop through images and read bounding boxes:
    n_pages = np.shape(registered_images)[2]
    print_sep()
    print("Processing bubblesheets.")
    for page_no in range(n_pages):
        print(f"{page_no+1} of {n_pages}.")
        annotated_image_filename = ""
        page_results = dict.fromkeys(all_field_labels)
        image = registered_images[:, :, page_no]
        xl = [0, image.shape[1]]
        yl = [0, image.shape[0]]

        annotation_data = dict()

        for key, bbox in bboxes.items():
            image_in_bbox = crop(image, bbox.extent)
            H = bin_and_count_image(image_in_bbox, bbox)
            # bbox_result = parse_filled_bubbles(bbox, H)
            bbox_result = parse_filled_bubbles_with_corrections(image, bbox, H)
            # print(f"{key}: {bbox_result}")

            if bbox.merge_results:
                page_results[bbox.field_labels[0]] = bbox_result
                annotated_image_filename += f"{bbox_result}_"
            else:
                for idx, field_label in enumerate(bbox.field_labels):
                    page_results[field_label] = bbox_result[idx]

            H_threshold = estimate_filled_threshold(H)
            idx = np.where(H > H_threshold)
            H_mask = np.ones(H.shape)
            H_mask[idx[0], idx[1]] += 1

            annotation_data[key] = {
                "H": H,
                "H_mask": H_mask,
                "bbox_result": bbox_result,
            }

        combined_results = combined_results.append(page_results, ignore_index=True)

        # -------------------------------------------------------------------------------
        # BUILD AN ANNOTATED IMAGE
        # -------------------------------------------------------------------------------

        image_destination = image_folder / annotated_image_filename
        plot_annotated_document(
            bboxes,
            annotated_image_filename,
            image,
            xl,
            yl,
            annotation_data,
            image_destination,
        )
        # plt.close()

    # -------------------------------------------------------------------------------
    # SAVE RESULTS IN EXCEL FILE
    # -------------------------------------------------------------------------------
    spreadsheet_destination = excel_folder / "results.xlsx"
    combined_results.to_excel(str(spreadsheet_destination), index=False)
    print(f"\nResults saved to {str(destination_folder)}")

    print("\nExiting.")
    # end of program


if __name__ == "__main__":
    main()

# %%
