# BubbleGrader
___

 █████╗    ██████╗    ██████╗    ██████╗ 
██╔══██╗  ██╔═══██╗  ██╔═══██╗  ██╔═══██╗
███████║  ██║ . ██║  ██║ . ██║  ██║ . ██║
██╔══██║  ██║   ██║  ██║   ██║  ██║   ██║
██║  ██║  ╚██████╔╝  ╚██████╔╝  ╚██████╔╝
╚═╝  ╚═╝   ╚═════╝    ╚═════╝    ╚═════╝ 
 ██████╗    ██████╗    ██████╗   ██████╗ 
██╔═══██╗  ██╔═══██╗  ██╔════╝  ██╔═══██╗
██║ . ██║  ██║ . ██║  ██║       ██║ . ██║
██║   ██║  ██║   ██║  ██║       ██║   ██║
╚██████╔╝  ╚██████╔╝  ╚██████╗  ╚██████╔╝
 ╚═════╝    ╚═════╝    ╚═════╝   ╚═════╝ 
 █████╗    ██████╗    ██████╗    ██████╗ 
██╔══██╗  ██╔═══██╗  ██╔═══██╗  ██╔═══██╗
███████║  ██║ . ██║  ██║ . ██║  ██║ . ██║
██╔══██║  ██║   ██║  ██║   ██║  ██║   ██║
██║  ██║  ╚██████╔╝  ╚██████╔╝  ╚██████╔╝
╚═╝  ╚═╝   ╚═════╝    ╚═════╝    ╚═════╝      
 ██████╗   ██████╗    ██████╗    ██████╗ 
██╔═══██╗  ██╔══██╗  ██╔═══██╗  ██╔═══██╗
██║ . ██║  ██████╔╝  ██║ . ██║  ██║ . ██║
██║   ██║  ██╔══██╗  ██║   ██║  ██║   ██║
╚██████╔╝  ██████╔╝  ╚██████╔╝  ╚██████╔╝
 ╚═════╝   ╚═════╝    ╚═════╝    ╚═════╝ 
 
___
### What it includes:
- Command line tool for reading OMR bubble sheets (e.g. from Remark Office OMR Software). 
- Functionality to output: 
  - An Excel spreadsheet containing the read data.  
  - Annotated images of each bubble sheet.
  - Session recovery using `pickle`.
___
### What it doesn't include:
- Any warranty of any kind
- No actual grading capability - just reading. The name is a lie. Grading can be done in -post based off the spreadsheet. 

___
### How to use it: 
1. Clone this repository.
2. Create a new `conda` environment using the `environment.yaml` file: `conda env create -f environment.yaml`
3. Make some suckers write a test using bubble sheets and scan the sheets into one multi-page PDF. 
4. In Terminal, `cd` to the cloned `BubbleGrader` repository and run `python main.py --pdf-folder <path-to-pdf>`. 
5. Follow the prompts. Results (Excel spreadsheet, annotated images, cache) saved in `/output/` folder. 

___
### How it works: 
- [ ] PDF loading with `pdf2image` and conversion to a 3D numpy array of binary-thresholded images using `opencv`
- [ ] Feature extraction using ORB (Oriented FAST and Rotated BRIEF) keypoints enabled by `opencv`
- [ ] Image registration using FLANN (Fast Library for Approximate Nearest Neighbours) kNN feature matching from `opencv`
- [ ] GUI for boundary box selection using `matplotlib`'s `RectangleSelector`
- [ ] Rule-based automatic bubble grid detection using local maxima in whitespace between bubbles (features smoothed with `scipy.signal` Savitzky-Golay filter with window length based on peak frequency from Welch's method power spectrum estimate)
- [ ] Rule-based 'is-bubble-filled-or-not' logic using binned counts of pixels in each bubble grid element (a variable grid 2D histogram). 
