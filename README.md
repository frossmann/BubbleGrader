# BubbleGrader


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
  - Annotated images of each bubble sheet   
___
### What it doesn't include:
- Any warranty of any kind
- No actual grading capability - just reading. Grading can be done in -post using the spreadsheet. 
- Crash recovery

___
### How to use it: 
1. Clone this repository.
2. Create a new `conda` environment using the `environment.yaml` file: `conda env create -f environment.yaml`
3. Make some suckers write a test using bubble sheets and scan the sheets into one multi-page PDF. 
4. In Terminal, `cd` to the cloned `BubbleGrader` repository and run `python main.py <path-to-pdf>`. 
5. Follow the prompts. 

___
### How it works: 
- [ ] PDF loading with `pdf2image` and conversion to a 'stack' of binary images as `numpy` arrays
- [ ] Feature extraction using ORB (Oriented FAST and Rotated BRIEF) keypoints enabled by `opencv`
- [ ] Image registration using FLANN (Fast Library for Approximate Nearest Neighbours) kNN feature matching
- [ ] GUI for boundary box selection
- [ ] Rule-based automatic bubble edge-grid detection using local maxima in whitespace between bubbles (smoothed w/ Savitzky Golay filter with a window length optimized with the dominant spatial frequency estimated using Welch's method from the PSD)
- [ ] Rule-based is-bubble-filled-or-not logic using binned counts of pixels in each bubble grid element. 
