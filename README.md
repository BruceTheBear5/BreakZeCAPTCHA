# Precog CAPTCHA Task

Author: Krishak Aneja

## Overview
This repository contains the code and resources for solving the CAPTCHA classification and text extraction tasks as part of the Precog recruitment challenge. The project consists of two main tasks:

1. **Task 1: CAPTCHA Classification** - Classifying CAPTCHA images into one of 200 predefined classes.
2. **Task 2: CAPTCHA Text Extraction** - Extracting the embedded text from CAPTCHA images using sequence models.

## Directory Structure
```
precog-captcha-task/
├── README.md                # Overview of the project
├── requirements.txt         # List of dependencies
├── data/                    # Folder for storing dataset scripts and samples
│   ├── generate_dataset.py  # Script to generate the dataset
│   └── samples/             # Subfolder with sample images
├── src/                     # Folder for source code
│   └── models/              # Folder for model architectures
│       ├── classifier.pth       # CNN model for classification
│       ├── generator.pth       # Sequence-to-sequence model for text generation
│       └── transformer.py   
├── experiments/             # Records of all experiments
│   ├── experiment-1.ipynb       # Notebook for experiment 1
│   ├── experiment-2.ipynb       # Notebook for experiment 2
└── presentation/            # Final presentation/report resources
    ├── PaperReadingPresentation.pdf    # PDF of paper presentation
    └── Report.pdf              # Written report 
```

## Models
The final working models extracted from experiments are:
- **Task 1 (Classification):** `model_4` (CNN architecture) from `experiment-1`
- **Task 2 (Text Extraction):** `model_2` (CRNN architecture) from `experiment-2`

## Running the Project
### Installation
Ensure you have Python 3.8+ installed. Install dependencies using:
```
pip install -r requirements.txt
```
Please traverse through the Experiment notebooks. That is the intended way to explore the project.
