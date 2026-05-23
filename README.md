# Project Name: AV_Test

This repository contains the source code and a demo dataset for the manuscript: **"Mobility inequality in global cities via dual-layer concentration of autonomous vehicle services"**.

---

## Required Components Checklist
- [x] Compiled standalone software and/or source code
- [x] A small (simulated or real) dataset to demo the software/code
- [x] README file (this document)

---

## 1. System Requirements

### Operating Systems
* **Tested OS:** This software has been tested on the following systems:
    * macOS: <e.g., Ventura 13.5>
    * Linux: <e.g., Ubuntu 22.04 LTS>
    * Windows: <e.g., Windows 11>

### Software Dependencies & Version Numbers
The project requires both Python and R environments. The specific versions tested and required are:

#### Python Environment
* **Python Version:** 3.9.16
* **Required Packages:**
    * <package_name_1> (>= version)
    * <package_name_2> (>= version)
    *(Note: All python dependencies are also listed in requirements.txt)*

#### R Environment
* **R Version:** 4.5.2
* **Required Packages:**
    * <package_name_1> (>= version)
    * <package_name_2> (>= version)

### Non-Standard Hardware Requirements
* <State "None" OR describe specific hardware if required, e.g., "NVIDIA GPU with CUDA capability (minimum 8GB VRAM) is required. Standard desktop computers without a GPU can run the analysis but at a slower speed.">

---

## 2. Installation Guide

### Instructions

Step 1: Clone the Repository
$git clone https://github.com/<your-username>/<your-repo-name>.git$ cd <your-repo-name>

Step 2: Python Environment Setup (Recommended using conda)
$conda create -n nature_env python=3.9.16 -y$ conda activate nature_env
$ pip install -r requirements.txt

Step 3: R Environment Setup
Open your R console and run:
install.packages(c("<package_1>", "<package_2>"))

### Typical Install Time
* **Estimated Time:** Approximately <e.g., 5-10 minutes> on a "normal" desktop computer (dependent on internet speed for downloading packages).

---

## 3. Demo

We provide a small simulated/real dataset in the /demo_data folder to test the pipeline.

### Instructions to Run on Demo Data
1. Run the Python processing step:
   python demo_script.py --input demo_data/sample_input.csv --output demo_data/python_output.csv

2. Run the R analysis/visualization step:
   Rscript demo_script.R --input demo_data/python_output.csv --output demo_data/final_result.pdf

### Expected Output
Upon successful execution, the software will generate:
* A CSV file (demo_data/python_output.csv) containing <briefly describe expected data points>.
* A PDF plot (demo_data/final_result.pdf) displaying <briefly describe the expected figure, e.g., a scatter plot showing treatment effects>.

### Expected Run Time for Demo
* **Estimated Time:** <e.g., Less than 1 minute> on a "normal" desktop computer.

---

## 4. Instructions for Use

### How to Run the Software on Your Data
To apply this software to your own datasets, ensure your input files are formatted exactly like the provided demo files (demo_data/sample_input.csv).

Command Syntax:
# For Python step
python main_pipeline.py --input <path_to_your_data> --output <path_to_output_dir>

# For R step
Rscript main_analysis.R --input <path_to_python_output> --output <path_to_final_report>

### (OPTIONAL) Reproduction Instructions
To fully reproduce the findings, figures, and tables presented in the manuscript:
1. Download the full public dataset from <insert DOI link or database accession number>.
2. Place the raw data in the /data directory.
3. Execute the master reproduction script: bash run_reproduction_pipeline.sh
