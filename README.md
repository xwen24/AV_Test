This repository contains the source code and a demo dataset for the manuscript: **"Mobility inequality in global cities via dual-layer concentration of autonomous vehicle services"**.

---

## 1. System Requirements

### Operating Systems
* **Tested OS:** This software has been tested on the following systems:
    * macOS: <Tahoe 26.5>

### Software Dependencies & Version Numbers
The project requires both Python and R environments. The specific versions tested and required are:

#### Python Environment
* **Python Version:** 3.9.16
* **Required Packages:**
    *All python dependencies are also listed in requirements.txt*

#### R Environment
* **R Version:** 4.5.2
* **Required Packages:**
    * tidyverse (>= 2.0.0)
    * lme4 (>= 1.1-38)
    * broom.mixed (>=0.2.9.6)
    * spdep (>=1.4-1)

### Non-Standard Hardware Requirements
* None

---

## 2. Installation Guide

### Instructions

Step 1: Clone the Repository
$git clone https://github.com/<your-username>/<your-repo-name>.git$ cd <your-repo-name>

Step 2: Python Environment Setup (Recommended using conda)
$ conda create -n nature_env python=3.9.16 -y$ conda activate nature_env
$ pip install -r requirements.txt

Step 3: R Environment Setup
Open your R console and run:
install.packages(c("<package_1>", "<package_2>"))

### Typical Install Time
* **Estimated Time:** Approximately 10 minutes on a "normal" desktop computer (dependent on internet speed for downloading packages).

---

## 3. Demo

We provide a small real dataset in the /demo_data folder to test the pipeline.

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
1. Download the full public dataset from https://drive.google.com/file/d/1YwxumgjKuW3JZwpIrZQ0mWAHTRJeYV63/view?usp=drive_link.
2. Place the raw data in the /data directory.
3. Execute the master reproduction script: bash run_reproduction_pipeline.sh
