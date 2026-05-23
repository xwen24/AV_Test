This repository contains the source code and a dataset link for the manuscript: **"Mobility inequality in global cities via dual-layer concentration of autonomous vehicle services"**.

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

### Data Download
The dataset required to demo this software is publicly available and can be downloaded from:
* **Download Link:** https://drive.google.com/file/d/1YwxumgjKuW3JZwpIrZQ0mWAHTRJeYV63/view?usp=drive_link
*(Note: Please download the data, unzip it, and place it into a folder named `Data/` in the root directory of this project).*

### Instructions to Run on Demo Data
1. Run the Python processing step:
   python src/Total_Dist_2.py

### Expected Output
Upon successful execution, the software will automatically generate the following four output folders containing visual results:
* `Built_Up/`: Contains generated `.png` files.
* `Built_Up_Intersect/`: Contains generated `.png` files.
* `City/`: Contains generated `.png` files.
* `Corp/`: Contains generated `.png` files.

### Expected Run Time for Demo
* **Estimated Time:** Approximately 10 minutes on a "normal" desktop computer.

---

## 4. Instructions for Use

### How to Run the Software on Your Data
To apply this pipeline to your own datasets, please format your input files identically to the provided files in the `Data/` directory.

CRITICAL: The source code files inside the `src/` directory must be executed sequentially based on the prefix of their filenames (from 1 to 10). 

Please run the 10 scripts strictly in the following numerical order:

1. [Python] Step 1 - python src/Data_Check_1.py

2. [Python] Step 2 - python src/Total_Dist_2.py

3. [Python] Step 3 - python src/City_Dist_3.py

4. [Python] Step 4 - python src/Area_Calc_4.py

5. [Python] Step 5 - python src/Road_Calc_5.py

6. [Python] Step 6 - python src/Global_Comp_6.py

7. [Python] Step 7 - python src/Bias_Analysis_7.py

8. [Python] Step 8 - python src/EB_Comp_8.py

9. [R] Step 9 - Rscript src/Intra_City_9.R

10. [Python] Step 10 - python src/Intra_City_10.py

### (OPTIONAL) Reproduction Instructions
To fully reproduce the findings, figures, and tables presented in the manuscript:
1. Download the full public dataset from https://drive.google.com/file/d/1YwxumgjKuW3JZwpIrZQ0mWAHTRJeYV63/view?usp=drive_link.
2. Place the raw data in the Data/ directory.
3. Follow the sequential 10-step pipeline in `src/` as described above.
