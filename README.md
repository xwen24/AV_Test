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
git clone https://github.com/xwen24/AV_Test.git
cd AV_Test

Step 2: Python Environment Setup (Recommended using conda)
conda create -n nature_env python=3.9.16 -y
conda activate nature_env
pip install -r requirements.txt

Step 3: R Environment Setup
Open your R console and run:
install.packages(c("tidyverse", "lme4", "broom.mixed", "spdep"))

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

CRITICAL: The scripts in `src/` must be executed in numerical order according to their filename prefixes. 

Please run the 10 scripts strictly in the following numerical order:

The main functionality of each script is summarized below.

| Step | Script | Language | Functionality |
|---:|---|---|---|
| 1 | `Data_Check_1.py` | Python | Checks and standardizes raw GeoJSON files. It normalizes longitude coordinates and overwrites the corrected GeoJSON files. |
| 2 | `Total_Dist_2.py` | Python | Processes AV service-area boundaries, organizes raw GeoJSON files by country, merges service areas at the city level, intersects them with global built-up urban-area data, and generates city-, corporation-, and built-up-area maps. |
| 3 | `City_Dist_3.py` | Python | Produces global distribution maps of AV-served cities. It geocodes cities, calculates GDP per capita, extracts climate types, counts AV service providers by city, and generates global and regional visualization outputs. |
| 4 | `Area_Calc_4.py` | Python | Calculates the AV-served built-up area and estimated covered population for each city by intersecting AV service areas with GHS-UCDB urban-center data. It also generates summary plots of absolute and percentage coverage. |
| 5 | `Road_Calc_5.py` | Python | Downloads or loads road-network data from OpenStreetMap, calculates road mileage, intersection counts, traffic signals, stop signs, and roundabouts for AV-served and AV-unserved areas, and generates road-network summary plots. |
| 6 | `Global_Comp_6.py` | Python | Performs global comparison between AV-served and AV-unserved cities. It computes socioeconomic, environmental, terrain, road-network, and POI-based indicators, and generates comparative statistical plots. |
| 7 | `Bias_Analysis_7.py` | Python | Analyzes deployment bias and compatibility gaps between AV-served and AV-unserved cities. It computes correlation, VIF, radar charts, TOPSIS/VIKOR-based deployment compatibility gaps, regional inequality plots, and potential deployment city rankings. |
| 8 | `EB_Comp_8.py` | Python | Conducts entropy-balancing analysis to compare AV-served and AV-unserved cities within countries. It estimates weighted treatment differences, checks covariate balance, runs robustness checks, and generates balance and regression visualizations. |
| 9 | `Intra_City_9.R` | R | Runs intra-city generalized linear mixed models. It merges grid-level indicators, standardizes variables within cities, fits main and robustness GLMMs, calculates VIF, Moran’s I, diagnostics, and prediction outputs for ROC analysis. |
| 10 | `Intra_City_10.py` | Python | Generates intra-city grid-level indicators and final figures. It creates city grids, calculates night-time light, slope, road density, road entropy, sinuosity, intersection complexity, transit-station distance, and POI richness, and produces diagnostic and visualization outputs. |

### Reproduction Instructions
To fully reproduce the findings, figures, and tables presented in the manuscript:
1. Download the full public dataset from https://drive.google.com/file/d/1YwxumgjKuW3JZwpIrZQ0mWAHTRJeYV63/view?usp=drive_link.
2. Place the raw data in the Data/ directory.
3. Follow the sequential 10-step pipeline in `src/` as described above.
