# PV-Sentinel | Standalone Clinical Safety Triage & Signal Detection

A rule-based pharmacovigilance (PV) Individual Case Safety Report (ICSR) triage dashboard and disproportionality signal detection engine built with Streamlit and SQLite.

## Overview

PV-Sentinel provides automated, deterministic triage of adverse event reports without relying on external LLMs or third-party APIs. It enables rapid review of serious adverse events (SAEs), monitors regulatory reporting timelines, and computes disproportionality metrics (PRR, ROR, $\chi^2$) for safety signal identification.

## Key Features

* **Rule-Based ICSR Triage:** Evaluates cases for regulatory seriousness criteria (death, hospitalization, life-threatening, disability).
* **Automated Timeline Tracking:** Categorizes reporting deadlines (e.g., 7-day fatal/life-threatening vs. 15-day serious expedited reporting).
* **Disproportionality Analysis:** Implements standard $2 \times 2$ contingency calculations for:
  * Proportional Reporting Ratio (PRR)
  * Reporting Odds Ratio (ROR) with 95% Confidence Intervals
  * Chi-square ($\chi^2$) test with Yates' continuity correction
* **Local Persistence:** Standalone SQLite database for offline data integrity.
* **Clinical Dark Theme:** High-contrast UI optimized for sustained review workflows.

## Tech Stack

* **Language:** Python 3.10+
* **Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly
* **Database:** SQLite3

## Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/AbhijeetMadikeshwar/my_new_app.git](https://github.com/AbhijeetMadikeshwar/my_new_app.git)
cd my_new_app

2. Create and Activate a Virtual Environment
Bash
# Create environment
python -m venv venv

# Activate (Windows PowerShell):
.\venv\Scripts\Activate.ps1

# Activate (macOS/Linux):
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Launch the Dashboard
Bash
streamlit run app.py
License
Distributed under the MIT License.


