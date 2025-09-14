
# EV Battery Defect Detection — Streamlit Portfolio Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ginwalla-ev-battery-defect-dashboard.streamlit.app)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Short pitch**  
Interactive Streamlit demo that showcases a lightweight pipeline to detect defective EV battery cells from sensor logs. Includes data upload, synthetic data generator, validation, a model playground (demo), lightweight explainability, and PPTX export.

---

## Live demo
Live app: **`<paste your Streamlit URL here>`**

## Loom demo (4–5 min)
Watch: **https://www.loom.com/share/e9eb163f4e8a4de6a5be49afc1b2f4ca?sid=15fbb21a-4ecb-4b07-a1fe-e4328dd12875**

---

## Screenshots

### Home / Data Source
![Home screenshot](assets/synthetic_data.png)

### Validation — Data Health Checks
![Validation - Data Health Checks](assets/validation_data_health.png)

### Validation — Histogram & Heatmap
![Validation - Histogram](assets/validation_histogram.png)  
![Validation - Heatmap](assets/validation_heatmap.png)

### Model Playground
![Model Playground screenshot](assets/model_playground.png)

### Export / PPTX
![PPTX Export screenshot](assets/export.png)

---

## Features
- Upload CSV or generate synthetic data
- Dataset validation (missing values, invalid ranges)
- Visualizations: histograms, fast cached correlation heatmap
- Model playground with single-sample inference (demo stub) and simulated explainability
- Session history of predictions and CSV export
- Export a stakeholder PPTX with architecture image and summary

---

## Quick start (local)

1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/ev-battery-defect-dashboard.git
cd ev-battery-defect-dashboard