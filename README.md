# EV Battery Defect Detection — Streamlit Portfolio Demo

**Short pitch**  
Interactive Streamlit demo that showcases a lightweight pipeline to detect defective EV battery cells from sensor logs. Includes data upload, synthetic data generator, validation, a model playground (demo), lightweight explainability, and PPTX export.

---

## Live demo
(If you deploy) Live app: **`<paste your Streamlit URL here>`**

## Loom demo (4-5 min)
Watch: **`https://www.loom.com/share/e9eb163f4e8a4de6a5be49afc1b2f4ca?sid=15fbb21a-4ecb-4b07-a1fe-e4328dd12875`**

---

## Screenshots
Home / KPIs  
![home](assets/home.png)

Model Playground (single-run + explanation)  
![playground](assets/playground.png)

Architecture exported PNG used in PPTX  
![arch](assets/arch.png)

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


## Screenshots

### Home / Data Source
![Home screenshot](assets/Export.png)

### Validation — Data Health Checks
![Validation - Data Health Checks](assets/Validation - Data Health Checks.png)

### Validation — Histogram & Heatmap
![Validation - Histogram](assets/Validation - Histogram.png)
![Validation - Heatmap](assets/Validation - Heat Map.png)

### Model Playground
![Model Playground screenshot](assets/Model Playground.png)

### Export / PPTX
![PPTX Export screenshot](assets/Export.png)
