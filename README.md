# EV Battery Defect Detection — Streamlit Portfolio Demo

**Short pitch**  
Interactive Streamlit demo that showcases a lightweight pipeline to detect defective EV battery cells from sensor logs. Includes data upload, synthetic data generator, validation, a model playground (demo), lightweight explainability, and PPTX export.

---

## Live demo
(If you deploy) Live app: **`<paste your Streamlit URL here>`**

## Loom demo (2–3 min)
Watch: **`<paste your Loom URL here>`**

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
