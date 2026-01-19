# 🌊 DN Flood Forecast

**DN Flood Forecast** is a Machine Learning project that predicts flood levels based on environmental and hydrological data.  
This tool demonstrates how data-driven models can assist in early warning systems for flood risk mitigation.

---

## 🧠 Project Overview

Flooding is a natural disaster with significant impact on communities.  
The goal of this project is to:

✅ Use historical flood and weather data  
✅ Train Machine Learning models to forecast future flood levels  
✅ Evaluate and visualize predictions for decision support

This repository includes:
- Data preprocessing
- Model training & evaluation
- Prediction visualization tools
- Notebooks for experimentation

---

## 📂 Repository Structure

```text
DN-Flood-Forecast/
├── data/                # Raw & processed datasets
├── notebooks/           # Jupyter notebooks for experiments
├── src/                 # Source scripts for training & evaluation
├── models/              # Trained model checkpoints
├── results/             # Prediction plots / reports
├── requirements.txt     # Python dependencies
└── README.md
🛠️ Tech Stack
<p align="left"> <img src="https://skillicons.dev/icons?i=python,pandas,matplotlib,scikit,git" /> </p>
Python – Main programming language

NumPy & Pandas – Data manipulation

Scikit-learn – Machine Learning models

Matplotlib / Seaborn – Visualization

Jupyter Notebook – Interactive experimentation

🚀 Getting Started
1. Clone Repository
bash
Sao chép mã
git clone https://github.com/loiphan2202/DN-Flood-Forecast.git
cd DN-Flood-Forecast
2. Install Dependencies
bash
Sao chép mã
pip install -r requirements.txt
3. Explore Datasets
All raw and cleaned datasets are located in the data/ folder.
Open the notebooks to understand structure and preprocessing steps.

🧪 Running Experiments
Open and run any notebook inside notebooks/:

bash
Sao chép mã
jupyter notebook
Try these key notebooks:

Data Exploration & Cleanup

Model Training & Evaluation

Prediction Visualization

📊 Example
Example forecast plot:


(Replace with your actual images after you add some)

🧠 How It Works
Load & preprocess data

Train ML models

Evaluate performance

Visualize forecast results

Typical models used:

Linear Regression

Random Forest

XGBoost
(Update accordingly if you use different models)

📈 Results & Metrics
You can check:

Mean Absolute Error (MAE)

R² Score

Prediction vs Actual visual plots

📌 Include your model performance summary or table here.

📫 Contact
For questions or contributions:

🔗 GitHub: https://github.com/loiphan2202
📧 Email: loiphan2102004ptl@gmail.com

⭐ Feel free to explore, experiment, and expand this project for real-world flood forecasting applications!

---

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/github/license/loiphan2202/DN-Flood-Forecast)

Kết quả so sánh Actual vs Pred

📖 Model Details
Thêm bảng:

Model	MAE	R²
RandomForest	2.12	0.86
XGBoost	1.98	0.89
