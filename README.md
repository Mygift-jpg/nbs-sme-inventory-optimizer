# 🇳🇬 NBS Inventory Optimizer for Lagos SMEs

[![Streamlit](https://img.shields.io/badge/Streamlit-Live-success)](https://nbs-sme-inventory-optimizer-fhdmjy35iu25mmqx2vkoib.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![GitHub](https://img.shields.io/badge/Build_In_Public-PROOF2026-green)](#)

**Predict monthly inventory needs using real Nigerian NBS food price data**

## Live Demo
**[Try the deployed app here!](https://nbs-sme-inventory-optimizer-fhdmjy35iu25mmqx2vkoib.streamlit.app/)**

## 📖 The Story Behind This Project
*This isn't just code—it's a testimony of God's faithfulness through 3 days of deployment battles, Python version wars, and community-powered breakthroughs. [Read the full journey on LinkedIn](#)*

## Business Problem
Small businesses in Lagos struggle with inventory management—overstocking leads to waste, understocking loses sales. This project provides a data-driven solution using publicly available Nigerian economic data.

## Nigerian Innovation
What makes this project unique:
- **10% accuracy improvement** from adding Nigerian context features (`is_festive`, `is_harvest`)
- **Real NBS data** (2017-2024)
- **Business insights in Nigerian Naira** (₦) for real Lagos SMEs

## Methodology
1. **Data Collection:** 8 years of NBS food price data
2. **Feature Engineering:** Nigerian seasonal patterns, price change analysis
3. **Modeling:** Random Forest vs Linear Regression comparison
4. **Deployment:** 3-day battle with Python 3.13 compatibility
5. **Validation:** Expert reviews from data scientists

## Key Results
- **Best Model:** Random Forest Regressor
- **Performance:** MAE = 37.88 units, R² = 0.9848
- **Key Insight:** `price_change` accounted for **94%** of predictive power
- **Business Impact:** Monthly price movement is the strongest signal for SME inventory planning

## Tech Stack
- **Language:** Python 3.13
- **Libraries:** Streamlit, Scikit-learn, Pandas, NumPy, Joblib
- **Model:** Random Forest Regressor (100 trees)
- **Deployment:** Streamlit Cloud
- **Validation:** MAE, R², Feature Importance Analysis

## Actual Project Structure

nbs-sme-inventory-optimizer

app.py                    # Streamlit application (LIVE!)

requirements.txt          # Dependencies for Python 3.13

random_forest_model.joblib # Trained model file

01_2026_NG_SME_Inventory_Optimizer_NBS_Data.ipynb  # Complete analysis
README.md                 # This file

## How to Run Locally
# Clone repository
git clone https://github.com/Mygift-jpg/nbs-sme-inventory-optimizer.git

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Community & Acknowledgments

This project survived deployment thanks to:

· Divine help through 3 days of Python version conflicts
· Community support from fellow data practitioners
· Expert validation from industry professionals
· God's faithfulness in every technical detail

Special thanks to everyone who engaged with the #BuildInPublic journey!

# About the Author

Blessing Okagbare – Data Practitioner & Founder, LB EdTech Solutions

# This project serves as a real-world case study for the LB EdTech Data Analytics Bootcamp, demonstrating how Nigerian data can solve Nigerian business problems while documenting the REAL journey of deployment struggles and victories.

# License

MIT License. Data from NBS Nigeria. Educational/portfolio use.


From January vision to January proof – documenting God's faithfulness in code. 🇳🇬
