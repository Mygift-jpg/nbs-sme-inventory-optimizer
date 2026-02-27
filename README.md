# 🇳🇬 NBS SME Inventory Optimizer for Lagos SMEs

[![Streamlit](https://img.shields.io/badge/Streamlit-Live-success)](https://nbs-sme-inventory-optimizer-fhdmjy35iu25mmqx2vkoib.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![GitHub](https://img.shields.io/badge/Build_In_Public-PROOF2026-green)](#)

**Predict monthly rice prices for Lagos SMEs using real Nigerian data + FX awareness**

## 🔴 Live Demo
**[Try the app here!](https://nbs-sme-inventory-optimizer-fhdmjy35iu25mmqx2vkoib.streamlit.app/)**

## 📖 The Story
*This project survived 3 weeks of debugging, 1 stubborn model, and multiple "I quit" moments. [Read the full journey on Linkedln - https://www.linkedin.com/in/blessing-okagbare]*

## 🇳🇬 Nigerian Problem
Lagos rice sellers struggle to know **WHEN to buy**. Prices move with FX, seasons, and market shocks. Most rely on guesswork.

**This tool gives them data-driven answers.**

## 📊 What Makes It Unique
- ✅ **FX-aware predictions** (₦ impact of exchange rates)
- ✅ **Seasonal context** (Festive +₦2,714, Harvest -₦708)
- ✅ **"What-If" simulator** (test scenarios before buying)
- ✅ **Sensitivity chart** (see risk at a glance)
- ✅ **Insights in Nigerian Naira** (not abstract numbers)

## 🛠️ Methodology
1. **Data:** 2023-2024 NBS + 2025-2026 FX-extended projections
2. **Features:** Price momentum, FX changes, seasons, shocks
3. **Model:** Linear Regression (transparent, not black box)
4. **Deployment:** Streamlit Cloud (after 3 weeks of debugging 😅)

## 📈 Key Drivers (₦ Impact)
| Factor | Impact |
|--------|--------|
| 🎉 Festive Season | **+₦2,714** |
| ⚡ Shock Month | **+₦984** |
| 📈 Price Momentum | **+₦819 per 1%** |
| 💰 Inflation | **+₦808 per 1%** |
| 🌾 Harvest Season | **-₦708** |

## 🚀 Features
- 📊 **FX Scenario Simulator** — slide to see price impact
- 📈 **Sensitivity Chart** — visualize risk instantly
- ⚡ **Quick Scenarios** — one-click "what if" tests
- 🔑 **Key Insights** — clear, actionable advice
- 📋 **Full Transparency** — see exactly how decisions are made

## 💻 Tech Stack
- **Language:** Python 3.12
- **Framework:** Streamlit
- **Model:** Scikit-learn LinearRegression
- **Data:** Pandas, NumPy
- **Deployment:** Streamlit Cloud

## 📁 Project Structure
```
nbs-sme-inventory-optimizer/
├── app.py                    # Main Streamlit app
├── random_forest_model.joblib # Trained Linear Regression model
├── requirements.txt          # Dependencies
├── case_study.md             # Full 4-week journey
├── train_model.py             # Training script
├── data_v2/                   # Data folder
└── README.md                  # This file
```

## 🚀 Run Locally
```
git clone https://github.com/Mygift-jpg/nbs-sme-inventory-optimizer.git
cd nbs-sme-inventory-optimizer
pip install -r requirements.txt
streamlit run app.py
```

## 🙏 Acknowledgments
- **God's faithfulness** through every error message
- **Community support** from #PROOF2026
- **NBS Nigeria** for the foundational data
- **Everyone who didn't let me quit**

## 👩‍💻 Author
**Blessing Okagbare**  
Data Practitioner | Founder, LB EdTech Solutions  
Building in Public • 🇳🇬 • #PROOF2026

## 📄 License
MIT License. Data from NBS Nigeria. Educational/portfolio use.

**From January vision to February victory — documenting every struggle and breakthrough.** 🇳🇬
```
