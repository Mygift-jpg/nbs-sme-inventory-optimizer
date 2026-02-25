import os
import streamlit as st
import sys
import subprocess
import joblib
import numpy as np
from datetime import datetime
import pandas as pd

# Get port from Railway environment variable
PORT = int(os.environ.get("PORT", 8501))

# Set page config
st.set_page_config(
    page_title="NBS Inventory Optimizer for Lagos SMEs",
    page_icon="🇳🇬",
    layout="wide"
)

# Title and description
st.title("🇳🇬 NBS Inventory Optimizer for Lagos SMEs")
st.markdown("**Predict monthly inventory needs using Nigerian NBS food price data**")
st.markdown("*Currently optimized for: Rice (local sold loose)*")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('random_forest_model.joblib')
        
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is not None:
    st.sidebar.write(f"Model type: {type(model)}")
    if hasattr(model, 'coef_'):
        st.sidebar.write(f"Model coefficients: {model.coef_}")
        st.sidebar.write(f"Model intercept: {model.intercept_}")

# Sidebar for inputs
st.sidebar.header("📊 Input Parameters")
st.sidebar.markdown("---")

# Month selection
st.sidebar.subheader("1️⃣ Select Month to Predict")
prediction_month = st.sidebar.selectbox(
    "Which month are you planning for?",
    ["January", "February", "March", "April", "May", "June", 
     "July", "August", "September", "October", "November", "December"]
)
month_num = ["January", "February", "March", "April", "May", "June", 
             "July", "August", "September", "October", "November", "December"].index(prediction_month) + 1

# Calculate months_since_start (from January 2017)
current_year = datetime.now().year
months_since_start = (current_year - 2017) * 12 + month_num

# Price inputs
st.sidebar.subheader("2️⃣ Price Information (per 50kg bag)")
col1, col2 = st.sidebar.columns(2)
with col1:
    current_price = st.number_input(
        "Current Price (₦/bag)",
        min_value=10000,
        max_value=200000,
        value=65000,
        step=1000
    )
with col2:
    last_month_price = st.number_input(
        "Last Month Price (₦/bag)",
        min_value=10000,
        max_value=200000,
        value=67000,
        step=1000
    )

price_change = current_price - last_month_price
price_change_pct = (price_change / last_month_price) * 100

# FX Rate inputs
st.sidebar.subheader("3️⃣ Exchange Rate (USD/NGN)")
col3, col4 = st.sidebar.columns(2)
with col3:
    current_fx = st.number_input(
        "Current FX Rate (₦/$)",
        min_value=1000,
        max_value=2000,
        value=1500,
        step=10
    )
with col4:
    last_month_fx = st.number_input(
        "Last Month FX Rate (₦/$)",
        min_value=1000,
        max_value=2000,
        value=1480,
        step=10
    )

fx_change_pct = ((current_fx - last_month_fx) / last_month_fx) * 100

# Inflation
st.sidebar.subheader("4️⃣ Economic Indicators")
inflation = st.number_input(
    "Current Inflation Rate (%)",
    min_value=10.0,
    max_value=40.0,
    value=22.5,
    step=0.1
)

# Seasonal factors
st.sidebar.subheader("5️⃣ Seasonal Context")
col5, col6 = st.sidebar.columns(2)
with col5:
    is_festive = st.selectbox(
        "Festive Month?",
        ["No", "Yes"],
        help="December, January (Christmas/New Year), or April (Easter)"
    )
with col6:
    is_harvest = st.selectbox(
        "Harvest Season?",
        ["No", "Yes"],
        help="July-September (peak harvest period)"
    )

is_festive_num = 1 if is_festive == "Yes" else 0
is_harvest_num = 1 if is_harvest == "Yes" else 0

# Shock month
is_shock = st.sidebar.selectbox(
    "6️⃣ Shock Month?",
    ["No", "Yes"],
    help="Unusual price spike or market disruption?"
)
shock_num = 1 if is_shock == "Yes" else 0

# Predict button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 Predict Inventory Needs", type="primary")

# Main content area
if predict_button and model is not None:
    # Prepare input features
    input_features = np.array([[
        float(last_month_price),      
        float(current_fx),            
        float(inflation),             
        float(price_change_pct),      
        float(fx_change_pct),         
        float(is_festive_num),        
        float(is_harvest_num),        
        float(shock_num),             
        float(month_num)         
    ]], dtype=np.float32)
    
    
    try:
        prediction_bag = model.predict(input_features)[0]

        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📦 Predicted Price (per 50kg)",
                value=f"₦{int(prediction_bag):,}",
                delta=f"{price_change_pct:+.1f}% vs last month"
            )
        
        with col2:
            st.metric(
                label="💰 Price Change",
                value=f"₦{int(price_change):+,.0f}",
                delta=f"{price_change_pct:+.1f}%"
            )
        
        with col3:
            profit_now = current_price * 0.15 * 100
            profit_later = prediction_bag * 0.15 * 100
            st.metric(
                label="💵 Profit (100 bags)",
                value=f"₦{int(profit_later):,}",
                delta=f"₦{int(profit_later - profit_now):+,.0f}"
            )

        # =====================
        # SCENARIO SIMULATOR 
        # =====================

        st.markdown("---")
        st.subheader("🔄 What-If Scenario Simulator")

        st.markdown("""
        *See how changes in exchange rate could affect rice prices*
        """)

        # Get current prediction as baseline
        if 'prediction_bag' in locals():
            prediction_bag = prediction_bag
        else:
            prediction_bag = current_price  # fallback

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Adjust FX Rate (₦/$)**")
            
            # Slider for FX simulation
            sim_fx = st.slider(
                "Move to see impact",
                min_value=1000,
                max_value=1800,
                value=int(current_fx),
                step=10,
                key="fx_simulator"
            )
            # Calculate price impact (approximate rule from my model)
            fx_difference = sim_fx - current_fx
            price_impact = fx_difference * 3.8  # Based on my coefficient (~₦3.8 per ₦1 FX change)
            simulated_price = prediction_bag + price_impact

            st.metric(
                label="Simulated Price (per 50kg)",
                value=f"₦{int(simulated_price):,}",
                delta=f"{((simulated_price - prediction_bag)/prediction_bag)*100:+.1f}%"
            )

        with col2:
            st.markdown("**Quick Scenarios**")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🇳🇬 Naira Weakens (₦1,500)"):
                    st.session_state.fx_simulator = 1500
                    st.rerun()
                if st.button("📈 Festive Season"):
                    st.info("Festive would add ~₦2,500 to price")

        with col_b:
            if st.button("🇺🇸 Naira Strengthens (₦1,200)"):
                st.session_state.fx_simulator = 1200
                st.rerun()
            if st.button("🌾 Harvest Season"):
                st.info("Harvest would subtract ~₦1,400 from price")
                
    
        # ===================
        # SENSITIVITY CHART
        # ===================
        
        st.markdown("---")
        st.subheader("📈 FX Sensitivity Chart")

        # Generate data for chart
        fx_range = list(range(1000, 1801, 50))
        price_range = [prediction_bag + ((fx - current_fx) * 3.8) for fx in fx_range]

        # Create DataFrame
        chart_data = pd.DataFrame({
            'FX Rate (₦/$)': fx_range,
            'Predicted Price (₦)': [int(p) for p in price_range]  # Convert to integers
        })
            

        # Display chart
        st.line_chart(chart_data.set_index('FX Rate (₦/$)'))

        st.caption("Shows how rice price changes as FX rate moves")

        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 Quick Scenario")
        quick_fx = st.sidebar.radio(
            "Test FX change:",
            ["Current", "Weaker Naira (+100)", "Stronger Naira (-100)"]
        )
        
        if quick_fx == "Weaker Naira (+100)":
            test_fx = current_fx + 100
            st.sidebar.info(f"If FX = ₦{test_fx}, price would be ~₦{int(prediction_bag + (100*3.8)):,}")
        
        elif quick_fx == "Stronger Naira (-100)":
            test_fx = current_fx - 100
            st.sidebar.info(f"If FX = ₦{test_fx}, price would be ~₦{int(prediction_bag - (100*3.8)):,}")

        # Insights section
        st.markdown("---")
        st.subheader("🔑 Key Insights")
        
        insights = []
        
        if price_change_pct > 0:
            insights.append(f"📈 **Price increased by {price_change_pct:+.1f}%** - Prices are rising. Consider buying now.")
        elif price_change_pct < 0:
            insights.append(f"📉 **Price decreased by {abs(price_change_pct):.1f}%** - Prices are dropping. You may want to wait.")
        else:
            insights.append("📊 **Price stable** - Market is steady.")
        
        if is_festive == "Yes":
            insights.append("🎉 **Festive season active** - Expect higher demand and prices (+₦2,714 impact).")
        
        if is_harvest == "Yes":
            insights.append("🌾 **Harvest season** - Prices typically drop by ₦708 during this period.")
        
        if is_shock == "Yes":
            insights.append("⚡ **Shock month detected** - Market volatility expected (+₦984 momentum).")
        
        if fx_change_pct > 5:
            insights.append(f"💱 **Naira weakened** by {fx_change_pct:.1f}% - This pushes prices up.")
        elif fx_change_pct < -5:
            insights.append(f"💱 **Naira strengthened** by {abs(fx_change_pct):.1f}% - This may lower prices.")
        
        for insight in insights:
            st.info(insight)
        
        if not insights:
            st.info("✅ Market conditions are normal. No strong signals.")
        
        # Input summary - NOW INDENTED INSIDE THE TRY BLOCK
        with st.expander("📋 Input Summary"):
            # Create a clean dataframe with proper types
            summary_data = {
                'Parameter': [
                    'Prediction Month',
                    'Current Price (₦/bag)',
                    'Last Month Price (₦/bag)',
                    'Price Change (%)',
                    'Current FX Rate',
                    'Last Month FX Rate',
                    'FX Change (%)',
                    'Inflation (%)',
                    'Festive Month',
                    'Harvest Season',
                    'Shock Month'
                ],
                
                'Value': [
                    str(prediction_month),
                    f"₦{current_price:,}",           
                    f"₦{last_month_price:,}",        
                    f"{price_change_pct:.1f}%",      
                    f"₦{current_fx}",                
                    f"₦{last_month_fx}",           
                    f"{fx_change_pct:.1f}%",         
                    f"{inflation}%",                  
                    str(is_festive),                  
                    str(is_harvest),                  
                    str(is_shock)                    
                ]
            }
            # Convert to DataFrame - this ensures all arrays same length
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")

else:
    st.info("👈 **Adjust the parameters in the sidebar and click 'Predict Inventory Needs' to get started!**")
    st.subheader("📊 How This Works")
    st.markdown("""
    This tool uses a **Linear Regression model** trained on:
    - 📊 NBS Nigeria price data (2023-2024)
    - 📈 Extended projections (2025-2026) using FX rates and market trends
    
    **Key Drivers (Feb 2026 Update):**
    1. **Time Trend (29.8%)** - Prices generally increasing over time
    2. **FX Rate (28.9%)** - Naira value impacts rice prices significantly
    3. **Price Momentum (27.5%)** - Past price changes predict future
    
    ### Actual ₦ Impact of Each Factor:
    - 🎉 **Festive season**: **+₦2,714 per bag**
    - ⚡ **Shock month**: **+₦984 momentum**
    - 📈 **Price momentum**: **+₦819 per 1% increase**
    - 💰 **Inflation**: **+₦808 per 1% increase**
    - 🌾 **Harvest season**: **-₦708 discount**
    """)

# About section
st.markdown("---")
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    ### Model Details
    - **Algorithm:** Linear Regression (updated Feb 2026)
    - **Training Data:** 
      - 2023-2024: NBS Nigeria Food Price Data
      - 2025-2026: Extended using reliable market sources + FX data
    - **Current Product:** Rice (50kg bag)
    - **Target Market:** Lagos SMEs in food retail
    
    ### Feature Impact (₦ per unit change)
    | Feature | Impact (₦) |
    |---------|-----------|
    | Festive Season | +₦2,714 |
    | Shock Month | +₦984 |
    | Price Momentum | +₦819 per 1% |
    | Inflation | +₦808 per 1% |
    | Harvest Season | -₦708 |
    
    ### Key Insight (Feb 2026 Update)
    - Even when prices drop, absolute profit may be higher buying now!
    - Example: ₦65,000 now vs ₦59,526 later → ₦195,000 vs ₦178,579 profit (15% markup)
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>👩‍💻 Built by <a href='https://www.linkedin.com/in/blessing-okagbare' target='_blank'>Blessing Okagbare</a> | 
        <a href='https://github.com/Mygift-jpg/nbs-sme-inventory-optimizer' target='_blank'>View on GitHub</a> | 
        Part of #PROOF2026 Data Journey</p>
        <p style='font-size: 0.9em;'>🙏 LB EdTech Solutions | Building in Public</p>
    </div>
    """,
    unsafe_allow_html=True
)