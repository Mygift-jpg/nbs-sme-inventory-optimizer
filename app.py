import streamlit as st
import joblib
import numpy as np
from datetime import datetime
import pandas as pd

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
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Sidebar for inputs
st.sidebar.header("📊 Input Parameters")
st.sidebar.markdown("---")

# Month selection for prediction
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
st.sidebar.subheader("2️⃣ Price Information")
col1, col2 = st.sidebar.columns(2)
with col1:
    current_price = st.number_input(
        "Current Price (₦/kg)",
        min_value=100,
        max_value=10000,
        value=1500,
        step=50
    )
with col2:
    last_month_price = st.number_input(
        "Last Month Price (₦/kg)",
        min_value=100,
        max_value=10000,
        value=1450,
        step=50
    )

price_change = current_price - last_month_price

# Last month's sales
st.sidebar.subheader("3️⃣ Sales History")
units_sold_lag1 = st.sidebar.number_input(
    "Last Month's Units Sold",
    min_value=0,
    max_value=5000,
    value=1008,  # Median from your data
    step=10,
    help="Enter your actual sales from last month. Default is the typical median (1008 units)."
)

# Seasonal factors
st.sidebar.subheader("4️⃣ Seasonal Context")
is_festive = st.sidebar.selectbox(
    "Festive Month?",
    ["No", "Yes"],
    help="December, January (Christmas/New Year), or April (Easter)"
)
is_harvest = st.sidebar.selectbox(
    "Harvest Season?",
    ["No", "Yes"],
    help="July-September (peak harvest period)"
)

# Convert to numeric
is_festive_num = 1 if is_festive == "Yes" else 0
is_harvest_num = 1 if is_harvest == "Yes" else 0

# Calculate month_sin and month_cos for cyclical encoding
month_sin = np.sin(2 * np.pi * month_num / 12)
month_cos = np.cos(2 * np.pi * month_num / 12)

# Create prediction button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 Predict Inventory Needs", type="primary")

# Main content area
if predict_button and model is not None:
    # Prepare input features in correct order
    input_features = np.array([[
        months_since_start,
        units_sold_lag1,
        price_change,
        month_sin,
        month_cos,
        is_festive_num,
        is_harvest_num
    ]])
    
    # Make prediction
    try:
        prediction = model.predict(input_features)[0]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📦 Predicted Units Needed",
                value=f"{int(prediction):,}",
                delta=f"{int(prediction - units_sold_lag1):,} vs last month"
            )
        
        with col2:
            st.metric(
                label="💰 Price Change",
                value=f"₦{price_change:,.0f}",
                delta=f"{(price_change/last_month_price*100):.1f}%"
            )
        
        with col3:
            revenue_estimate = prediction * current_price
            st.metric(
                label="💵 Est. Revenue",
                value=f"₦{revenue_estimate:,.0f}"
            )
        
        # Insights section
        st.markdown("---")
        st.subheader("💡 Key Insights")
        
        # Dynamic insights based on inputs
        insights = []
        
        if price_change > 0:
            insights.append(f"⚠️ **Price increased by ₦{price_change:.0f}** - Demand may decrease slightly.")
        elif price_change < 0:
            insights.append(f"✅ **Price decreased by ₦{abs(price_change):.0f}** - Demand may increase.")
        else:
            insights.append("📊 **Price stable** - Demand follows typical seasonal patterns.")
        
        if is_festive == "Yes":
            insights.append("🎉 **Festive season active** - Expect 15-25% higher demand than usual.")
        
        if is_harvest == "Yes":
            insights.append("🌾 **Harvest season** - Supply is typically abundant, prices may stabilize.")
        
        change_pct = ((prediction - units_sold_lag1) / units_sold_lag1) * 100
        if change_pct > 10:
            insights.append(f"📈 **Significant increase expected** - Stock up {abs(change_pct):.1f}% more than last month.")
        elif change_pct < -10:
            insights.append(f"📉 **Demand may drop** - Consider {abs(change_pct):.1f}% less inventory to avoid waste.")
        
        for insight in insights:
            st.info(insight)
        
        # Show input summary
        with st.expander("📋 Input Summary"):
            summary_df = pd.DataFrame({
                'Parameter': [
                    'Prediction Month',
                    'Months Since Start',
                    'Current Price',
                    'Last Month Price',
                    'Price Change',
                    'Last Month Sales',
                    'Festive Month',
                    'Harvest Season'
                ],
                'Value': [
                    prediction_month,
                    months_since_start,
                    f"₦{current_price:,}",
                    f"₦{last_month_price:,}",
                    f"₦{price_change:,}",
                    f"{units_sold_lag1:,} units",
                    is_festive,
                    is_harvest
                ]
            })
            st.dataframe(summary_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")

else:
    # Default view before prediction
    st.info("👈 **Adjust the parameters in the sidebar and click 'Predict Inventory Needs' to get started!**")
    
    # Show example/demo
    st.subheader("📊 How This Works")
    st.markdown("""
    This tool uses a **Random Forest model** trained on 5 years of Nigerian National Bureau of Statistics (NBS) food price data combined with synthetic sales patterns.
    
    **Key Features:**
    1. **Price Change Analysis** (94% feature importance) - The primary driver of demand
    2. **Seasonal Patterns** - Cyclical monthly trends encoded as sine/cosine
    3. **Nigerian Context** - Festive periods (Dec/Jan) and harvest seasons (Jul-Sep)
    4. **Sales Momentum** - Previous month's performance informs predictions
    
    **Typical Use Case:**
    - A Lagos SME owner planning February inventory
    - Rice price increased from ₦1,450 to ₦1,500/kg
    - Last month sold 1,008 units
    - Not festive, not harvest season
    - **Prediction: ~950-1,050 units needed**
    """)

# About section
st.markdown("---")
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    ### Model Details
    - **Algorithm:** Random Forest Regressor (100 trees)
    - **Training Data:** NBS Nigeria Food Price Data (2017-2024) + Synthetic sales patterns
    - **Current Product:** Rice (local sold loose)
    - **Target Market:** Lagos SMEs in food retail
    
    ### Feature Importance
    | Feature | Importance |
    |---------|-----------|
    | Price Change | 94% |
    | Sales Lag | 3% |
    | Month (Cyclical) | 2% |
    | Seasonal Flags | 1% |
    
    ### Nigerian Context Integration
    - **Festive Months:** December, January (Christmas/New Year), April (Easter)
    - **Harvest Season:** July-September (peak supply period)
    - Adding these contextual features improved model accuracy by **10%**
    
    ### Limitations
    - Uses synthetic `units_sold` data for demonstration
    - Real implementation requires connecting to actual business sales records
    - Model trained on rice; would need retraining for other products
    
    ### Data Source
    All price data sourced from [National Bureau of Statistics (NBS) Nigeria](https://nigerianstat.gov.ng/)
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