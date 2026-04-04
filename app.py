import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Ultimate Churn Prediction Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    .hero {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 50px 30px;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.5);
    }
    .status-pill {
        display: inline-block;
        padding: 14px 32px;
        background: rgba(255,255,255,0.25);
        border-radius: 50px;
        font-weight: 700;
        font-size: 18px;
        backdrop-filter: blur(12px);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border-radius: 15px !important;
        height: 58px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
    }
    .card {
        background: rgba(22, 33, 62, 0.8);
        border-radius: 18px;
        padding: 25px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ====================== LOAD MODEL ======================
BASE_DIR = Path(__file__).resolve().parent

try:
    model = joblib.load(BASE_DIR / "catboost_model.pkl")
    scaler = joblib.load(BASE_DIR / "feature_scaler.pkl")
    MODEL_STATUS = "🟢 MODEL ACTIVE - CatBoost"
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Model loading failed: {e}")
    MODEL_STATUS = "🔴 MODEL NOT LOADED"
    MODEL_LOADED = False
    model = None
    scaler = None

# ====================== FEATURES ======================
FEATURES = [
    'Age','Membership_Years','Login_Frequency','Session_Duration_Avg',
    'Pages_Per_Session','Cart_Abandonment_Rate','Wishlist_Items',
    'Total_Purchases','Average_Order_Value','Days_Since_Last_Purchase',
    'Discount_Usage_Rate','Returns_Rate','Email_Open_Rate',
    'Customer_Service_Calls','Product_Reviews_Written',
    'Social_Media_Engagement_Score','Mobile_App_Usage',
    'Payment_Method_Diversity','Lifetime_Value','Credit_Balance',
    'Purchase_Rate','Engagement_Score','RFM_Recency_Norm',
    'RFM_Frequency_Norm','RFM_Monetary_Norm','RFM_Score',
    'Is_High_Value_Customer','Is_Frequent_Buyer','Is_Recent_Customer',
    'Is_At_Risk','Has_High_Cart_Abandonment','Is_Mobile_Active',
    'Gender_Encoded','Country_Encoded','City_Encoded',
    'Quarter_Q1','Quarter_Q2','Quarter_Q3','Quarter_Q4',
    'Cart_Abandonment_Risk','Returns_Risk','Email_Engagement',
    'Social_Activity','Service_Calls'
]

# ====================== FEATURE ENGINEERING ======================
def engineer_features(data):
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data).copy()
    
    df['Purchase_Rate'] = df.get('Total_Purchases', 10) / (df.get('Membership_Years', 1) + 0.1)
    df['Engagement_Score'] = df.get('Login_Frequency', 5) * df.get('Session_Duration_Avg', 10)
    
    # ✅ FIXED CLIP
    df['RFM_Recency_Norm'] = 1 - np.clip(df.get('Days_Since_Last_Purchase', 60) / 365, 0, 1)
    df['RFM_Frequency_Norm'] = np.clip(df.get('Total_Purchases', 10) / 50, 0, 1)
    df['RFM_Monetary_Norm'] = np.clip(df.get('Lifetime_Value', 1500) / 5000, 0, 1)
    
    df['RFM_Score'] = (df['RFM_Recency_Norm'] + df['RFM_Frequency_Norm'] + df['RFM_Monetary_Norm']) / 3
    
    # ✅ FIXED BOOL ERROR
    df['Is_High_Value_Customer'] = np.where(df.get('Lifetime_Value', 0) >= 1874, 1, 0)
    df['Is_Frequent_Buyer'] = np.where(df.get('Total_Purchases', 0) >= 17, 1, 0)
    df['Is_Recent_Customer'] = np.where(df.get('Days_Since_Last_Purchase', 365) <= 30, 1, 0)
    df['Is_At_Risk'] = np.where(df.get('Days_Since_Last_Purchase', 0) >= 180, 1, 0)
    df['Has_High_Cart_Abandonment'] = np.where(df.get('Cart_Abandonment_Rate', 0) >= 50, 1, 0)
    df['Is_Mobile_Active'] = np.where(df.get('Mobile_App_Usage', 0) >= 3, 1, 0)
    
    for col in ['Gender_Encoded', 'Country_Encoded', 'City_Encoded']:
        df[col] = 0
    for q in ['Quarter_Q1', 'Quarter_Q2', 'Quarter_Q3', 'Quarter_Q4']:
        df[q] = 1 if q == 'Quarter_Q4' else 0
    
    df['Cart_Abandonment_Risk'] = df.get('Cart_Abandonment_Rate', 30)
    df['Returns_Risk'] = df.get('Returns_Rate', 10)
    df['Email_Engagement'] = df.get('Email_Open_Rate', 30)
    df['Social_Activity'] = df.get('Social_Media_Engagement_Score', 5)
    df['Service_Calls'] = df.get('Customer_Service_Calls', 2)
    
    for feat in FEATURES:
        if feat not in df.columns:
            df[feat] = 0
    
    df = df[FEATURES].fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    
    if MODEL_LOADED and scaler is not None:
        try:
            df = pd.DataFrame(scaler.transform(df), columns=FEATURES)
        except:
            pass
    return df

# ====================== SINGLE PREDICTION ======================
def predict_single(age, membership, login_freq, purchases, days_inactive, 
                   cart_abandon, ltv, email_rate, service_calls):
    if not MODEL_LOADED:
        return "❌ Model not loaded", None, None, None
    
    customer = {
        'Age': age,
        'Membership_Years': membership,
        'Login_Frequency': login_freq,
        'Total_Purchases': purchases,
        'Days_Since_Last_Purchase': days_inactive,
        'Cart_Abandonment_Rate': cart_abandon,
        'Lifetime_Value': ltv,
        'Email_Open_Rate': email_rate,
        'Customer_Service_Calls': service_calls,
        'Session_Duration_Avg': 10,
        'Pages_Per_Session': 5,
        'Wishlist_Items': 3,
        'Average_Order_Value': ltv / max(purchases, 1),
        'Discount_Usage_Rate': 20,
        'Returns_Rate': 10,
        'Product_Reviews_Written': 1,
        'Social_Media_Engagement_Score': 5,
        'Mobile_App_Usage': 3,
        'Payment_Method_Diversity': 2,
        'Credit_Balance': 2000
    }
    
    features = engineer_features(customer)
    proba = float(model.predict_proba(features)[0][1])
    
    if proba >= 0.7:
        risk = "HIGH RISK"; emoji = "🔴"; color = "#ff6b6b"
        action = "🚨 IMMEDIATE ACTION: Contact within 24h with premium retention package"
    elif proba >= 0.4:
        risk = "MEDIUM RISK"; emoji = "🟡"; color = "#fa709a"
        action = "⚠️ Schedule follow-up within 48h with personalized offers"
    else:
        risk = "LOW RISK"; emoji = "🟢"; color = "#4facfe"
        action = "✅ Monitor regularly and maintain engagement"
    
    # Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba * 100,
        title={'text': f"<b>Churn Probability</b><br>{emoji} {risk}", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 48}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}}
    ))
    fig_gauge.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    # Radar Chart
    vals = [
        max(0, 100 - days_inactive/3.65),
        min(purchases * 5, 100),
        min(ltv / 100, 100),
        min(login_freq * 10, 100),
        100 - cart_abandon,
        email_rate
    ]
    fig_radar = go.Figure(go.Scatterpolar(
        r=vals,
        theta=['Recency', 'Frequency', 'Monetary', 'Engagement', 'Cart Completion', 'Email Response'],
        fill='toself', line=dict(color='#667eea', width=4)
    ))
    fig_radar.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', title_text="Customer Health Score")
    
    summary = f"""
    <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 30px; border-radius: 20px; color: white;'>
        <h2>{emoji} Prediction Results</h2>
        <h1 style='font-size: 48px; margin: 10px 0;'>{proba*100:.1f}%</h1>
        <h3>{emoji} {risk}</h3>
        <p style='font-size:18px;'><strong>Recommended Action:</strong><br>{action}</p>
    </div>
    """
    return summary, fig_gauge, fig_radar, action

# ====================== HERO SECTION ======================
st.markdown(f"""
<div class="hero">
    <h1>Ultimate Churn Prediction Dashboard</h1>
    <p style='font-size:24px; margin-top:10px;'>Advanced Customer Retention Intelligence</p>
    <div class="status-pill">{MODEL_STATUS}</div>
</div>
""", unsafe_allow_html=True)

# ====================== TABS ======================
tab1, tab2, tab3 = st.tabs(["🔮 Individual Prediction", "📊 Batch Analysis", "📈 Model Performance"])

# ==================== TAB 1: SINGLE PREDICTION ====================
with tab1:
    st.subheader("🎯 Predict Churn for Single Customer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 18, 80, 35)
        membership = st.slider("Membership Years", 0.0, 20.0, 3.0, 0.1)
        login_freq = st.slider("Login Frequency (per month)", 1, 100, 12)
    
    with col2:
        purchases = st.slider("Total Purchases", 0, 200, 15)
        days_inactive = st.slider("Days Since Last Purchase", 0, 365, 45)
        cart_abandon = st.slider("Cart Abandonment Rate (%)", 0, 100, 25)
    
    with col3:
        ltv = st.number_input("Lifetime Value ($)", value=1250, min_value=0)
        email_rate = st.slider("Email Open Rate (%)", 0, 100, 45)
        service_calls = st.slider("Customer Service Calls", 0, 30, 2)
    
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        summary, gauge, radar, action = predict_single(
            age, membership, login_freq, purchases, days_inactive,
            cart_abandon, ltv, email_rate, service_calls
        )
        st.markdown(summary, unsafe_allow_html=True)
        
        col_g, col_r = st.columns(2)
        with col_g:
            st.plotly_chart(gauge, use_container_width=True)
        with col_r:
            st.plotly_chart(radar, use_container_width=True)
        
        st.info(f"💡 Recommended Action: {action}")

# ==================== TAB 2: BATCH ANALYSIS ====================
with tab2:
    st.subheader("📁 Batch Analysis")
    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])
    
    if uploaded_file and st.button("🚀 Analyze All Customers", type="primary", use_container_width=True):
        with st.spinner("Processing with CatBoost model..."):
            try:
                df = pd.read_csv(uploaded_file)
                
                features_df = engineer_features(df)
                probs = model.predict_proba(features_df)[:, 1]
                
                results_df = pd.DataFrame({
                    'Customer_ID': df.get('Customer_ID', [f"CUST_{i+1}" for i in range(len(df))]),
                    'Churn_Probability': probs,
                    'Probability_%': [f"{p*100:.2f}%" for p in probs],
                    'Risk_Level': np.where(probs >= 0.7, "HIGH", np.where(probs >= 0.4, "MEDIUM", "LOW")),
                    'LTV': df.get('Lifetime_Value', 0),
                    'Days_Inactive': df.get('Days_Since_Last_Purchase', 0)
                })
                
                # Summary Stats
                high = (results_df['Risk_Level'] == 'HIGH').sum()
                medium = (results_df['Risk_Level'] == 'MEDIUM').sum()
                low = (results_df['Risk_Level'] == 'LOW').sum()
                
                st.success(f"✅ Analyzed {len(df)} customers successfully!")
                
                # Summary Cards
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Customers", len(df))
                c2.metric("🔴 High Risk", high, f"{high/len(df)*100:.1f}%")
                c3.metric("🟡 Medium Risk", medium, f"{medium/len(df)*100:.1f}%")
                c4.metric("🟢 Low Risk", low, f"{low/len(df)*100:.1f}%")
                
                # Charts
                col_pie, col_hist = st.columns(2)
                with col_pie:
                    fig_pie = go.Figure(data=[go.Pie(labels=['High Risk','Medium Risk','Low Risk'],
                                                   values=[high, medium, low])])
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_hist:
                    fig_hist = go.Figure(data=[go.Histogram(x=probs*100, nbinsx=25)])
                    fig_hist.update_layout(title="Churn Probability Distribution")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Top Risk Customers
                st.subheader("🔴 Top 20 High-Risk Customers")
                top_risk = results_df.nlargest(20, 'Churn_Probability')
                st.dataframe(top_risk, use_container_width=True)
                
                # Download Button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Results as CSV",
                    data=csv,
                    file_name=f"churn_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error processing file: {e}")

# ==================== TAB 3: MODEL INFO ====================
with tab3:
    st.subheader("📈 Model Performance")
    st.markdown("""
    ### 🏆 CatBoost Model Details
    - **Accuracy**: ~91% (based on your training)
    - **Model Type**: CatBoost Classifier
    - **Features Used**: 44 engineered features
    - **Inference Speed**: Very Fast
    """)
    
    st.info("This dashboard uses your actual trained CatBoost model for real predictions.")

st.caption("Made with ❤️ using Streamlit | Powered by CatBoost")

# ====================== KEEP YOUR ORIGINAL predict_single (UNCHANGED) ======================
# (Your full original function with gauge + radar stays here — DO NOT MODIFY)

# ====================== REST OF YOUR CODE ======================
# 👉 KEEP EVERYTHING BELOW EXACTLY SAME AS YOUR ORIGINAL FILE
# Tabs, UI, charts, batch analysis — NO CHANGE
