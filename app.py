import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load(r"best_fit_model.pkl")

# Page configuration
st.set_page_config(page_title="📱 Smartphone Discount Predictor", page_icon="📉", layout="wide")

# === Custom CSS ===
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700;900&family=Montserrat:wght@700&display=swap');

        /* Background: multi-stop gradient + overlay + blur */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(120deg, #b2fefa 0%, #f6d365 100%);
            background-size: cover;
            position: relative;
        }
        [data-testid="stAppViewContainer"]::before {
            content:'';
            position: fixed;
            top:0; left:0; right:0; bottom:0;
            background: rgba(255,255,255,0.75);
            backdrop-filter: blur(10px) saturate(120%);
            z-index: -1;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #fff3cd 40%, #e3f2fd 100%);
            border-right: 2px solid #fff176;
            font-family: 'Nunito', sans-serif;
        }
        /* Headings */
        h1, h2, h3 {
            font-family: 'Montserrat', sans-serif !important;
            font-weight: 900 !important;
            letter-spacing: 0.02em;
        }
        h1 {
            color: #009688 !important;
            font-size: 2.8rem !important;
            text-shadow: 1px 1px 8px #fffde4;
        }
        h2 {
            color: #8e24aa !important;
            font-size: 2.1rem !important;
        }
        h3 {
            color: #3949ab !important;
            font-size: 1.35rem !important;
            margin-top: 18px !important;
        }
        /* General font and color */
        html, body, [class*="css"] {
            font-family: 'Nunito', sans-serif !important;
            color: #232b2b !important;            
            font-weight: 400 !important;
        }
        p, li, .st-emotion-cache-1kyxreq {
            color: #232b2b !important;
            font-size: 1.12rem !important;
        }
        /* Card style containers */
        .card, .overview-card {
            background: rgba(255, 255, 255, 0.98);
            padding: 30px 30px 22px 30px;
            border-radius: 18px;
            box-shadow: 0 8px 32px rgba(0, 173, 239, 0.10), 0 1.5px 5px rgba(255, 205, 0, 0.08);
            margin-bottom: 30px;
            border-left: 7px solid #ff9800;
            border-top: 2px solid #8e24aa;
            font-size: 1.10rem;
        }
        /* Special Overview card border color */
        .overview-card {
            border-left: 7px solid #8e24aa;
            border-top: 2px solid #00bcd4;
        }
        /* Info highlight with icon */
        .info-box {
            padding: 17px 20px;
            background: linear-gradient(90deg,#fffde4 70%,#f6d365 100%);
            border: 1.5px solid #ffe082;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 1.05rem;
            color: #6d4c41;
            display: flex;
            align-items: center;
            font-weight: 600;
        }
        .info-box .icon-pulse {
            margin-right: 9px;
            animation: pulse 1.2s infinite;
            font-size: 1.42rem;
            color: #ffab00;
        }
        @keyframes pulse {
            0% { transform: scale(1);}
            50% { transform: scale(1.16);}
            100% { transform: scale(1);}
        }
        /* Prediction result box */
        .prediction-result {
            padding: 24px;
            margin-top: 20px;
            background: linear-gradient(90deg, #b2ff59 60%, #4dd0e1 100%);
            border: 2.2px solid #1769aa;
            border-radius: 14px;
            text-align: center;
            font-size: 1.7rem;
            color: #124217;
            font-weight: 800;
            letter-spacing: 0.01em;
            box-shadow: 0 4px 18px rgba(55,150,131,0.1);
            transition: box-shadow 0.24s;
        }
        .prediction-result:hover {
            box-shadow: 0 8px 28px rgba(40,188,177,0.18);
        }
        /* Animated button */
        button[kind="primary"], .stButton>button, .stButton>button:active {
            background: linear-gradient(90deg,#283e51 60%,#485563 100%);
            color: #fff !important;
            font-weight: 900 !important;
            font-size: 1.08rem !important;
            border-radius: 8px !important;
            border: none !important;
            transition: background 0.25s, transform 0.19s;
        }
        button[kind="primary"]:hover, .stButton>button:hover {
            background: linear-gradient(90deg,#2096ff 40%, #05ffa3 100%);
            color: #fff !important;
            transform: translateY(-2.5px) scale(1.03);
        }
        /* Labels and inputs brighter */
        label, .st-emotion-cache-1c7y2kd, .st-emotion-cache-1oa8j9e {
            color: #0d47a1 !important;
            font-weight: 700 !important;
            font-size: 1.14rem !important;
        }
        /* Input fields style */
        .stTextInput>div>div>input, .stNumberInput>div>input, .stSelectbox>div>div>div>input {
            background: #e3f2fd !important;
            border: 1.6px solid #b2ebf2 !important;
            border-radius: 8px !important;
            color: #124217 !important;
            font-weight: 700 !important;
        }
        /* Remove focus outline on select boxes and sliders for slick look */
        .stDropdown>div>div>div>div:focus, 
        .stSelectbox>div>div>div>div:focus,
        .stNumberInput>div>input:focus {
            box-shadow: 0 0 0 2.2px #ffd600 !important;
            border-color: #ffd600 !important;
        }
        /* List and links color for accessibility */
        ul, ol {
            color: #283593 !important;
        }
        a {
            color: #0288d1 !important;
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# === Sidebar Navigation ===
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "ℹ️ Overview", "📊 Prediction"])

# ===== HOME =====
if page == "🏠 Home":
    st.markdown(
        "<div class='card'>"
        "<h1 style='text-align:center; margin-bottom: 12px;'>📱 Products Discount Data Analysis & Estimation</h1>",
        unsafe_allow_html=True
    )
    st.image(
        "https://sm.mashable.com/t/mashable_in/photo/default/00-aa-art-cover-new-2024-09-16t132757941_zrt7.1248.jpg",
        use_container_width=True
    )
    st.markdown("""
        <h2>💡 Welcome to the Smartphone Discount Prediction App</h2>
        <p>This interactive tool lets you <b>predict the discount price</b> of smartphones using brand, specs, and features.</p>
        <h3>Why use this app?</h3>
        <ul>
          <li>📊 <b style='color:#42a5f5;'>For Sellers</b>: Plan competitive pricing strategies.</li>
          <li>🛒 <b style='color:#fbc02d;'>For Buyers</b>: Anticipate the best deal before purchase.</li>
          <li>📈 <b style='color:#8e24aa;'>For Analysts</b>: Study pricing patterns & tech trends.</li>
          <li>🧠 <b style='color:#4caf50;'>For Businesses</b>: Data-driven revenue optimization.</li>
        </ul>
    """, unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'><span class='icon-pulse'>🔔</span>Go to the <b>Prediction</b> tab to start forecasting smartphone discounts instantly.</div></div>",
        unsafe_allow_html=True
    )

# ===== OVERVIEW =====
elif page == "ℹ️ Overview":
    st.markdown(
        "<div class='overview-card'>"
        "<h1 style='text-align:center;'>📖 Project Overview</h1>"
        """
        <h2>🎯 <span style='color:#bc00dd;'>Objective</span></h2>
        <p><span style='color:#262626'><b>Predict the discount price</b> of smartphones using a trained ML pipeline that includes preprocessing and feature engineering.</span></p>
        <h2>📊 <span style='color:#2ec4b6;'>Dataset Sources</span></h2>
        <ul>
            <li>Amazon 📦 <span style='color:#03a9f4;'>(E-commerce)</span></li>
            <li>Flipkart 🛒 <span style='color:#fdc500;'>(E-commerce)</span></li>
        </ul>
        <h2>🔎 <span style='color:#3949ab;'>Features Used</span></h2>
        <ul>
            <li>Brand 🏷️</li>
            <li>RAM (GB) 💾</li>
            <li>ROM (GB) 📂</li>
            <li>Display Size (inches) 📱</li>
            <li>Battery (mAh) 🔋</li>
            <li>Front Camera (MP) 🤳</li>
            <li>Back Camera (MP) 📷</li>
        </ul>
        <h2>⚙️ <span style='color:#ff9800;'>How It Works</span></h2>
        <ol>
            <li style='margin-bottom:8px'>User provides smartphone specifications.</li>
            <li style='margin-bottom:8px'>Data is passed through a preprocessing pipeline.</li>
            <li>Model predicts the <b>discount price</b>.</li>
        </ol>
        <h2>💼 <span style='color:#43a047;'>Use Cases</span></h2>
        <ul>
            <li>Strategic pricing for <b style='color:#ef6c00;'>e-commerce platforms</b>.</li>
            <li>Smart purchasing decisions for <b style='color:#0288d1;'>customers</b>.</li>
            <li>Competitive <b style='color:#6a1b9a;'>market analysis</b> for analysts.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

# ===== PREDICTION =====
elif page == "📊 Prediction":
    st.markdown(
        "<div class='card'>"
        "<h1 style='color:#1769aa; text-align: center; margin-bottom:10px;'>📊 Predict Smartphone Discount Price</h1>"
        "<p style='font-size:1.09rem;'>Fill in the details below and click <b>Predict</b> to estimate the discount price.</p>",
        unsafe_allow_html=True
    )
    input_features = {}
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            input_features['Brand'] = st.selectbox("Select Brand",
                ['Samsung', 'Apple', 'Redmi', 'OnePlus', 'Realme', 'Vivo', 'Oppo', 'Motorola', 'Poco', 'Others'])
            input_features['RAM'] = st.number_input("Enter RAM (GB)", min_value=0.0, step=1.0, value=6.0)
            input_features['ROM'] = st.number_input("Enter ROM (GB)", min_value=0.0, step=16.0, value=128.0)
        with col2:
            input_features['Display_Size'] = st.number_input("Enter Display Size (inches)", min_value=0.0, step=0.1, value=6.5)
            input_features['Battery'] = st.number_input("Enter Battery Capacity (mAh)", min_value=0.0, step=500.0, value=5000.0)
            input_features['Front_Cam(MP)'] = st.number_input("Enter Front Camera (MP)", min_value=0.0, step=1.0, value=16.0)
            input_features['Back_Cam(MP)'] = st.number_input("Enter Back Camera (MP)", min_value=0.0, step=4.0, value=64.0)
    if st.button("🚀 Predict Discount Price", use_container_width=True):
        with st.spinner("🔄 Predicting... Please wait..."):
            df = pd.DataFrame([input_features])
        prediction = model.predict(df)
        # Convert prediction to scalar float
        pred_value = float(prediction.item())  # .item() extracts single value

        st.markdown(
            f"<div class='prediction-result'>💰 Predicted Discount Price: ₹{pred_value:,.2f}</div>",
            unsafe_allow_html=True
        )


