"""
app.py — Dynamic Pricing Dashboard
Run after completing the notebook: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="Dynamic Pricing Engine", page_icon="💰", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #0a0f1e; color: #e8e6df; }
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .price-tag {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        color: #34d399;
    }
    .label { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }
    .up   { color: #34d399; font-weight: 700; }
    .down { color: #f87171; font-weight: 700; }
    .neutral { color: #fbbf24; font-weight: 700; }
    div[data-testid="stSidebar"] { background: #060d1a; border-right: 1px solid #1f2937; }
    .stSlider > div { color: #e8e6df !important; }
    .stSelectbox label { color: #9ca3af !important; }
    .stSlider label { color: #9ca3af !important; }
    .stButton > button {
        background: #2563eb !important; color: #fff !important;
        font-weight: 700 !important; border: none !important;
        border-radius: 8px !important; width: 100% !important;
        padding: 0.7rem !important; font-size: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model + artifacts ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('categories.pkl', 'rb') as f:
        categories = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, le, categories, features

try:
    model, le, categories, features = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:1.8rem;font-weight:700;color:#60a5fa;margin-bottom:4px">
💰 Dynamic Pricing Engine
</div>
<div style="color:#6b7280;font-size:0.9rem;margin-bottom:2rem">
XGBoost / LightGBM · Olist E-commerce · Feature-driven price recommendations
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("Model not found. Please run the notebook first to train and save the model.")
    st.code("# Run all cells in notebook.py first, then come back here", language="python")
    st.stop()

# ── Sidebar — Product Inputs ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:1.1rem;font-weight:700;color:#60a5fa;margin-bottom:1rem">Product Details</div>', unsafe_allow_html=True)

    category = st.selectbox("Product category", categories)
    weight   = st.slider("Weight (grams)", 100, 5000, 500, step=50)
    length   = st.slider("Length (cm)", 5, 100, 20)
    height   = st.slider("Height (cm)", 5, 100, 15)
    width    = st.slider("Width (cm)", 5, 100, 15)
    freight  = st.slider("Freight value (BRL)", 5.0, 80.0, 15.0, step=0.5)
    review   = st.slider("Avg review score", 1.0, 5.0, 4.2, step=0.1)
    month    = st.selectbox("Month", list(range(1, 13)), index=5)
    current_price = st.number_input("Your current price (BRL)", min_value=1.0, value=100.0, step=5.0)

    predict_btn = st.button("Get Price Recommendation")


# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    # Build feature vector
    cat_encoded      = le.transform([category])[0]
    volume           = length * height * width
    freight_ratio    = freight / (current_price + freight + 1e-6)

    # Use median seller volume and demand score for a new product
    seller_vol       = 50
    demand_score     = 0.3
    cat_avg_price    = current_price   # assume close to current
    price_vs_avg     = 1.0
    review_quality   = review * np.log1p(seller_vol)

    input_data = pd.DataFrame([{
        'freight_value':          freight,
        'avg_review_score':       review,
        'product_weight_g':       weight,
        'product_volume_cm3':     volume,
        'freight_ratio':          freight_ratio,
        'category_demand_score':  demand_score,
        'category_avg_price':     cat_avg_price,
        'price_vs_category_avg':  price_vs_avg,
        'seller_order_volume':    seller_vol,
        'review_quality':         review_quality,
        'order_month':            month,
        'order_dayofweek':        2,
        'order_quarter':          (month - 1) // 3 + 1,
        'category_encoded':       cat_encoded,
    }])[features]

    recommended_price = float(model.predict(input_data)[0])
    price_gap         = recommended_price - current_price
    pct_change        = (price_gap / current_price) * 100

    # ── Results ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Current price</div>
            <div style="font-family:'Space Mono',monospace;font-size:2.2rem;font-weight:700;color:#9ca3af">
                R$ {current_price:.2f}
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Recommended price</div>
            <div class="price-tag">R$ {recommended_price:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        direction = "up" if price_gap > 0 else "down" if price_gap < 0 else "neutral"
        arrow     = "▲" if price_gap > 0 else "▼" if price_gap < 0 else "●"
        action    = "Increase price" if price_gap > 0 else "Reduce price" if price_gap < 0 else "Price is optimal"
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Recommendation</div>
            <div class="{direction}" style="font-size:1.4rem">{arrow} {abs(pct_change):.1f}%</div>
            <div style="color:#9ca3af;font-size:0.85rem;margin-top:6px">{action} by R$ {abs(price_gap):.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Business Impact ───────────────────────────────────────────────────────
    st.markdown("### Revenue impact simulation")
    st.markdown("How does your revenue change at different price points?")

    price_range    = np.linspace(max(1, current_price * 0.5), current_price * 1.8, 50)
    demand_penalty = np.exp(-0.005 * (price_range - recommended_price) ** 2)
    revenue        = price_range * demand_penalty * 100  # assume 100 units base

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    ax.plot(price_range, revenue, color='#34d399', linewidth=2.5)
    ax.axvline(current_price,     color='#9ca3af', linestyle='--', linewidth=1.5, label=f'Current: R${current_price:.0f}')
    ax.axvline(recommended_price, color='#60a5fa', linestyle='--', linewidth=1.5, label=f'Recommended: R${recommended_price:.0f}')
    ax.fill_between(price_range, revenue, alpha=0.15, color='#34d399')

    ax.set_xlabel('Price (BRL)', color='#9ca3af')
    ax.set_ylabel('Estimated Revenue (BRL)', color='#9ca3af')
    ax.tick_params(colors='#9ca3af')
    ax.spines['bottom'].set_color('#374151')
    ax.spines['left'].set_color('#374151')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1f2937', labelcolor='#e8e6df', framealpha=0.8)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Key Insights ──────────────────────────────────────────────────────────
    st.markdown("### What's driving this recommendation?")

    insights = []
    if review >= 4.5:
        insights.append(("High review score", f"{review}/5.0 — customers trust this product, supports premium pricing"))
    elif review < 3.5:
        insights.append(("Low review score", f"{review}/5.0 — price pressure to remain competitive"))

    if freight / current_price > 0.2:
        insights.append(("High freight ratio", f"{freight/current_price*100:.0f}% of price — may deter buyers, consider pricing adjustment"))

    if volume > 10000:
        insights.append(("Large product volume", f"{volume:,} cm³ — bulky product, freight cost is significant factor"))

    if not insights:
        insights.append(("Balanced signals", "Product metrics are within normal range for this category"))

    for title, detail in insights:
        st.markdown(f"**{title}** — {detail}")

else:
    # Default state — show instructions
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:50vh;gap:12px;text-align:center">
        <div style="font-size:3rem">💰</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.2rem;color:#60a5fa">
            Configure your product in the sidebar
        </div>
        <div style="color:#4b5563;max-width:420px;line-height:1.8;font-size:0.9rem">
            Enter product details on the left, then click<br>
            <strong style="color:#9ca3af">Get Price Recommendation</strong><br>
            to see the model's optimal price and revenue impact.
        </div>
        <div style="margin-top:1.5rem;display:flex;gap:10px;flex-wrap:wrap;justify-content:center">
            <span style="background:#1f2937;color:#6b7280;padding:4px 14px;border-radius:20px;
                         font-size:0.78rem;border:1px solid #374151">XGBoost / LightGBM</span>
            <span style="background:#1f2937;color:#6b7280;padding:4px 14px;border-radius:20px;
                         font-size:0.78rem;border:1px solid #374151">SHAP explainability</span>
            <span style="background:#1f2937;color:#6b7280;padding:4px 14px;border-radius:20px;
                         font-size:0.78rem;border:1px solid #374151">Revenue simulation</span>
            <span style="background:#1f2937;color:#6b7280;padding:4px 14px;border-radius:20px;
                         font-size:0.78rem;border:1px solid #374151">100k+ real orders</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
