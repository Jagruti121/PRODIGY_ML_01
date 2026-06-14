# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# # =====================================================================
# # 1. LOAD THE DATASET
# # =====================================================================
# data_path = 'train.csv'

# if not os.path.exists(data_path):
#     raise FileNotFoundError(
#         f"'{data_path}' not found. Please place 'train.csv' in the same directory as this script."
#     )

# df = pd.read_csv(data_path)
# print("--- Dataset Loaded Successfully ---")
# print(f"Dataset Shape: {df.shape}\n")

# # =====================================================================
# # 2. FEATURE SELECTION & PREPROCESSING
# # =====================================================================
# df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath'])

# features = ['GrLivArea', 'BedroomAbvGr', 'TotalBathrooms']
# target = 'SalePrice'

# data = df[features + [target]].copy()

# print("Missing values per column:")
# print(data.isnull().sum())

# data = data.dropna()
# print(f"Cleaned Dataset Shape: {data.shape}\n")

# # =====================================================================
# # 3. EXPLORATORY DATA ANALYSIS (EDA) - Visualizing Relationships
# # =====================================================================
# print("--- Generating Correlation Plot ---")
# plt.figure(figsize=(8, 6))
# # Using lowercase 'coolwarm' explicitly to ensure system compatibility
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix of Selected Features')
# plt.show()

# # =====================================================================
# # 4. SPLITTING DATA INTO TRAINING AND TESTING SETS
# # =====================================================================
# X = data[features]
# y = data[target]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("--- Data Split ---")
# print(f"Training Features Shape: {X_train.shape}")
# print(f"Testing Features Shape: {X_test.shape}\n")

# # =====================================================================
# # 5. INITIALIZE AND TRAIN THE LINEAR REGRESSION MODEL
# # =====================================================================
# print("--- Training Linear Regression Model ---")
# model = LinearRegression()
# model.fit(X_train, y_train)

# print(f"Model Intercept (b0): {model.intercept_:.2f}")
# for feature, coef in zip(features, model.coef_):
#     print(f"Coefficient for {feature} (b1): {coef:.2f}")
# print()

# # =====================================================================
# # 6. MODEL EVALUATION
# # =====================================================================
# print("--- Evaluating Model ---")
# y_pred = model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
# print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
# print(f"R-squared Score (R2): {r2:.4f}\n")

# # =====================================================================
# # 7. VISUALIZING PREDICTIONS VS ACTUAL PRICES
# # =====================================================================
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# plt.xlabel('Actual Sale Price ($)')
# plt.ylabel('Predicted Sale Price ($)')
# plt.title('Actual vs. Predicted House Prices')
# plt.tight_layout()
# plt.show()

# # =====================================================================
# # 8. MAKE A CUSTOM PREDICTION (INFERENCE)
# # =====================================================================
# print("--- Making a Custom Prediction ---")
# custom_house = pd.DataFrame([{
#     'GrLivArea': 2000,
#     'BedroomAbvGr': 3,
#     'TotalBathrooms': 2.5
# }])

# predicted_price = model.predict(custom_house)[0]
# print(f"Custom House Specs: 2,000 sqft, 3 Bed, 2.5 Bath")
# print(f"Predicted Sale Price: ${predicted_price:,.2f}")





import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =====================================================================
# 1. PAGE CONFIGURATION & THEME SETUP
# =====================================================================
st.set_page_config(
    page_title="Enterprise Real Estate Valuator",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force elegant plotting style for professional presentation
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'axes.facecolor': '#f8f9fa',
    'figure.facecolor': '#ffffff',
    'grid.color': '#e9ecef'
})

# =====================================================================
# 2. DATA PIPELINE & CACHED CORE ENGINE
# =====================================================================
@st.cache_resource
def run_analytics_and_model():
    """Loads data, engineers features, splits data, and trains the model."""
    if not os.path.exists('train.csv'):
        return None
    
    df = pd.read_csv('train.csv')
    
    # Feature Engineering
    df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath'])
    features = ['GrLivArea', 'BedroomAbvGr', 'TotalBathrooms']
    target = 'SalePrice'
    
    # Processed Clean Data Slice
    clean_data = df[features + [target]].dropna()
    X = clean_data[features]
    y = clean_data[target]
    
    # Holdout validation splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Fit
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Performance Evaluation
    y_pred = model.predict(X_test)
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    return model, clean_data, X_test, y_test, y_pred, metrics

# Run the core data backend execution
engine_output = run_analytics_and_model()

if engine_output is None:
    st.error("🚨 Missing Source Dataset: Please place 'train.csv' in the root script folder.")
    st.stop()

model, clean_data, X_test, y_test, y_pred, metrics = engine_output

# =====================================================================
# 3. SIDEBAR SYSTEM DOCUMENTATION
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluent/100/000000/real-estate.png", width=80)
    st.title("System Parameters")
    st.markdown("""
    This corporate tool runs a multiple multivariate **OLS Linear Regression Model** optimizing real estate assets against baseline metrics.
    
    **Dataset Profile:**
    * **Observations:** 1,460 properties
    * **Engineered Scope:** 3 target parameters
    * **Optimization Split:** 80% Train / 20% Valid
    """)
    st.divider()
    st.caption("v2.2.0 • Optimized Presentation Layout")

# =====================================================================
# 4. MAIN USER INTERFACE HEADER
# =====================================================================
st.title("🏢 Enterprise Real Estate Valuation Engine")
st.markdown("Assess portfolio market values using structural parameters and regression diagnostics.")

# Create the primary tab structure
tab_predict, tab_analytics = st.tabs(["🔮 Real-Time Asset Pricing", "📈 Analytics & Explainability Insights"])

# =====================================================================
# TAB 1: REAL-TIME ASSET PRICING TOOL
# =====================================================================
with tab_predict:
    st.subheader("Property Structural Configuration")
    st.write("Modify properties below to run live inference modeling:")
    
    # Structured Layout Grid
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    
    with row1_col1:
        sqft = st.number_input(
            "Square Footage (GLA):", 
            min_value=300, max_value=8000, value=1800, step=100,
            help="Gross Living Area Square Feet above ground level."
        )
    with row1_col2:
        bedrooms = st.slider(
            "Bedrooms:", 
            min_value=1, max_value=6, value=3, step=1,
            help="Total bedrooms above grade level."
        )
    with row1_col3:
        bathrooms = st.slider(
            "Bathrooms:", 
            min_value=1.0, max_value=5.0, value=2.0, step=0.5,
            help="Total combined bathrooms (Half baths calculated as 0.5)"
        )
        
    st.write("")
    
    # Operational Action Trigger Button
    if st.button("Calculate Market Value Valuation", type="primary", use_container_width=True):
        # Prepare inference structure
        input_payload = pd.DataFrame([{
            'GrLivArea': sqft,
            'BedroomAbvGr': bedrooms,
            'TotalBathrooms': bathrooms
        }])
        
        # Calculate
        raw_prediction = model.predict(input_payload)[0]
        final_price = max(0.0, raw_prediction) # Baseline floor
        
        # Display Results Section
        st.write("---")
        st.subheader("Valuation Estimation Output")
        
        res_col1, res_col2 = st.columns([2, 1])
        with res_col1:
            st.metric(
                label="Estimated Fair Market Asset Value (USD)", 
                value=f"${final_price:,.2f}",
                delta=f"${final_price/sqft:.2f} / Sq Ft Base Value"
            )
            st.toast("Valuation processed successfully.", icon="✅")
            
        with res_col2:
            # Quick summary text block
            st.markdown(f"""
            **Property Profile Summary:**
            * **Footprint Index:** {sqft:,} sq ft
            * **Layout Vector:** {bedrooms} Bed | {bathrooms} Bath
            """)

# =====================================================================
# TAB 2: ANALYTICS & EXPLAINABILITY INSIGHTS
# =====================================================================
with tab_analytics:
    st.subheader("📊 Regression Diagnostics & Performance Graphs")
    st.markdown("Understand the mathematical reasoning behind the property valuation predictions.")
    
    # Row 1: KPI Scoreboard Metrics Cards
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(
            label="R² Variance Explanation Score", 
            value=f"{metrics['r2']:.2%}", 
            help="Indicates the percentage of volatility in pricing accounted for by these 3 variables."
        )
    with kpi2:
        st.metric(label="Mean Absolute Error (MAE)", value=f"${metrics['mae']:,.2f}")
    with kpi3:
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"${metrics['rmse']:,.2f}")
        
    st.write("---")
    
    # Row 2: Mathematical Reasoning Explainer (Coefficients)
    st.subheader("🧮 How the Model Calculates the Price (Mathematical Weights)")
    st.markdown(
        f"The pricing engine relies on a base calculation starting at a baseline valuation intercept of "
        f"**${model.intercept_:,.2f}**. Every feature added alters that starting asset price:"
    )
    
    coef_cols = st.columns(3)
    feature_names = ["Square Footage (Per Sq Ft)", "Per Bedroom", "Per Bathroom"]
    for idx, col in enumerate(coef_cols):
        with col:
            weight = model.coef_[idx]
            direction = "Increases" if weight >= 0 else "Decreases"
            st.markdown(f"""
            <div style="background-color:#f8f9fa; padding:24px; margin: 16px 0px; border-radius:12px; border-left: 6px solid #0066cc; box-shadow: 0px 4px 6px rgba(0,0,0,0.06);">
                <h5 style="margin-top:0; color:#495057; font-size:1.1em;">{feature_names[idx]}</h5>
                <h2 style="color:#0066cc; margin:10px 0;">${abs(weight):,.2f}</h2>
                <p style="font-size:0.95em; color:#6c757d; margin-bottom:0;">{direction} price per unit added.</p>
            </div>
            """, unsafe_allow_html=True)
            
    st.write("---")
    
    # Row 3: Analytical Structural Graphs Visualization
    st.subheader("📈 Statistical Model Charts")
    graph_col1, graph_col2 = st.columns(2)
    
    with graph_col1:
        st.markdown("**Feature Inter-Correlation Grid Matrix**")
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(clean_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax1, cbar=False)
        plt.title("Correlation Strengths (Closer to 1.0 = Stronger Impact)", fontsize=10)
        st.pyplot(fig1)
        st.caption("Notice that Square Footage (0.71) and Bathrooms (0.60) show strong visual correlations to SalePrice.")
        
    with graph_col2:
        st.markdown("**Error Distribution: Predictions vs. Ground Truth**")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        ax2.scatter(y_test, y_pred, alpha=0.5, color='#0066cc', edgecolors='w', s=40)
        # 45-degree reference standard line
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#e63946', linestyle='--', linewidth=2)
        ax2.set_xlabel('Actual Real-World Market Prices ($)', fontsize=9)
        ax2.set_ylabel('Model Predicted Prices ($)', fontsize=9)
        plt.title("Valuation Residual Scatter Alignment Plot", fontsize=10)
        st.pyplot(fig2)
        st.caption("Properties lining up precisely along the red dashed line denote zero-error predictions.")