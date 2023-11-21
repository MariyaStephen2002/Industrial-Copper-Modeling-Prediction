import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
@st.cache
def load_data():
    df = pd.read_csv("C:\\Users\\stephen\\Downloads\\Copper_final.csv")
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    return df

# Train models
@st.cache(allow_output_mutation=True)
def train_models(df):
    x = df.drop(['selling_price'], axis=1)
    y = df['selling_price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    
    model_price = RandomForestRegressor(n_estimators=40)
    model_price.fit(x_train, y_train)

    a = df.drop(['status'], axis=1)
    b = df['status']
    a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.20)
    
    model_status = RandomForestClassifier(n_estimators=10)
    model_status.fit(a_train, b_train)

    return model_price, model_status

# Main code
main_bg = "rgb(240, 240, 240)"
main_fg = "#000000"

# Set page config
st.set_page_config(
    page_title="Industrial Copper Modelling",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply styling
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background-color: {main_bg};
            color: {main_fg};
        }}
        .sidebar .sidebar-content {{
            background-color: {main_bg};
        }}
        .streamlit-button {{
            color: {main_fg};
            background-color: #4CAF50;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load data
df = load_data()

# Train models
model_price, model_status = train_models(df)

# Sidebar with tabs
selected_tab = st.sidebar.radio("Select a tab", ["Copper Price Prediction", "Status Prediction"])

# Main content
if selected_tab == "Copper Price Prediction":
    # Copper Price Prediction
    st.header("Copper Price Prediction")

    # User input for copper price prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        customer = st.selectbox("Customer ID", df["customer"].unique())
        thickness = st.number_input("Thickness of Copper", min_value=0.0)
        product_ref = st.selectbox("Product Reference", df["product_ref"].unique())
    with col2:
        country = st.selectbox("Country Code", df["country"].unique())
        quantity_tons = st.number_input("Quantity in Tons", min_value=0.0)
        status = st.radio("Status", ("Won", "Lost"))
        nday = st.number_input("Expected Number of Days for Delivery", min_value=0)
    with col3:
        application = st.selectbox("Application Code", df["application"].unique())
        width = st.number_input("Width of Copper", min_value=0.0)
        item_type = st.selectbox("Item Type", df["item type"].unique())

    # Prediction button
    submit = st.button("Predict Copper Price")
    if submit:
        data = np.array([[quantity_tons, customer, country, 1.0 if status == "Won" else 0.0, item_type, application, thickness, width, product_ref, nday]])
        pred_r = model_price.predict(data)
        st.success(f"The predicted selling price is: {pred_r[0]}")

if selected_tab == "Status Prediction":
    # Status Prediction
    st.header("Status Prediction")

    # User input for status prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_ = st.selectbox("Customer ID", df["customer"].unique())
        thickness_ = st.number_input("Thickness of Copper", min_value=0.0)
        product_ref_ = st.selectbox("Product Reference", df["product_ref"].unique())
    with col2:
        country_ = st.selectbox("Country Code", df["country"].unique())
        quantity_tons_ = st.number_input("Quantity in Tons", min_value=0.0)
        selling_price_ = st.number_input("Selling Price of Copper", min_value=0.0)
        nday_ = st.number_input("Expected Number of Days for Delivery", min_value=0)
    with col3:
        application_ = st.selectbox("Application Code", df["application"].unique())
        width_ = st.number_input("Width of Copper", min_value=0.0)
        item_type_ = st.selectbox("Item Type", df["item type"].unique())

    # Prediction button
    submit_ = st.button("Predict Status")
    if submit_:
        data1 = np.array([[quantity_tons_, customer_, country_, item_type_, application_, thickness_, width_, product_ref_, selling_price_, nday_]])
        pred_c = model_status.predict(data1)
        status_message = "The status is Won" if pred_c[0] == 1 else "The status is Lost"
        st.success(status_message)
 
 # streamlit run "e:/visual studio/Copper vi.py"
