import streamlit as st
import google.generativeai as genai
import pandas as pd
from PIL import Image

# ---------------------------------------------------------
# 1. Page Configuration (Title, Layout, & Styling)
# ---------------------------------------------------------
st.set_page_config(
    page_title="KisanGrade AI",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 KisanGrade AI")
st.caption("Instant Produce Quality Grading & Mandi Price Intelligence")

# ---------------------------------------------------------
# 2. Sidebar Setup (API Key Input)
# ---------------------------------------------------------
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Get a free key from aistudio.google.com")

st.sidebar.markdown("---")
st.sidebar.markdown("### How to use:")
st.sidebar.markdown("1. Enter your Gemini API Key above.")
st.sidebar.markdown("2. Select your crop type.")
st.sidebar.markdown("3. Upload or take a picture of your produce.")
st.sidebar.markdown("4. Tap **Analyze Quality & Find Prices**.")

# ---------------------------------------------------------
# 3. Sample Mandi Dataset (Pandas DataFrame)
# ---------------------------------------------------------
@st.cache_data
def load_mandi_data():
    data = {
        "Commodity": ["Tomatoes", "Tomatoes", "Tomatoes", "Potatoes", "Potatoes", "Potatoes", "Onions", "Onions", "Onions"],
        "Mandi Name": ["Central District Mandi", "Regional APMC Market", "Local Village Market", "Central District Mandi", "Regional APMC Market", "Local Village Market", "Central District Mandi", "Regional APMC Market", "Local Village Market"],
        "Distance (km)": [12, 35, 4, 12, 35, 4, 12, 35, 4],
        "Grade A Rate (₹/kg)": [32, 38, 28, 24, 29, 20, 26, 31, 22],
        "Grade B Rate (₹/kg)": [24, 28, 20, 18, 22, 15, 19, 23, 16],
        "Grade C Rate (₹/kg)": [15, 18, 12, 11, 14, 9, 11, 14, 9]
    }
    return pd.DataFrame(data)

mandi_df = load_mandi_data()

# ---------------------------------------------------------
# 4. Main App Interface
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Produce Details")
    commodity = st.selectbox("Select Crop Type", ["Tomatoes", "Potatoes", "Onions"])
    
    input_method = st.radio("Image Source", ["Upload Image", "Use Camera"], horizontal=True)
    
    image_data = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload a clear photo of your produce", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image_data = Image.open(uploaded_file)
            st.image(image_data, caption="Uploaded Image Preview", use_column_width=True)
    else:
        camera_file = st.camera_input("Take a photo of your produce")
        if camera_file:
            image_data = Image.open(camera_file)
            st.image(image_data, caption="Captured Image Preview", use_column_width=True)

    analyze_btn = st.button("🔍 Analyze Quality & Find Prices", type="primary", use_container_width=True)

# ---------------------------------------------------------
# 5. AI Vision Processing & Results Rendering
# ---------------------------------------------------------
with col2:
    st.subheader("2. AI Analysis & Market Intelligence")
    
    if analyze_btn:
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar to proceed.")
        elif image_data is None:
            st.warning("Please upload or capture an image of your produce first.")
        else:
            with st.spinner("Analyzing image features with Gemini 1.5 Flash..."):
                try:
                    # Configure Gemini API
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # Construct Prompt
                    prompt = f"""
                    You are an expert agricultural quality inspector for {commodity}.
                    Analyze the provided image and give a detailed inspection report:
                    1. Assign a clear trade quality grade: Grade A (Premium), Grade B (Standard), or Grade C (Discounted/Damaged).
                    2. Evaluate Surface Color & Ripeness Stage.
                    3. Evaluate Size Uniformity & Physical Defects (spots, bruises, skin breaks).
                    4. Give a 1-sentence recommendation on whether pre-sorting will increase net earnings.

                    Format your response clearly using markdown with bullet points and bold headers.
                    """
                    
                    # Get AI Response
                    response = model.generate_content([prompt, image_data])
                    
                    # Display AI Quality Report
                    st.success("Quality Inspection Complete!")
                    st.markdown(response.text)
                    
                    st.markdown("---")
                    st.subheader("📊 Nearby Mandi Price Comparison")
                    
                    # Filter and Display Mandi Prices
                    filtered_df = mandi_df[mandi_df["Commodity"] == commodity]
                    st.dataframe(filtered_df, hide_index=True, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
    else:
        st.info("Upload/capture an image and click the button above to view quality grading and market price results.")
