import streamlit as st
import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv('financial_risk_assessment.csv')  # Ensure the dataset is in the working directory

st.markdown("<h1 style='text-align: center; font-size: 40px; color: #333;'>Risk Rating Prediction</h1>", unsafe_allow_html=True)

# Drop rows where 'Risk Rating' is 'Medium'
df_filtered = df[df['Risk Rating'] != 'Medium']

# Function to assign city value based on the Risk Rating
def get_city_value(city_name):
    city_data = df_filtered[df_filtered['City'] == city_name]
    if (city_data['Risk Rating'] == 'High').all():
        return 1
    else:
        return 0

# Get unique cities for the dropdown
unique_cities = df_filtered['City'].unique()

# Load the trained model
model = joblib.load('best_model.pkl')  # Replace with your model file path

# Apply a professional CSS style
st.markdown(
    """
    <style>
        /* Main background and text color */
        .stApp {
            background-color: #fff;
            color: #fff;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #000;
            font-family: 'Arial', sans-serif;
            font-weight: bold;
        }

        /* Body Text */
        body, p, label, div {
            color: #FFF;
            font-family: 'Arial', sans-serif;
        }

        /* Inputs */
        .stSelectbox, .stTextInput, .stNumberInput, .stSlider {
            font-size: 16px;
           
        }

        /* Buttons */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }

        .stButton > button:hover {
             background-color: #45a049;
        }

        /* Adjusting slider styles */
        .stSlider > div {
            color: #fff;
            

        }

        /* Center the content */
        .stApp {
            margin-left: 10%;
            margin-right: 10%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Create two columns
col1, _, col2 = st.columns([8, 1, 8])

# Gender and Age inputs in the first column
with col1:
    st.markdown("<h5 >Gender</h5>", unsafe_allow_html=True)
    gender = st.selectbox(' ', ['Male', 'Female'], help="Select your gender.")
    gender_female = 1 if gender == 'Female' else 0
    gender_male = 1 if gender == 'Male' else 0

    st.markdown("<h5 >Age</h5>", unsafe_allow_html=True)
    age = st.slider(' ', 0, 100, 1, help="Select Age.")

    st.markdown("<h5 >City</h5>", unsafe_allow_html=True)
    city = st.selectbox(' ', options=unique_cities, format_func=lambda x: x, help="Select your city.")
    city_value = get_city_value(city)

    st.markdown("<h5 >Marital Status</h5>", unsafe_allow_html=True)
    marital_status = st.selectbox(' ', ['Single', 'Married'], help="Select your marital status.")
    marital_status_married = 1 if marital_status == 'Married' else 0
    marital_status_single = 1 if marital_status == 'Single' else 0

   

# Payment History and Employment Status inputs in the second column
with col2:

    st.markdown("<h5 >Employment Status</h5>", unsafe_allow_html=True)
    employment_status = st.selectbox(' ', ['Employed', 'Self-employed', 'Unemployed'], help="Select your employment status.")
    employment_status_employed = 1 if employment_status == 'Employed' else 0
    employment_status_self_employed = 1 if employment_status == 'Self-employed' else 0
    employment_status_unemployed = 1 if employment_status == 'Unemployed' else 0

    st.markdown("<h5 >Income</h5>", unsafe_allow_html=True)
    assets_value = st.number_input(' ', value=50000, help="Enter the total value of your assets.")

    st.markdown("<h5 >Years at Current Job</h5>", unsafe_allow_html=True)
    years_at_current_job = st.slider(' ', 0, 40, 5, help="Select the number of years you've been at your current job.")

    st.markdown("<h5 >Payment History</h5>", unsafe_allow_html=True)
    payment_history_options = ['Bad', 'Poor', 'Good', 'Excellent']
    payment_history_scores = [0, 1, 2, 3]
    payment_history = st.selectbox(' ', options=payment_history_options, format_func=lambda x: x, help="Select your payment history rating.")
    payment_history_score = payment_history_scores[payment_history_options.index(payment_history)]

st.markdown("<h5 >Loan Purpose</h5>", unsafe_allow_html=True)
loan_purpose = st.selectbox(' ', ['Auto', 'Business', 'Personal'], help="Select the purpose of the loan.")
loan_purpose_auto = 1 if loan_purpose == 'Auto' else 0
loan_purpose_business = 1 if loan_purpose == 'Business' else 0
loan_purpose_personal = 1 if loan_purpose == 'Personal' else 0

   

# Loan Amount and Assets Value input across both columns
st.markdown("<h5 >Loan Amount</h5>", unsafe_allow_html=True)
loan_amount = st.number_input(' ', value=10010, help="Enter the amount of the loan.")



# Prepare input data for prediction
input_data = {
    'Payment History': payment_history_score,  # Use the numerical score
    'City': city_value,
    'Marital Status Change': 1,
    'Gender_Female': gender_female,
    'Gender_Male': gender_male,
    'Marital Status_Married': marital_status_married,
    'Marital Status_Single': marital_status_single,
    'Loan Purpose_Auto': loan_purpose_auto,
    'Loan Purpose_Business': loan_purpose_business,
    'Loan Purpose_Personal': loan_purpose_personal,
    'Employment Status_Employed': employment_status_employed,
    'Employment Status_Unemployed': employment_status_unemployed,
    'Loan Amount': loan_amount,
    'Assets Value': assets_value,
    'Years at Current Job': years_at_current_job
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Ensure the input matches the model's expected columns
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns

# Predict button
if st.button('Predict Risk Rating'):
    # Reorder columns to match the model's training data
    input_df = input_df[model.feature_names_in_]
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0]

    # Display the result
    if prediction[0] == 1:  # High Risk
        st.markdown(f"<h3 style='color: red; text-align: center;'>Our Prediction is you are in: High Risk</h3>", unsafe_allow_html=True)

        # Path to your WebP image
        image_path = 'deny.webp'  # Replace with your WebP file path

        # Display the WebP image
        st.image(image_path, caption='High Risk', use_column_width=True)

    else:  # Low Risk
        st.markdown(f"<h3 style='color: green; text-align: center;'>Our Prediction is you are in: Low Risk</h3>", unsafe_allow_html=True)

         # Path to your WebP image
        image_path = 'approved.webp'  # Replace with your WebP file path

        # Display the WebP image
        st.image(image_path, caption='Low Risk', use_column_width=True)
