
# ==============================================================================
# 1. IMPORTS & PAGE CONFIGURATION
# ==============================================================================
import os
import streamlit as st
import pandas as pd
import joblib

# Configure browser tab title and wide dashboard layout (must be the first st.* call)
st.set_page_config(page_title="Tourism Package Prediction", layout="wide")


# ==============================================================================
# 2. MODEL LOADING
# ==============================================================================
# Securely resolve path relative to this script's location (essential for Streamlit Cloud)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_model.joblib")
model = joblib.load(model_path)


# ==============================================================================
# 3. APPLICATION HEADER & INPUT FORM
# ==============================================================================
st.title("Tourism Package Prediction")
st.write("Enter customer and sales interaction details below to predict travel package purchase likelihood.")

# Group inputs inside a form so predictions run only when "Predict" is clicked
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    # --------------------------------------------------------------------------
    # 3.1 Left Column: Customer Demographic & Financial Profile
    # --------------------------------------------------------------------------
    with col1:
        st.subheader("Customer Profile")
        Age = st.number_input("Age", min_value=18, max_value=70, value=36, step=1)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
        Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=23500.0, step=500.0)
        CityTier = st.selectbox("City Tier", [1, 2, 3])
        Passport = st.selectbox("Has Passport?", ["Yes", "No"])
        OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

    # --------------------------------------------------------------------------
    # 3.2 Right Column: Trip Preferences & Sales Pitch Metrics
    # --------------------------------------------------------------------------
    with col2:
        st.subheader("Trip & Sales Interaction")
        TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
        NumberOfPersonVisiting = st.number_input("Total Persons Visiting", min_value=1, max_value=10, value=3, step=1)
        NumberOfChildrenVisiting = st.selectbox("Children Visiting (under age 5)", [0, 1, 2, 3], index=1)
        NumberOfTrips = st.number_input("Annual Number of Trips", min_value=1, max_value=22, value=3, step=1)
        DurationOfPitch = st.number_input("Duration of Pitch (mins)", min_value=5, max_value=130, value=15, step=1)
        NumberOfFollowups = st.selectbox("Number of Follow-ups", [1, 2, 3, 4, 5, 6], index=2)
        PitchSatisfactionScore = st.radio("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=2, horizontal=True)

    submitted = st.form_submit_button("Predict")


# ==============================================================================
# 4. INPUT VALIDATION & INFERENCE
# ==============================================================================
# Decision threshold aligned with the evaluation cutoff used in train.py
classification_threshold = 0.5

if submitted:
    # --------------------------------------------------------------------------
    # 4.1 Input Guardrails (Missing Values & Domain Logic)
    # --------------------------------------------------------------------------
    # Guard 1: Ensure no numeric field was cleared and submitted as None
    required_numbers = {
        "Age": Age,
        "Monthly Income": MonthlyIncome,
        "Total Persons Visiting": NumberOfPersonVisiting,
        "Annual Number of Trips": NumberOfTrips,
        "Duration of Pitch": DurationOfPitch,
    }
    missing_fields = [name for name, val in required_numbers.items() if val is None]

    # Guard 2: Ensure at least one adult is included when children under 5 are visiting
    if missing_fields:
        st.error(f"Please fill in all required numeric fields: {', '.join(missing_fields)}.")
    elif NumberOfChildrenVisiting >= NumberOfPersonVisiting:
        st.error(
            f"Invalid party size: Children visiting ({NumberOfChildrenVisiting}) must be fewer than "
            f"total persons visiting ({NumberOfPersonVisiting}), as at least one adult is required."
        )
    else:
        # ----------------------------------------------------------------------
        # 4.2 Feature Formatting & Model Prediction
        # ----------------------------------------------------------------------
        # Assemble single-row DataFrame matching the training feature schema
        input_data = pd.DataFrame([{
            "Age": Age,
            "TypeofContact": TypeofContact,
            "CityTier": CityTier,
            "DurationOfPitch": DurationOfPitch,
            "Occupation": Occupation,
            "Gender": Gender,
            "NumberOfPersonVisiting": NumberOfPersonVisiting,
            "NumberOfFollowups": NumberOfFollowups,
            "ProductPitched": ProductPitched,
            "PreferredPropertyStar": PreferredPropertyStar,
            "MaritalStatus": MaritalStatus,
            "NumberOfTrips": NumberOfTrips,
            "Passport": 1 if Passport == "Yes" else 0,
            "PitchSatisfactionScore": PitchSatisfactionScore,
            "OwnCar": 1 if OwnCar == "Yes" else 0,
            "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
            "Designation": Designation,
            "MonthlyIncome": MonthlyIncome,
        }])

        # Predict positive class probability and apply threshold
        prob = model.predict_proba(input_data)[0, 1]
        pred = int(prob >= classification_threshold)

        # Display formatted prediction banner
        if pred == 1:
            st.success(f"Prediction: Customer **will purchase** the travel package (Probability: {prob:.1%})")
        else:
            st.info(f"Prediction: Customer **is unlikely to purchase** (Probability: {prob:.1%})")
