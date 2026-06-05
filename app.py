import streamlit as st
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)
import pickle

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🎓 Student Performance Prediction System")
st.markdown(
    "Predict whether a student is likely to pass based on academic and personal factors."
)

study_hours = st.number_input(
    "Study Hours per Week",
    min_value=0.0,
    max_value=50.0,
    value=10.0
)

attendance = st.number_input(
    "Attendance Rate",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous_grades = st.number_input(
    "Previous Grades",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

activities = st.selectbox(
    "Participation in Extracurricular Activities",
    ["No", "Yes"]
)

parent_education = st.selectbox(
    "Parent Education Level",
    ["Associate", "Bachelor", "Doctorate", "High School", "Master"]
)

# Convert values
activities_value = 1 if activities == "Yes" else 0

parent_map = {
    "Associate": 0,
    "Bachelor": 1,
    "Doctorate": 2,
    "High School": 3,
    "Master": 4
}

parent_value = parent_map[parent_education]

if st.button("Predict"):

    prediction = model.predict([[
        study_hours,
        attendance,
        previous_grades,
        activities_value,
        parent_value
    ]])

    if prediction[0] == 1:
        st.success("Prediction: Student Will Pass")
    else:
        st.error("Prediction: Student May Fail")