import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction System")
st.caption("A machine-learning capstone project for academic analytics and improvement insights.")

DATA_PATH = Path(__file__).parent / "data" / "student_performance.csv"
FEATURES = ["attendance", "study_hours", "internal_marks", "participation", "previous_score"]
TARGET = "performance_level"
LABELS = ["Low", "Average", "High"]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model(data):
    X = data[FEATURES].copy()
    y = data[TARGET].astype(str)
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    preprocessor = ColumnTransformer([("numeric", numeric_pipe, FEATURES)])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, random_state=42, class_weight="balanced"
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, y_test, predictions

data = load_data()
model, y_test, y_pred = train_model(data)

with st.sidebar:
    st.header("Student details")
    attendance = st.slider("Attendance (%)", 0, 100, 80)
    study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
    internal_marks = st.slider("Internal marks (%)", 0, 100, 65)
    participation = st.slider("Class participation (%)", 0, 100, 70)
    previous_score = st.slider("Previous academic score (%)", 0, 100, 65)

with st.sidebar:
    st.caption("Adjust the inputs to explore a prediction. The model is a demonstration, not a validated student assessment.")

tab1, tab2, tab3 = st.tabs(["Prediction", "Dataset insights", "Model evaluation"])

with tab1:
    st.subheader("Student performance estimate")
    input_df = pd.DataFrame([{
        "attendance": attendance,
        "study_hours": study_hours,
        "internal_marks": internal_marks,
        "participation": participation,
        "previous_score": previous_score
    }])
    predicted = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = float(max(probabilities)) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted level", predicted)
    c2.metric("Model confidence", f"{confidence:.1f}%")
    c3.metric("Dataset records", f"{len(data):,}")

    st.progress(min(max(confidence / 100, 0), 1))
    st.caption("Confidence is the model's estimated class probability, not a guarantee of a student's outcome.")

    tips = []
    if attendance < 75:
        tips.append("Improve attendance and attend classes consistently.")
    if study_hours < 2:
        tips.append("Try a regular study schedule and gradually increase focused study time.")
    if internal_marks < 60:
        tips.append("Review internal assessment feedback and practise weak topics.")
    if participation < 50:
        tips.append("Ask questions and participate in class or practice sessions.")
    if previous_score < 60:
        tips.append("Revisit foundational topics and seek help in difficult subjects.")
    if not tips:
        tips.append("Maintain your current habits and keep reviewing subject-wise progress.")

    st.markdown("#### Suggested improvement actions")
    for tip in tips:
        st.write("• " + tip)
    st.markdown("#### Inputs used")
    st.dataframe(input_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Explore the academic dataset")
    a, b, c = st.columns(3)
    a.metric("Students", len(data))
    b.metric("Average attendance", f"{data['attendance'].mean():.1f}%")
    c.metric("Average internal marks", f"{data['internal_marks'].mean():.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("**Performance-level distribution**")
        counts = data[TARGET].value_counts().reindex(LABELS, fill_value=0)
        st.bar_chart(counts)
    with right:
        st.markdown("**Average feature values by performance level**")
        means = data.groupby(TARGET)[FEATURES].mean().reindex(LABELS)
        st.bar_chart(means)
    st.markdown("**Dataset preview**")
    st.dataframe(data.head(20), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Model evaluation")
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Hold-out accuracy", f"{accuracy:.1%}")
    st.caption("This score is measured on a random hold-out split of the included synthetic demonstration dataset. It is not evidence of real-world accuracy.")
    st.markdown("**Classification report**")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    st.markdown("**Feature importance**")
    clf = model.named_steps["classifier"]
    importance = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
    st.bar_chart(importance)
    st.info("Replace the demonstration CSV with a suitable permitted academic dataset before making real-world claims.")

st.divider()
st.caption("Educational prototype only. Predictions should support students, not determine grades, access, or opportunities.")
