# Student Performance Prediction System

## Quick start (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed by Streamlit (usually http://localhost:8501).

## What is included
- `app.py`: Streamlit application with Prediction, Dataset insights, and Model evaluation tabs.
- `data/student_performance.csv`: 360 synthetic demonstration records.
- `requirements.txt`: Python dependencies.
- `PROJECT_REPORT.docx`: professional capstone report.
- `PRESENTATION.pptx`: capstone presentation.

## Dataset and results disclaimer
The included CSV is synthetic demonstration data, not real student records. Any displayed metrics (including hold-out accuracy) describe only a random split of this synthetic data and must not be presented as evidence of real-world performance. For an academic final evaluation, replace it with a suitable dataset whose source and licence are documented, then rerun evaluation.

## Model
A scikit-learn Pipeline applies median imputation and standard scaling, then trains a Random Forest classifier. A stratified 80/20 hold-out split is used. Recommendations are rule-based examples from input thresholds.

## Ethical use
This is an educational prototype. Predictions should support discussion and study planning only; they should not be used to make high-stakes decisions about students.
