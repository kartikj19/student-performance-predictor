# Student Performance Predictor

A regression project that estimates a student's final mathematics grade from study habits, school information, family background, and lifestyle factors.

## What this project demonstrates

- Loading and inspecting a structured dataset with Pandas
- Separating numerical and categorical features
- Handling missing values and one-hot encoding categorical data
- Training a Random Forest regression model
- Evaluating performance with mean absolute error and R2 score
- Saving a reusable trained model

## Dataset

The repository includes the Mathematics course data from the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance). `G3` is the final grade, measured from 0 to 20. Earlier grades (`G1` and `G2`) are excluded to make the prediction task less dependent on previous exam results.

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python src/train.py
```

The script creates `models/student_grade_model.joblib` and `results/metrics.json`.

## Resume description

Built a Python and Scikit-learn regression model to estimate student final grades. Cleaned and encoded structured data, trained a Random Forest model, and evaluated performance using MAE and R2 score.
