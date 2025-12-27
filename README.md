# 💪 Multi-Label Workout Recommender

A Streamlit web application that provides personalized workout recommendations using a multi-label machine learning model.

## Features

- **Personalized Recommendations**: Get workout suggestions based on your personal information and fitness goals
- **Multi-Label Classification**: Receive multiple workout recommendations simultaneously
- **Confidence Scores**: See probability scores for each recommended workout
- **Auto-Calculated BMI**: BMI is automatically calculated from height and weight

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Installation

1. Clone this repository:
```bash
git clone https://github.com/clarasaiid/Workout_Recommendation_System.git
cd Workout_Recommendation_System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Locally

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment

### Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select this repository: `clarasaiid/Workout_Recommendation_System`
5. Set the main file path to: `app.py`
6. Click "Deploy"

The app will be automatically deployed and accessible via a public URL.

### Required Files

Make sure these files are in the repository:
- `app.py` - Main Streamlit application
- `model.joblib` - Trained model
- `X_columns.joblib` - Feature column names
- `workout_labels.joblib` - Workout label names
- `requirements.txt` - Python dependencies

## Usage

1. Enter your personal information:
   - Age
   - Height (cm)
   - Weight (kg)
   - Sex
   - Health conditions (Hypertension, Diabetes)
   - BMI Category (auto-calculated)
   - Fitness Goal
   - Fitness Type
   - Equipment available

2. Click "Get Workout Recommendations"

3. View your personalized workout recommendations with confidence scores

## Model Details

- **Model Type**: MultiOutputClassifier with RandomForestClassifier
- **Preprocessing**: ColumnTransformer with StandardScaler for numeric features and OneHotEncoder for categorical features
- **Training**: scikit-learn 1.6.1

## License

This project is open source and available for personal and educational use.

