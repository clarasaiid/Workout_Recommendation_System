import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Handle scikit-learn version compatibility
# Fix for _RemainderColsList attribute error when loading models saved with older scikit-learn versions
import sys
import sklearn.compose._column_transformer as ct_module

# Ensure _RemainderColsList exists for compatibility with models saved in older scikit-learn versions
if not hasattr(ct_module, '_RemainderColsList'):
    class _RemainderColsList(list):
        """Compatibility class for older scikit-learn models"""
        pass
    # Add to module
    ct_module._RemainderColsList = _RemainderColsList
    # Also add to sys.modules for pickle to find it
    if 'sklearn.compose._column_transformer' in sys.modules:
        sys.modules['sklearn.compose._column_transformer']._RemainderColsList = _RemainderColsList

# Suppress version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Page configuration
st.set_page_config(
    page_title="Workout Recommender",
    page_icon="💪",
    layout="wide"
)

st.title("💪 Multi-Label Workout Recommender")
st.markdown("Enter your details below to get personalized workout recommendations.")

# Load model and artifacts with caching
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        return joblib.load("model.joblib")
    except AttributeError as e:
        if "_RemainderColsList" in str(e):
            st.error("""
            **Model Compatibility Error**
            
            The model was saved with an older version of scikit-learn (1.6.1) and cannot be loaded 
            with the current version (1.8.0).
            
            **Solution:** Please install scikit-learn version 1.6.1:
            ```bash
            pip3 install scikit-learn==1.6.1
            ```
            """)
        raise
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

@st.cache_resource
def load_columns():
    """Load feature column names"""
    return joblib.load("X_columns.joblib")

@st.cache_resource
def load_labels():
    """Load workout label names"""
    return joblib.load("workout_labels.joblib")

# Load artifacts
try:
    model = load_model()
    X_columns = load_columns()
    workout_labels = load_labels()
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.info("Make sure model.joblib, X_columns.joblib, and workout_labels.joblib are in the same directory as app.py")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Create form for user inputs
with st.form("workout_input_form"):
    st.subheader("Personal Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1)
        
        # Calculate BMI automatically: BMI = weight (kg) / (height (m))^2
        if height > 0:
            bmi = weight / ((height / 100) ** 2)
            st.metric("BMI", f"{bmi:.2f}")
        else:
            bmi = 0
    
    with col2:
        sex = st.selectbox("Sex", options=["Male", "Female"])
        hypertension = st.selectbox("Hypertension", options=["Yes", "No"])
        diabetes = st.selectbox("Diabetes", options=["Yes", "No"])
        # Level refers to BMI category - auto-calculate from BMI
        # Calculate BMI here too for the category selection (bmi from col1 may not be available yet)
        bmi_for_cat = weight / ((height / 100) ** 2) if height > 0 else 24.0
        if bmi_for_cat < 18.5:
            default_bmi_cat = "Underweight"
        elif bmi_for_cat < 25:
            default_bmi_cat = "Normal"
        elif bmi_for_cat < 30:
            default_bmi_cat = "Overweight"
        else:
            default_bmi_cat = "Obuse"
        
        bmi_category = st.selectbox("BMI Category (Level)", 
                                    options=["Underweight", "Normal", "Overweight", "Obuse"],
                                    index=["Underweight", "Normal", "Overweight", "Obuse"].index(default_bmi_cat),
                                    help="Auto-selected based on BMI. Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obuse (≥30)")
        fitness_goal = st.selectbox("Fitness Goal", options=["Weight Loss", "Weight Gain"])
        fitness_type = st.selectbox("Fitness Type", options=["Cardio Fitness", "Muscular Fitness"])
        equipment = st.selectbox("Equipment", 
                                 options=["Dumbbells and barbells",
                                         "Dumbbells, barbells and Blood glucose monitor",
                                         "Ellipticals, Indoor Rowers,Treadmills, Rowing machine",
                                         "Ellipticals, Indoor Rowers,Treadmills, and Rowing machine",
                                         "Equipment Required",
                                         "Kettlebell, Dumbbells, Yoga Mat",
                                         "Kettlebell, Dumbbells, Yoga Mat, Treadmill",
                                         "Light athletic shoes, resistance bands, and light dumbbells.",
                                         "Light athletic shoes, resistance bands, light dumbbells and a Blood glucose monitor."])
    
    submitted = st.form_submit_button("Get Workout Recommendations", type="primary")

if submitted:
    # The model has a preprocessing pipeline (ColumnTransformer with OneHotEncoder)
    # so we need to pass raw categorical values, not one-hot encoded columns
    # Create DataFrame with raw column names matching the original training data
    # Note: The model expects these columns in this order: ['Sex', 'Age', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type', 'Equipment']
    X_row = pd.DataFrame({
        "Sex": [str(sex)],
        "Age": [float(age)],
        "Hypertension": [str(hypertension)],
        "Diabetes": [str(diabetes)],
        "BMI": [float(bmi)],
        "Level": [str(bmi_category)],
        "Fitness Goal": [str(fitness_goal)],
        "Fitness Type": [str(fitness_type)],
        "Equipment": [str(equipment)]
    })
    
    # Ensure columns are in the correct order as expected by the model
    X_row = X_row[X_columns]
    
    # Make predictions
    st.subheader("🎯 Workout Recommendations")
    
    # Check if predict_proba is available
    if hasattr(model, "predict_proba"):
        try:
            # Get probabilities for each label
            probabilities = model.predict_proba(X_row)
            
            # Extract probability of class 1 for each label
            prob_scores = []
            for i, label in enumerate(workout_labels):
                if isinstance(probabilities[i], np.ndarray) and len(probabilities[i][0]) > 1:
                    # Binary classification: probability of class 1
                    prob = probabilities[i][0][1]
                else:
                    # Fallback: use the probability array directly
                    prob = probabilities[i][0] if isinstance(probabilities[i][0], (int, float)) else probabilities[i][0][-1]
                prob_scores.append((label, prob))
            
            # Sort by probability (descending)
            prob_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Top-K selector
            max_k = min(len(workout_labels), 9)
            k = st.slider("Select number of top recommendations to show", 
                         min_value=1, max_value=max_k, value=min(5, max_k), step=1)
            
            # Display top-K recommendations
            st.markdown(f"### Top {k} Recommended Workouts")
            
            for i, (label, prob) in enumerate(prob_scores[:k], 1):
                confidence_pct = prob * 100
                st.markdown(f"**{i}. {label}**")
                st.progress(prob, text=f"Confidence: {confidence_pct:.2f}%")
                st.markdown("---")
            
            # Show all predictions in a table
            with st.expander("View all workout predictions"):
                results_df = pd.DataFrame({
                    "Workout": [label for label, _ in prob_scores],
                    "Confidence Score": [f"{prob:.4f}" for _, prob in prob_scores]
                })
                st.dataframe(results_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error during prediction with probabilities: {e}")
            st.info("Falling back to binary predictions...")
            predictions = model.predict(X_row)
            
            # Display predictions
            predicted_labels = [workout_labels[i] for i, pred in enumerate(predictions[0]) if pred == 1]
            
            if predicted_labels:
                st.markdown("### Recommended Workouts")
                for i, label in enumerate(predicted_labels, 1):
                    st.markdown(f"**{i}. {label}**")
            else:
                st.info("No workouts recommended based on your inputs.")
    else:
        # Fallback to binary predictions
        predictions = model.predict(X_row)
        
        # Display predictions
        predicted_labels = [workout_labels[i] for i, pred in enumerate(predictions[0]) if pred == 1]
        
        if predicted_labels:
            st.markdown("### Recommended Workouts")
            for i, label in enumerate(predicted_labels, 1):
                st.markdown(f"**{i}. {label}**")
        else:
            st.info("No workouts recommended based on your inputs.")
    
    # Show input row in expander
    with st.expander("Show input row (X)"):
        st.dataframe(X_row, use_container_width=True)
        st.markdown("**Note:** This shows the feature vector used for prediction.")

# Instructions sidebar
with st.sidebar:
    st.header("📋 Setup Instructions")
    st.markdown("""
    ### Installation
    
    Install required packages:
    ```bash
    pip3 install streamlit joblib scikit-learn pandas numpy
    ```
    
    ### Running the App
    
    Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    
    ### Required Files
    
    Make sure these files are in the same directory as `app.py`:
    - `model.joblib`
    - `X_columns.joblib`
    - `workout_labels.joblib`
    """)
    
    st.markdown("---")
    st.markdown("**Note:** The app will cache the loaded model and artifacts for better performance.")

