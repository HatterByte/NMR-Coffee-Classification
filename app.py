import streamlit as st
import numpy as np
import pickle

# --- Load Models ---
import joblib
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import streamlit as st

from config import get_sidebar_inputs
from compounds import compound_data
from simulation import (
    calculate_larmor, calculate_capacitance,
    simulate_fid,simulate_fidd, simulate_t1_t2
)
from visualization import (
    plot_fid, plot_spectrum, plot_relaxation
)
from analysis import extract_peak_intensities

st.set_page_config(page_title="Virtual NMR Spectroscope", layout="wide")
st.title("☕ Virtual NMR Spectroscope for Coffee Analysis")

country, B0, gamma, L = get_sidebar_inputs()
f_larmor = calculate_larmor(gamma, B0)
C_pF = calculate_capacitance(f_larmor, L)

st.sidebar.markdown(f"**Larmor Frequency:** {f_larmor/1e6:.2f} MHz")
st.sidebar.markdown(f"**Tuning Capacitance C:** {C_pF:.2f} pF")

country_data = compound_data[country]

N = 1024
t, fid = simulate_fid(N, f_larmor, country_data)
plot_fid(t, fid)

# Spectrum + Peaks
peaks = plot_spectrum(t, fid, f_larmor, country_data)

# Extract features (just intensities)
intensities = [intensity for _, _, intensity in peaks]
X_input = np.array([intensities])  # shape: (1, n_features)


@st.cache_resource
def load_models():
     rf_model = joblib.load("rf_coffee_classifier.pkl")
     svm_model = joblib.load("svm_coffee_classifier.pkl")
     return rf_model, svm_model  # make sure you return the models!

rf_model, svm_model = load_models()
scaler = joblib.load('svm_scaler.pkl')
X_new_scaled = scaler.transform(X_input)
# --- Predict ---
rf_pred = rf_model.predict(X_input)[0]
svm_pred = svm_model.predict(X_new_scaled)[0]

st.subheader("🧠 Model Predictions")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Random Forest Prediction**")
    st.success(rf_pred)

with col2:
    st.markdown("**SVM Prediction**")
    st.info(svm_pred)

# --- Prediction Probabilities ---
rf_probs = rf_model.predict_proba(X_input)[0]
svm_probs = svm_model.predict_proba(X_new_scaled)[0]
labels = rf_model.classes_

st.subheader("📊 Prediction Probabilities")
prob_df = pd.DataFrame({
    "Origin": labels,
    "RF" :rf_probs,
    "SVM":svm_probs
    
})

def highlight_max_custom(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)

    max_svm = df['SVM'].max()
    max_rf = df['RF'].max()

    # Match top prediction box styles
    svm_color = '#1e2f3e'
    rf_color = '#163b2f'

    styles.loc[df['SVM'] == max_svm, 'SVM'] = f'background-color: {svm_color}; color: white;'
    styles.loc[df['RF'] == max_rf, 'RF'] = f'background-color: {rf_color}; color: white;'

    return styles

st.dataframe(
    prob_df.set_index("Origin").style.apply(highlight_max_custom, axis=None),
    use_container_width=True
)

# st.dataframe(prob_df.set_index("Stats").style.highlight_max(axis=1), use_container_width=True)

# --- Load Test Accuracy from Metrics File ---
import json

# @st.cache_data
# def load_metrics():
#     with open("models/model_metrics.json", "r") as f:
#         return json.load(f)

# metrics = load_metrics()

# st.subheader("📈 Model Accuracy on Test Set")
# acc_col1, acc_col2 = st.columns(2)
# with acc_col1:
#     st.metric(label="Random Forest Accuracy", value=f"{metrics['rf_accuracy']*100:.2f}%")
# with acc_col2:
#     st.metric(label="SVM Accuracy", value=f"{metrics['svm_accuracy']*100:.2f}%")
#----------------------------------------------------------------------------
# Add this after your prediction probabilities section
# Add this after your prediction probabilities section


st.subheader("📊 Model Comparison: Confusion Matrices")

# Function to generate test data from your compound data
@st.cache_data
def generate_test_data(compound_data, n_samples_per_country=900):
    X_test = []
    y_test = []
    
    # For each country, generate multiple test samples with variations
    for country, compounds in compound_data.items():
        for _ in range(n_samples_per_country):
            # Simulate an FID and extract peaks
            N = 1024
            t = np.linspace(0, 1, N)
            f_larmor = calculate_larmor(42.576, 1.5)  # Using typical values
            
            # Simulate FID with some variation
            _, fid = simulate_fidd(N, f_larmor, compounds)
            
            # Get spectrum and extract peak intensities
            spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid)))
            freq = np.fft.fftshift(np.fft.fftfreq(N, d=(t[1] - t[0])))
            
            # Define a fixed ppm range
            ppm_min, ppm_max = 0.5, 10
            ppm_axis = np.linspace(ppm_max, ppm_min, N)
            
            # Convert fixed ppm axis to frequency in Hz
            freq_ppm = f_larmor * (ppm_axis - 4.7) / 1e6
            
            # Interpolate spectrum values
            spectrum_interp = np.interp(freq_ppm, freq, spectrum)
            
            # Extract peaks
            peaks = extract_peak_intensities(spectrum_interp, ppm_axis, compounds)
            
            # Get just the intensities
            intensities = [intensity for _, _, intensity in peaks]
            
            X_test.append(intensities)
            y_test.append(country)
    
    return np.array(X_test), np.array(y_test)

# Generate test data
X_test, y_test = generate_test_data(compound_data)
X_new_scaled = scaler.transform(X_test)

# Make predictions
rf_preds = rf_model.predict(X_test)
svm_preds = svm_model.predict(X_new_scaled)

# Get unique classes (sorted to ensure consistency)
classes = sorted(np.unique(y_test))

# Compute confusion matrices
rf_cm = confusion_matrix(y_test, rf_preds, labels=classes)
svm_cm = confusion_matrix(y_test, svm_preds, labels=classes)

# Visualize the confusion matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot RF confusion matrix
disp1 = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=classes)
disp1.plot(ax=ax1, cmap='Blues', values_format='d')
ax1.set_title('Random Forest Confusion Matrix')

# Plot SVM confusion matrix
disp2 = ConfusionMatrixDisplay(confusion_matrix=svm_cm, display_labels=classes)
disp2.plot(ax=ax2, cmap='Greens', values_format='d')
ax2.set_title('SVM Confusion Matrix')

plt.tight_layout()
st.pyplot(fig)

# Calculate and display accuracy metrics
rf_accuracy = np.sum(np.diag(rf_cm)) / np.sum(rf_cm)
svm_accuracy = np.sum(np.diag(svm_cm)) / np.sum(svm_cm)

st.markdown("### Model Accuracy Comparison")
col1, col2 = st.columns(2)
with col1:
    st.metric("Random Forest Accuracy", f"{rf_accuracy:.2%}")
with col2:
    st.metric("SVM Accuracy", f"{svm_accuracy:.2%}")

# Calculate class-specific metrics
class_metrics = []

for i, class_name in enumerate(classes):
    # True positives, false positives, false negatives
    rf_tp = rf_cm[i, i]
    rf_fp = np.sum(rf_cm[:, i]) - rf_tp
    rf_fn = np.sum(rf_cm[i, :]) - rf_tp
    
    svm_tp = svm_cm[i, i]
    svm_fp = np.sum(svm_cm[:, i]) - svm_tp
    svm_fn = np.sum(svm_cm[i, :]) - svm_tp
    
    # Calculate precision and recall
    rf_precision = rf_tp / (rf_tp + rf_fp) if (rf_tp + rf_fp) > 0 else 0
    rf_recall = rf_tp / (rf_tp + rf_fn) if (rf_tp + rf_fn) > 0 else 0
    
    svm_precision = svm_tp / (svm_tp + svm_fp) if (svm_tp + svm_fp) > 0 else 0
    svm_recall = svm_tp / (svm_tp + svm_fn) if (svm_tp + svm_fn) > 0 else 0
    
    class_metrics.append({
        "Class": class_name,
        "RF Precision": f"{rf_precision:.2f}",
        "RF Recall": f"{rf_recall:.2f}",
        "SVM Precision": f"{svm_precision:.2f}",
        "SVM Recall": f"{svm_recall:.2f}"
    })

# Display class-specific metrics
st.markdown("### Class-Specific Performance")
st.table(pd.DataFrame(class_metrics))

# Add interpretation of the differences
st.markdown("### Interpreting Model Differences")

# Find where models differ most
rf_class_acc = np.diag(rf_cm) / np.sum(rf_cm, axis=1)
svm_class_acc = np.diag(svm_cm) / np.sum(svm_cm, axis=1)
class_diff = rf_class_acc - svm_class_acc

# Handle division by zero or NaN
rf_class_acc = np.nan_to_num(rf_class_acc)
svm_class_acc = np.nan_to_num(svm_class_acc)
class_diff = np.nan_to_num(class_diff)

best_rf_idx = np.argmax(class_diff)
best_svm_idx = np.argmin(class_diff)

best_rf_class = classes[best_rf_idx]
best_svm_class = classes[best_svm_idx]

# Find common misclassifications
rf_misclass = []
svm_misclass = []

for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j and rf_cm[i,j] > 0:
            rf_misclass.append(f"{classes[i]} as {classes[j]} ({rf_cm[i,j]} times)")
        if i != j and svm_cm[i,j] > 0:
            svm_misclass.append(f"{classes[i]} as {classes[j]} ({svm_cm[i,j]} times)")

st.markdown(f"""
**Key Differences in Model Performance:**

1. **Overall Performance:** 
   - Random Forest Accuracy: {rf_accuracy:.2%}
   - SVM Accuracy: {svm_accuracy:.2%}
   - Difference: {abs(rf_accuracy - svm_accuracy):.2%} in favor of {'Random Forest' if rf_accuracy > svm_accuracy else 'SVM'}

2. **Class-Specific Performance:**
   - Random Forest performs better on **{best_rf_class}** samples
   - SVM performs better on **{best_svm_class}** samples

3. **Common Misclassifications:**
   - **Random Forest:** {', '.join(rf_misclass[:3])}
   - **SVM:** {', '.join(svm_misclass[:3])}

4. **How the Models Differ:**
   - **Random Forest:** Creates multiple decision trees based on different subsets of features (peaks) and samples. Each tree "votes" on the classification, making it good at handling complex non-linear relationships between peaks.
   
   - **SVM:** Finds an optimal hyperplane that separates coffee samples in high-dimensional space. Uses kernels to handle non-linearity, but processes all features simultaneously rather than in subsets.
   
   These fundamental differences explain why they perform differently on certain coffee origins.
""")

# Add a visual explanation of model differences
st.markdown("### 📈 Visual Model Comparison")

model_comparison = pd.DataFrame({
    "Characteristic": [
        "Decision Process", 
        "Feature Handling",
        "Robustness to Noise",
        "Interpretability",
        "Performance on Small Dataset"
    ],
    "Random Forest": [
        "Multiple decision trees voting",
        "Selects important features automatically",
        "High - averaging reduces impact of noise",
        "Medium - can extract feature importance",
        "Good - doesn't require large samples"
    ],
    "SVM": [
        "Hyperplane separation",
        "Uses all features with weights",
        "Medium - sensitive to parameter tuning",
        "Low - black box decision boundary",
        "Medium - needs good representative samples"
    ]
})

st.table(model_comparison)