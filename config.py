import streamlit as st

GYRO_CHOICES = {
    "Hydrogen-1 (¹H)": 42.577,   # MHz/T
    "Carbon-13 (¹³C)": 10.71     # MHz/T
}

B0_OPTIONS = {
    "Low Field (0.3 T ~ 12.8 MHz)": 0.3,
    "Medium Field (0.47 T ~ 20 MHz)": 0.47,
    "Standard Lab (0.7 T ~ 30 MHz)": 0.7,
    "Clinical (1.5 T ~ 64 MHz)": 1.5
}
options_list = list(B0_OPTIONS.keys())
default_index = options_list.index("Standard Lab (0.7 T ~ 30 MHz)")

def get_sidebar_inputs():
    st.sidebar.title("Spectrometer Settings")

    country = st.sidebar.selectbox("Select Coffee Origin", ["Brazil", "Colombia", "Ethiopia"])

    B0_label = st.sidebar.selectbox("Magnetic Field Strength", list(B0_OPTIONS.keys()), index=default_index)
    B0 = B0_OPTIONS[B0_label]

    gamma_label = st.sidebar.selectbox("Gyromagnetic Ratio", list(GYRO_CHOICES.keys()))
    gamma = GYRO_CHOICES[gamma_label]

    L = st.sidebar.number_input("Inductance L (μH)", value=10.0)

    return country, B0, gamma, L
