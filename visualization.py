import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from analysis import extract_peak_intensities


def plot_fid(t, fid):
    st.subheader("FID Signal")
    st.line_chart(fid)


def plot_spectrum(t, fid, f_larmor, country_data):
    N = len(fid)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid)))
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=(t[1] - t[0])))  # in Hz

    # Define a fixed ppm range for the x-axis
    ppm_min = 0.5
    ppm_max = 10
    ppm_axis = np.linspace(ppm_max, ppm_min, N)  # high to low

    # Convert fixed ppm axis to frequency in Hz
    freq_ppm = f_larmor * (ppm_axis - 4.7) / 1e6  # center at 4.7 ppm

    # Interpolate spectrum values onto the new ppm-based frequency axis
    spectrum_interp = np.interp(freq_ppm, freq, spectrum)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ppm_axis, spectrum_interp, label="Spectrum")
    ax.set_xlim(ppm_max, ppm_min)

    # Peak labels
    for peak_ppm, _, label in country_data:
        ax.axvline(peak_ppm, color='red', linestyle='--', alpha=0.6)
        ax.text(peak_ppm, max(spectrum_interp) * 0.85, label, rotation=90,
                fontsize=8, va='center', ha='center')

    ax.set_xlabel("Chemical Shift (ppm)")
    ax.set_ylabel("Intensity")
    st.subheader("NMR Spectrum")
    st.pyplot(fig)
    peaks = extract_peak_intensities(spectrum_interp, ppm_axis, country_data)

    # Display below the plot
    st.markdown("### Extracted Peak Intensities")
    for label, ppm, intensity in peaks:
        st.write(f"**{label}** at {ppm:.2f} ppm: **{intensity:.2f}**")
    return peaks  # Add this line      

def plot_relaxation(t, signal_T1, signal_T2):
    fig, ax = plt.subplots()
    ax.plot(t, signal_T1, label="T₁ Recovery", color='green')
    ax.plot(t, signal_T2, label="T₂ Decay", color='blue')
    ax.legend()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnetization")
    st.subheader("Relaxation Curves")
    st.pyplot(fig)
