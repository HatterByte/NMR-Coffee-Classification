
import numpy as np


π = np.pi

def calculate_larmor(gamma, B0):
    return gamma * B0 * 1e6

def calculate_capacitance(f_larmor, L_uH):
    L_H = L_uH * 1e-6
    C = 1 / ((2 * π * f_larmor)**2 * L_H)
    return C * 1e12  # in pF

def simulate_fid(N, f_larmor, country_data):
    t = np.linspace(0, 1, N)
    fid = np.zeros(N)

    for ppm, intensity, label in country_data:
        ppm_variation = ppm # Keep position fixed
        intensity_variation = intensity + np.random.uniform(-0.5, 0.5)  # Still vary intensity

        f_hz = f_larmor * (ppm_variation - 4.7) / 1e6
        decay = np.exp(-t / 0.15)
        fid += intensity_variation * decay * np.cos(2 * np.pi * f_hz * t)

    fid += np.random.normal(0, 1, N) * 0.5  # Add noise
    return t, fid

def simulate_t1_t2():
    T1 = np.round(np.random.uniform(0.8, 1.2), 2)
    T2 = np.round(np.random.uniform(0.12, 0.25), 2)
    t = np.linspace(0, 3, 300)
    return T1, T2, t, 1 - np.exp(-t / T1), np.exp(-t / T2)

def simulate_fidd(N, f_larmor, country_data):
    t = np.linspace(0, 1, N)
    fid = np.zeros(N)

    for ppm, intensity, label in country_data:
        ppm_variation = ppm # Keep position fixed
        intensity_variation = intensity + np.random.uniform(-5, 5)  # Still vary intensity

        f_hz = f_larmor * (ppm_variation - 4.7) / 1e6
        decay = np.exp(-t / 0.15)
        fid += intensity_variation * decay * np.cos(2 * np.pi * f_hz * t)

    fid += np.random.normal(0, 1, N) * 0.5  # Add noise
    return t, fid