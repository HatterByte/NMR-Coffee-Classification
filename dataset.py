import numpy as np
import pandas as pd
from tqdm import tqdm
from simulation import simulate_fid, calculate_larmor
from analysis import extract_peak_intensities
from compounds import compound_data

# Simulation constants
N = 1024
L = 10.0  # μH
gamma = 42.577  # for ¹H
B0 = 0.7  # Tesla
f_larmor = calculate_larmor(gamma, B0)

# For fixed ppm axis
ppm_min, ppm_max = 0.5, 10
ppm_axis = np.linspace(ppm_max, ppm_min, N)

def fid_to_spectrum(t, fid):
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(fid)))
    freq = np.fft.fftshift(np.fft.fftfreq(len(t), d=(t[1] - t[0])))
    freq_ppm = f_larmor * (ppm_axis - 4.7) / 1e6
    spectrum_interp = np.interp(freq_ppm, freq, spectrum)
    return spectrum_interp

def simulate_single_sample(country):
    country_info = compound_data[country]
    t, fid = simulate_fid(N, f_larmor, country_info)
    spectrum = fid_to_spectrum(t, fid)
    peaks = extract_peak_intensities(spectrum, ppm_axis, country_info)

    features = {label: intensity for label, ppm, intensity in peaks}
    features['origin'] = country
    return features

def generate_dataset(samples_per_class=10000):
    all_data = []
    for country in ["Brazil", "Colombia", "Ethiopia"]:
        for _ in tqdm(range(samples_per_class), desc=f"Simulating {country}"):
            sample = simulate_single_sample(country)
            all_data.append(sample)
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    df = generate_dataset(samples_per_class=10000)
    
    # Fill missing metabolite intensities with 0 if any label is not present in all samples
    df.fillna(0, inplace=True)

    print("Sample dataset:")
    print(df.head())

    # Optional: Save to CSV
    df.to_csv("nmr_coffee_dataset.csv", index=False)
