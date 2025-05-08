import numpy as np

def extract_peak_intensities(spectrum, ppm_axis, country_data):
    peak_data = []

    for peak_ppm, _, label in country_data:
        # Find index closest to the metabolite's ppm
        idx = np.argmin(np.abs(ppm_axis - peak_ppm))

        # Optional: Look in a small window to find max around expected ppm
        window_size = 3  # +/- 3 points
        start = max(0, idx - window_size)
        end = min(len(spectrum), idx + window_size + 1)

        peak_intensity = np.max(spectrum[start:end])
        peak_data.append((label, peak_ppm, peak_intensity))

    return peak_data