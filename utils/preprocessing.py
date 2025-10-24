from tqdm.auto import tqdm
import neurokit2 as nk
import numpy as np
from scipy import signal


# DENOISING
def apply_low_pass_filter(sample, sampling_frequency, cutoff_frequency):
    """
    Applies a zero-phase low pass Butterworth filter to every sample in X
    returns the filtered ECG sample 

    -------- Parameters ------
    cutoff_frequency: The frequency above which signals are attenuated
    """

    nyquist_freq = 0.5 * sampling_frequency
    normalized_cutoff = cutoff_frequency / nyquist_freq

    order = 3
    b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)

    # Apply filter to each sample
    sample_filtered = signal.filtfilt(b,a, sample)

    return sample_filtered

# R-PEAK
def get_r_peak_location(sample, sampling_rate, method="pantompkins"):
    # detect R-peaks using neurokit2
    _, rpeaks_info = nk.ecg_peaks(sample, sampling_rate=sampling_rate, method=method)

    # Extract R peak index (We simply extract the first R peak encountered)
    if rpeaks_info.get("ECG_R_Peaks") is not None and len(rpeaks_info['ECG_R_Peaks'] > 0):
        peak_index = rpeaks_info["ECG_R_Peaks"][0]

    else:
        # If we cant find anything we simply take the largest value
        peak_index = np.argmax(sample)

    return peak_index


def get_r_peak_locations(X, sampling_rate, method="pantompkins"):

    """
    Detects R peaks in the sample and returns indices of the R-Peaks
    """

    r_peak_indices = []

    # now we loop through each sample and detect the R peaks
    for sample in tqdm(X, leave=False):
        current_peak_index = get_r_peak_location(sample=sample, sampling_rate=sampling_rate, method=method)

        r_peak_indices.append(current_peak_index)

    return r_peak_indices

def linear_alignment_of_r_peak(sample, 
                     sampling_rate, 
                     target_index, 
                     method="pantompkins"):
    
    """
    X is a single sample

    Aligns all ECG segments in the X by detecting the R-peak and shifting them
    linearly to the target index. Gaps are filled with zeros
    """

    SIGNAL_LENGTH = len(sample)

    X_aligned_array = np.zeros_like(sample)

    # now we get the r-peak of this segment
    current_peak_index = get_r_peak_location(sample=sample, 
                                                sampling_rate=sampling_rate,
                                                method=method)
    
    # calculate the shift
    shift = target_index - current_peak_index

    # apply linear shift
    if shift > 0: 
        # shift right, prepend zeros, discard data from end
        sample_shifted = np.concatenate([np.zeros(shift), sample[:SIGNAL_LENGTH - shift]])

    elif shift < 0:
        abs_shift = abs(shift)
        # shift left, append zeros, discard data from start
        sample_shifted = np.concatenate([sample[abs_shift:], np.zeros(abs_shift)])

    else:
        sample_shifted = sample

    # Store the aligned sample
    X_aligned_array = sample_shifted[:SIGNAL_LENGTH]

    return X_aligned_array
            

    
# PIPELINE
def preprocessing(X):

    X_preprocessed = np.copy(X)

    # First we perform Low-Pass Filter
    print("Applying Low-Pass Filter ...")
    for i in tqdm(range(X_preprocessed.shape[0]), desc="Applying Low Pass Filter", leave=False):

        X_sample = X_preprocessed[i]

        X_preprocessed[i] = apply_low_pass_filter(sample=X_sample,
                            sampling_frequency=125,
                            cutoff_frequency=25)
    
    # Then we apply R-Peak Re alignment
    print("Perforing R-Peak Realignment ... ")
    for i in tqdm(range(X_preprocessed.shape[0]), leave=False):

        X_sample = X_preprocessed[i]

        X_preprocessed[i] = linear_alignment_of_r_peak(sample=X_sample,
                                        sampling_rate=125,
                                        target_index=94,
                                      method="pantompkins")
        
    return X_preprocessed



