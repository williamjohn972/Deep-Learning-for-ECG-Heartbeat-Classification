from tqdm.auto import tqdm
import neurokit2 as nk
import numpy as np
from scipy import signal


# DENOISING
def apply_low_pass_filter(sample, sampling_frequency=125, cutoff_frequency=25, order=3):
    """
    Applies a zero-phase low pass Butterworth filter to every sample in X
    returns the filtered ECG sample 

    -------- Parameters ------
    cutoff_frequency: The frequency above which signals are attenuated
    """

    nyquist_freq = 0.5 * sampling_frequency
    normalized_cutoff = cutoff_frequency / nyquist_freq

    b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)

    # Apply filter to each sample
    sample_filtered = signal.filtfilt(b,a, sample)

    return sample_filtered

def apply_band_pass_filter(sample, sampling_frequency=125, low_cutoff=0.5, high_cutoff=40, order=4):
    """
    Applies a zero-phase band pass Butterworth filter to a single X sample
    returns the filtered ECG sample 

    -------- Parameters ------
    low_cutoff: The frequency below which signals are attenuated
    high_cutoff: The frequency above which signals are attenuated
    order: filter order (higher = steeper roll-off)
    """

    nyquist_freq = 0.5 * sampling_frequency
    low = low_cutoff / nyquist_freq
    high = high_cutoff / nyquist_freq


    b, a = signal.butter(order, [low,high], btype="band", analog=False)

    # Apply filter to each sample
    sample_filtered = signal.filtfilt(b,a, sample, method="pad")

    return sample_filtered

# R-PEAK
def get_r_peak_location(sample, sampling_rate, method="neurokit"):
    # detect R-peaks using neurokit2
    _, rpeaks_info = nk.ecg_peaks(sample, sampling_rate=sampling_rate, method=method)


    r_locs = rpeaks_info.get("ECG_R_Peaks",[])

    # Handle segments that are too short, empty, or perfectly flat
    if len(sample) < 10 or np.std(sample) < 1e-6:
        # Fallback to argmax or a safe index
        return int(np.argmax(sample)) if len(sample) > 0 else -1 
    # -------------------------------------------------

    # Extract R peak index (We simply extract the first R peak encountered)
    if r_locs is None or len(r_locs) == 0:
        peak_index = int(np.argmax(sample))

    elif len(r_locs) > 1:
        amax = int(np.argmax(sample))
        peak_index = int(r_locs[np.argmin(np.abs(r_locs-amax))])

    else:
        # If we cant find anything we simply take the largest value
        peak_index = int(r_locs[0])

    return peak_index


def get_r_peak_locations(X, sampling_rate, method="neurokit"):

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
                     method="neurokit"):
    
    """
    Aligns all ECG segments in the sample by detecting the R-peak and shifting them
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
            
# NORMALIZING 
def z_score_normalization(X):
    """
    Each sample is normalised on its own
    sample is a row and each column is the time
    """
    sample_means = np.mean(X, axis=1, keepdims=True)
    sample_stds = np.std(X, axis=1, keepdims=True)

    # we add 1e-8 to the std to prevent a zero division
    X_normalized = (X - sample_means) / (sample_stds + 1e-8)

    return X_normalized


# PIPELINE
class Preprocessing():

    def __init__(self, 
                 sample_freq, cutoff_freq, order, 
                 target_r_peak_index,method):
        
        """
        # ---------- Step 1: Low-Pass Filter ---------
        params:
        sample_freq
        cutoff_freq

        # -----------Step 2: R-Peak Alignment --------
        params:
        target_r_peak_index
        method
        
        """
        
        self.sample_freq = sample_freq
        self.cutoff_freq = cutoff_freq
        self.order = order

        self.target_r_peak_index = target_r_peak_index
        self.method = method

    def transform(self,X):

        X_preprocessed = np.copy(X)

        # First we perform Low-Pass Filter
        print("Applying Low-Pass Filter ...")
        for i in tqdm(range(X_preprocessed.shape[0]), desc="Applying Low Pass Filter", leave=False):

            X_sample = X_preprocessed[i]

            X_preprocessed[i] = apply_low_pass_filter(sample=X_sample,
                                sampling_frequency=self.sample_freq, order=self.order,
                                cutoff_frequency=self.cutoff_freq)
        
        # Then we apply R-Peak Re alignment
        print("Perforing R-Peak Realignment ... ")
        for i in tqdm(range(X_preprocessed.shape[0]), leave=False):

            X_sample = X_preprocessed[i]

            X_preprocessed[i] = linear_alignment_of_r_peak(sample=X_sample,
                                            sampling_rate=self.sample_freq,
                                            target_index=self.target_r_peak_index,
                                        method=self.method)
            
        # print("Performing Z-Score Normalization for Each Sample ...")
        # X_preprocessed_final = z_score_normalization(X_preprocessed)
        
        print("Completed Preprocessing")
        return X_preprocessed



