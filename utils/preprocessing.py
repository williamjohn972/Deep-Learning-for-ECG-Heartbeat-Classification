from tqdm.auto import tqdm
import neurokit2 as nk
import numpy as np

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

def linear_alignment_of_r_peaks(X:np.ndarray, 
                     sampling_rate, 
                     target_index, 
                     method="pantompkins"):
    
    """
    Aligns all ECG segments in X by detecting the R-peak and shifting them
    linearly to the target index. Gaps are filled with zeros
    """

    SIGNAL_LENGTH = X.shape[1]

    X_aligned_array = np.zeros_like(X)

    for i in tqdm(range(X.shape[0]), leave=False, desc="Performing Linear Alignment"):

        # we select the whole row 
        sample = X[i,:] 

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
        X_aligned_array[i, :] = sample_shifted[:SIGNAL_LENGTH]

    return X_aligned_array
            

    




