import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import WeightedRandomSampler

# DATA 
def split_x_y(df: pd.DataFrame):

    X = X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].astype(int).values

    return X,y

def calc_stats(X):
  mean_amplitude = np.mean(X)
  std_amplitude = np.std(X)
  min_amplitude = np.min(X)
  max_amplitude = np.max(X)

  print(f"Overall Mean Amplitude: {mean_amplitude:.4f}")
  print(f"Overall Std Dev Amplitude: {std_amplitude:.4f}")
  print(f"Overall Minimum Amplitude: {min_amplitude:.4f}")
  print(f"Overall Maximum Amplitude: {max_amplitude:.4f}")

def check_for_outliers(X, min_clip, max_clip):
    
    """
    Checks for outliers beyond clip_value * std of a normalised X
    """
    
    outlier_mask_min = (X < min_clip) 
    outlier_mask_max = (X > max_clip)

    outlier_count_min = np.sum(outlier_mask_min)
    outlier_count_max = np.sum(outlier_mask_max)
    total_outlier_count = outlier_count_max + outlier_count_min

    total_elements = X.size

    print(f"---- Training Set Outliers (Magnitude < {min_clip} & Magnitude > {max_clip}) ---")
    print(f"Total elements: {total_elements}")

    print(f"Outlier Count below {min_clip}: {outlier_count_min} | Percentage: {(outlier_count_min / total_elements) * 100: .3f}%")
    print(f"Outlier Count above {max_clip}: {outlier_count_max} | Percentage: {(outlier_count_max / total_elements) * 100: .3f}%")

    print(f"Total Outlier Count: {total_outlier_count} Percentage: {(total_outlier_count / total_elements) * 100: .3f}%")

def clip_data(X,min_clip, max_clip): 

    """
    Returns a X clipped to clip_value
    """
    return np.clip(X, a_min=min_clip, a_max=max_clip)

def calculate_class_weights(y):

    class_counts = np.bincount(y)

    """
    Calculates the class weights using inverse frequency 
    Therefore rarer classes would have the higher weight
    """

    return 1.0 / class_counts

def calculate_sample_weights(y, class_weights):
    return class_weights[y]

def sample(X, sample_size_fraction, random_state = None):

    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X)
    else:
        X_df = X

    X_sample = X_df.sample(frac=sample_size_fraction, 
                                       random_state= random_state)
    return X_sample.values

# PLOTTING
def plot_unique_class_samples(X, y, num_samples: int, figsize: tuple, xlabel: str, ylabel: str):
    
    for label in np.unique(y):
    
        sample_indices = np.where(y == label)[0][:num_samples]

        plt.figure(figsize=figsize) 
        
        for i, index in enumerate(sample_indices):
            plt.plot(X[index], label=f'Sample {i+1}')
        
        plt.title(f"Representative Samples for Class {label}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend() 
        plt.show()

def plot_class_distributions(y,
                             title: str = "Class Distribution",
                             xlabel: str = "Class Label",
                             ylabel: str = "Number of Samples",
                             ):
    

    unique, counts = np.unique(y, return_counts=True)
    total_samples = len(y)

    max_count = np.max(counts)
    y_limit_offset = max_count * 0.10
    plt.ylim(0, max_count + y_limit_offset)

    bars = plt.bar(unique, counts)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_samples) * 100

        plt.text(bar.get_x() + bar.get_width() / 2.0,
                 height + 0.05 * np.max(counts), 
                 f'{percentage: .1f}%',
                 ha="center", va='bottom', fontsize=9)

    plt.show()

    



