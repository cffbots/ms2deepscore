""" Functions to create binned vector from spectrum using linearly increasing width bins.
"""
import numpy as np
from tqdm import tqdm


def create_peak_list_linear(spectrums, peaks_vocab,
                            min_bin_size: float, slope: float,
                            mz_min: float = 10.0,
                            peak_scaling: float = 0.5,
                            progress_bar: bool = True):
    """Create list of (binned) peaks.

    Parameters
    ----------
    spectrums
        List of spectrums.
    peaks_vocab
        Dictionary of all known peak bins.
    slope
        Slope describing bin width change.
    mz_max
        Upper bound of m/z to include in binned spectrum. Default is 1000.0.
    mz_min
        Lower bound of m/z to include in binned spectrum. Default is 10.0.
    peak_scaling
        Scale all peak intensities by power pf peak_scaling. Default is 0.5.
    progress_bar
        Show progress bar if set to True. Default is True.
    """
    peak_lists = []
    missing_fractions = []

    for spectrum in tqdm(spectrums, desc="Spectrum binning",
                         disable=(not progress_bar)):
        doc = bin_number_array_linear(spectrum.peaks.mz, min_bin_size, slope, mz_min=mz_min)
        weights = spectrum.peaks.intensities ** peak_scaling
        
        # Find binned peaks present in peaks_vocab
        idx_in_vocab = [i for i, x in enumerate(doc) if x in peaks_vocab.keys()]
        idx_not_in_vocab = [i for i in np.arange(len(doc)) if i not in idx_in_vocab]
    
        doc_bow = [peaks_vocab[doc[i]] for i in idx_in_vocab]

        peak_lists.append(list(zip(doc_bow, weights[idx_in_vocab])))
        if len(idx_in_vocab) == 0:
            missing_fractions.append(1.0)
        else:
            missing_fractions.append(np.sum(weights[idx_not_in_vocab])/np.sum(weights))

    return peak_lists, missing_fractions


def unique_peaks_linear(spectrums, min_bin_size, slope, mz_min):
    """Collect unique (binned) peaks."""
    unique_peaks = set()
    for spectrum in spectrums:
        for mz in bin_number_array_linear(spectrum.peaks.mz, min_bin_size, slope, mz_min):
            unique_peaks.add(mz)
    unique_peaks = sorted(unique_peaks)
    class_values = {}

    for i, item in enumerate(unique_peaks):
        class_values[item] = i

    return class_values, unique_peaks


def set_slope_linear(number, min_bin_size, mz_min=10.0, mz_max=1000.0):
    """Set the slope for the change in bin widths depending on mz_min and mz_max
    as well as the minimum bin width min_bin_size.

    Parameters
    ----------
    mz_max
        Upper bound of m/z to include in binned spectrum. Default is 1000.0.
    mz_min
        Lower bound of m/z to include in binned spectrum. Default is 10.0.
    """
    return ((mz_max - mz_min) - min_bin_size * number)/((number-1) * number/2)


def bin_number_linear(mz, min_bin_size, slope, mz_min=10.0):
    return (2*(mz - mz_min)/slope + (min_bin_size/slope + 0.5)**2)**0.5 - min_bin_size/slope - 0.5


def bin_number_array_linear(mz, min_bin_size, slope, mz_min=10.0):
    if slope != 0.0:
        bin_numbers = (2*(mz - mz_min)/slope + (min_bin_size/slope + 0.5)**2)**0.5 - min_bin_size/slope - 0.5
    else:
        bin_numbers = mz/min_bin_size - mz_min * min_bin_size
    return bin_numbers.astype(int)
