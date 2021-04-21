import json
from typing import List
from tqdm import tqdm
import numpy as np
from matchms.typing import SpectrumType

from .BinnedSpectrum import BinnedSpectrum
from .typing import BinnedSpectrumType
from .spectrum_binning_custom import create_peak_list_custom
from .spectrum_binning_custom import unique_peaks_custom
from .utils import create_peak_dict


class SpectrumBinnerCustom:
    """Create binned spectrum data and keep track of parameters.

    Converts input spectrums into :class:`~ms2deepscore.BinnedSpectrum` objects.
    Binning is here done using custom set array (`mz_bins`).
    """
    def __init__(self, mz_bins: np.ndarray,
                 peak_scaling: float = 0.5, allowed_missing_percentage: float = 0.0):
        """

        Parameters
        ----------
        mz_bins
            Array of peak mz bins.
        peak_scaling
            Scale all peak intensities by power pf peak_scaling. Default is 0.5.
        allowed_missing_percentage:
            Set the maximum allowed percentage of the spectrum that may be unknown
            from the input model. This is measured as percentage of the weighted, unknown
            binned peaks compared to all peaks of the spectrum. Default is 0, which
            means no unknown binned peaks are allowed.
        """
        # pylint: disable=too-many-arguments
        self.number_of_bins = len(mz_bins) - 1
        self.mz_bins = mz_bins
        self.mz_max = mz_bins[-1]
        self.mz_min = mz_bins[0]
        self.peak_scaling = peak_scaling
        self.allowed_missing_percentage = allowed_missing_percentage
        self.peak_to_position = None
        self.known_bins = None

    @classmethod
    def from_json(cls, json_str: str):
        """Create SpectrumBinner instance from json.

        Parameters
        ---------
        json_str
            Json string containing the dictionary to create a SpectrumBinner.
        """
        binner_dict = json.loads(json_str)
        spectrum_binner = cls(binner_dict["number_of_bins"],
                              binner_dict["mz_max"], binner_dict["mz_min"],
                              binner_dict["peak_scaling"],
                              binner_dict["allowed_missing_percentage"])
        spectrum_binner.peak_to_position = {int(key): value for key, value in binner_dict["peak_to_position"].items()}
        spectrum_binner.known_bins = binner_dict["known_bins"]
        return spectrum_binner

    def fit_transform(self, spectrums: List[SpectrumType], progress_bar=True):
        """Transforms the input *spectrums* into binned spectrums as needed for
        MS2DeepScore.

        Includes creating a 'vocabulary' of bins that have peaks in spectrums,
        which is stored in SpectrumBinner.known_bins.
        Creates binned spectrums from input spectrums and returns them.

        Parameters
        ----------
        spectrums
            List of spectrums.
        progress_bar
            Show progress bar if set to True. Default is True.
        """
        print("Collect spectrum peaks...")
        peak_to_position, known_bins = unique_peaks_custom(spectrums, self.mz_bins)
        print(f"Calculated embedding dimension: {len(known_bins)}.")
        self.peak_to_position = peak_to_position
        self.known_bins = known_bins

        print("Convert spectrums to binned spectrums...")
        return self.transform(spectrums, progress_bar)

    def transform(self, input_spectrums: List[SpectrumType],
                  progress_bar=True) -> List[BinnedSpectrumType]:
        """Create binned spectrums from input spectrums.

        Parameters
        ----------
        input_spectrums
            List of spectrums.
        progress_bar
            Show progress bar if set to True. Default is True.

        Returns:
            List of binned spectrums created from input_spectrums.
        """
        peak_lists, missing_fractions = create_peak_list_custom(input_spectrums,
                                                                self.peak_to_position,
                                                                self.mz_bins,
                                                                peak_scaling=self.peak_scaling,
                                                                progress_bar=progress_bar)
        spectrums_binned = []
        for i, peak_list in enumerate(tqdm(peak_lists,
                                           desc="Create BinnedSpectrum instances",
                                           disable=(not progress_bar))):
            assert 100*missing_fractions[i] <= self.allowed_missing_percentage, \
                f"{100*missing_fractions[i]:.2f} of weighted spectrum is unknown to the model."
            spectrum = BinnedSpectrum(binned_peaks=create_peak_dict(peak_list),
                                      metadata={"inchikey": input_spectrums[i].get("inchikey")})
            spectrums_binned.append(spectrum)
        return spectrums_binned

    def to_json(self):
        """Return SpectrumBinner instance as json dictionary."""
        return json.dumps(self.__dict__)
