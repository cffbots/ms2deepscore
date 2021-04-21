import numpy as np
import pytest
from matchms import Spectrum
from ms2deepscore.SpectrumBinnerCustom import SpectrumBinnerCustom


def test_SpectrumBinnerCustom():
    """Test if default initalization works"""
    ms2ds_binner = SpectrumBinnerCustom(np.linspace(10, 1000, 101))
    assert ms2ds_binner.mz_max == 1000.0, "Expected different default value."
    assert ms2ds_binner.mz_min == 10.0, "Expected different default value."
    assert ms2ds_binner.mz_bins.shape == (101,), "Expected different calculated bin size."


def test_SpectrumBinnerCustom_fit_transform():
    """Test if collect binned spectrums method works."""
    ms2ds_binner = SpectrumBinnerCustom(np.linspace(0, 100, 101), peak_scaling=1.0)
    spectrum_1 = Spectrum(mz=np.array([10, 50, 99.9]),
                          intensities=np.array([0.7, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_01"})
    spectrum_2 = Spectrum(mz=np.array([10, 40, 90.]),
                          intensities=np.array([0.4, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_02"})

    binned_spectrums = ms2ds_binner.fit_transform([spectrum_1, spectrum_2])
    assert ms2ds_binner.known_bins == [10, 40, 50, 90, 99], "Expected different known bins."
    assert len(binned_spectrums) == 2, "Expected 2 binned spectrums."
    assert binned_spectrums[0].binned_peaks == {0: 0.7, 2: 0.2, 4: 0.1}, \
        "Expected different binned spectrum."
    assert binned_spectrums[0].get("inchikey") == "test_inchikey_01", \
        "Expected different inchikeys."


def test_SpectrumBinnerCustom_fit_transform_peak_scaling():
    """Test if collect binned spectrums method works with different peak_scaling."""
    ms2ds_binner = SpectrumBinnerCustom(np.linspace(0, 100, 101), peak_scaling=0.0)
    spectrum_1 = Spectrum(mz=np.array([10, 50, 99.]),
                          intensities=np.array([0.7, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_01"})
    spectrum_2 = Spectrum(mz=np.array([10, 40, 90.]),
                          intensities=np.array([0.4, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_02"})

    binned_spectrums = ms2ds_binner.fit_transform([spectrum_1, spectrum_2])
    assert ms2ds_binner.known_bins == [10, 40, 50, 90, 99], "Expected different known bins."
    assert len(binned_spectrums) == 2, "Expected 2 binned spectrums."
    assert binned_spectrums[0].binned_peaks == {0: 1.0, 2: 1.0, 4: 1.0}, \
        "Expected different binned spectrum."
    assert binned_spectrums[0].get("inchikey") == "test_inchikey_01", \
        "Expected different inchikeys."


def test_SpectrumBinner_transform():
    """Test if creating binned spectrums method works."""
    ms2ds_binner = SpectrumBinnerCustom(np.linspace(0, 100, 101), peak_scaling=1.0)
    spectrum_1 = Spectrum(mz=np.array([10, 20, 50, 99.]),
                          intensities=np.array([0.7, 0.6, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_01"})
    spectrum_2 = Spectrum(mz=np.array([10, 30, 40, 90.]),
                          intensities=np.array([0.4, 0.5, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_02"})

    binned_spectrums = ms2ds_binner.fit_transform([spectrum_1, spectrum_2])
    assert ms2ds_binner.known_bins == [10, 20, 30, 40, 50, 90, 99], "Expected different known bins."

    spectrum_3 = Spectrum(mz=np.array([10, 20, 30, 50.]),
                      intensities=np.array([0.4, 0.5, 0.2, 1.0]),
                      metadata={'inchikey': "test_inchikey_03"})
    spectrum_binned = ms2ds_binner.transform([spectrum_3])
    assert spectrum_binned[0].binned_peaks == {0: 0.4, 1: 0.5, 2: 0.2, 4: 1.0}, \
        "Expected different binned spectrum"


def test_SpectrumBinner_transform_missing_fraction():
    """Test if creating binned spectrums method works if peaks are unknown."""
    ms2ds_binner = SpectrumBinnerCustom(np.linspace(0, 100, 101), peak_scaling=1.0)
    spectrum_1 = Spectrum(mz=np.array([10, 20, 50, 99.]),
                          intensities=np.array([0.7, 0.6, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_01"})
    spectrum_2 = Spectrum(mz=np.array([10, 30, 40, 90.]),
                          intensities=np.array([0.4, 0.5, 0.2, 0.1]),
                          metadata={'inchikey': "test_inchikey_02"})

    binned_spectrums = ms2ds_binner.fit_transform([spectrum_1, spectrum_2])
    assert ms2ds_binner.known_bins == [10, 20, 30, 40, 50, 90, 99], "Expected different known bins."

    spectrum_3 = Spectrum(mz=np.array([10, 20, 30, 80.]),
                      intensities=np.array([0.4, 0.5, 0.2, 1.0]),
                      metadata={'inchikey': "test_inchikey_03"})
    with pytest.raises(AssertionError) as msg:
        _ = ms2ds_binner.transform([spectrum_3])
    assert "weighted spectrum is unknown to the model"in str(msg.value), \
        "Expected different exception."
