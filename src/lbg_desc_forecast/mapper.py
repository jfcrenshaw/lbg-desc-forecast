"""Functions to perform operations with depth maps."""

import numpy as np
import pyccl as ccl
import pymaster as nmt
from scipy.optimize import minimize_scalar

from .cosmo_tools import DEG2_PER_STER, _calc_snr_from_products, calc_lbg_spectra
from .density_interpolator import interpolate_number_density
from .utils import load_m5_map

_det_bands = {
    "u": "r",
    "g": "i",
    "r": "z",
}


class Mapper:

    def __init__(
        self,
        band: str,
        year: int,
        dropout: float = 1,
        snr_min: float = 3,
    ) -> None:
        """Create LBG mapper.

        Parameters
        ----------
        band : str
            Name of the dropout band
        year : int
            Year of LSST for depth map
        dropout : float, optional
            Minimum threshold in (dropout band) - (detection band) for LBG
            selection. Default is 1.
        snr_min : float, default=3
            Minimum SNR of LBGs in the dropout band. Default is 3.
        """
        # Save params
        self.drop_band = band
        self.det_band = _det_bands[band]
        self.year = year
        self.dropout = dropout
        self.snr_min = snr_min

        # Load maps
        self.drop_map = load_m5_map(self.drop_band, self.year)
        self.det_map = load_m5_map(self.det_band, self.year)

    def _construct_mask(
        self,
        mag_cut: float,
        m5_min: float,
    ) -> np.ndarray:
        """Construct the mask based on the cuts.

        Parameters
        ----------
        mag_cut : float
            Magnitude cut on LBGs in the detection band
        m5_min : float
            The minimum depth in the detection band

        Returns
        -------
        np.array
            Mask of healpy map
        """
        # m5_min can't be brighter than mag_cut
        m5_min = max(m5_min, mag_cut)

        # Mask pixels that aren't sufficiently deep
        mask = self.det_map.mask

        mask = mask | (self.det_map < m5_min).data

        # We can't include pixels where the dropout depth isn't
        # sufficiently deeper than mag_cut
        # Cut with reference to dropout threshold
        drop_min = mag_cut + self.dropout
        # Convert this to a cut on 5-sigma depth
        m5_drop_min = drop_min - 2.5 * np.log10(5 / self.snr_min)
        # Mask pixels not deep enough in dropout band
        mask = mask | (self.drop_map < m5_drop_min).data

        return mask

    def map_depth(
        self,
        mag_cut: float,
        m5_min: float,
    ) -> np.ma.MaskedArray:
        """Create map of m5 depths.

        Parameters
        ----------
        mag_cut : float
            Magnitude cut on LBGs in the detection band
        m5_min : float
            The minimum depth in the detection band

        Returns
        -------
        np.ma.MaskedArray
            Healpy map of detection-band number density
        np.ma.MaskedArray
            Healpy map of dropout-band number density
        """
        # Construct the mask
        mask = self._construct_mask(mag_cut=mag_cut, m5_min=m5_min)

        # Mask the dropout-band depth map
        drop = self.drop_map.copy()
        drop.mask = mask
        drop.data[mask] = np.nan

        # Mask the detection-band depth map
        det = self.det_map.copy()
        det.mask = mask
        det.data[mask] = np.nan

        return drop, det

    def map_density(
        self,
        mag_cut: float,
        m5_min: float,
    ) -> np.ma.MaskedArray:
        """Create map of LBG number density.

        Note this uses the interpolator, so is only accurate at the ~1% level.

        Parameters
        ----------
        mag_cut : float
            Magnitude cut on LBGs in the detection band
        m5_min : float
            The minimum depth in the detection band

        Returns
        -------
        np.ma.MaskedArray
            Healpy map of LBG number densities in deg^-2
        """
        # Construct mask
        mask = self._construct_mask(mag_cut=mag_cut, m5_min=m5_min)

        # Calculate density map
        density = interpolate_number_density(self.drop_band, mag_cut, self.det_map)
        density = np.ma.array(density, fill_value=np.nan)
        density.mask = mask

        return density

    def calc_spectra(
        self,
        mag_cut: float,
        m5_min: float,
        cosmology: ccl.Cosmology | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate power spectra associated with this tomographic bin.

        Parameters
        ----------
        mag_cut : float
            Magnitude cut on LBGs in the detection band
        m5_min : float
            The minimum depth in the detection band
        cosmology : ccl.Cosmology
            CCL cosmology object. If None, vanilla LCDM is used. Default is None.

        Returns
        -------
        np.ndarray
            Set of multipoles, ell
        np.ndarray
            Cgg -- LBG autospectrum
        np.ndarray
            Ckg -- LBG x CMB Lensing spectrum
        np.ndarray
            Ckk -- CMB Lensing autospectrum
        np.ndarray
            Cff -- non-uniformity autospectrum
        """
        # Create masked depth
        drop_map, det_map = self.map_depth(mag_cut, m5_min)

        # Get median m5_det
        m5_det = np.ma.median(det_map)

        # Calculate spectra
        ell, Cgg, Ckg, Ckk = calc_lbg_spectra(
            self.drop_band, mag_cut, m5_det, cosmology
        )

        # Calculate non-uniformity autos-pectrum...

        # Create scalar field from fractional density fluctuations
        density = self.map_density(mag_cut, m5_min)
        f = (density - np.nanmean(density)) / np.nanmean(density)
        field = nmt.NmtField((~f.mask).astype(int), f.data[None, :])

        # Calculate C_ell's from bandpasses
        b = nmt.NmtBin.from_nside_linear(128, 1)
        Cff = nmt.compute_full_master(field, field, b)[0]

        # Interpolate onto our ell grid
        Cff = np.interp(ell, b.get_effective_ells(), Cff)

        # Return all spectra
        return ell, Cgg, Ckg, Ckk, Cff

    def calc_lss_snr(
        self,
        mag_cut: float,
        m5_min: float,
        contamination: float,
        cosmology: ccl.Cosmology | None = None,
    ) -> tuple[float, float, float]:
        """Calculate SNR of LSS signals

        Parameters
        ----------
        mag_cut : float
            Magnitude cut on LBGs in the detection band
        m5_min : float
            The minimum depth in the detection band
        contamination : float
            Fraction of non-uniformity map that contaminates LSS signal
        cosmology : ccl.Cosmology or None
            CCL cosmology object. If None, vanilla LCDM is used. Default is None.

        Returns
        -------
        float
            Total SNR of clustering and CMB Lensing x-corr
        float
            SNR of clustering
        float
            SNR of CMB Lensing x-corr
        """
        # Create spectra
        _, *spectra = self.calc_spectra(mag_cut, m5_min, cosmology)

        # Weight contamination autospectrum
        spectra[-1] *= contamination

        # Calculate shot noise
        n = np.ma.mean(self.map_density(mag_cut, m5_min))
        Ngg = 1 / (n * DEG2_PER_STER)

        # Calculate f_sky from the mask
        f_sky = np.mean(~self._construct_mask(mag_cut, m5_min))

        return _calc_snr_from_products(*spectra, Ngg, f_sky)  # type: ignore

    def optimize_cuts(
        self,
        contamination: float,
        metric: str = "tot",
        cosmology: ccl.Cosmology | None = None,
    ) -> tuple[float, float]:
        """Optimize mag_cut and m5_min by maximizing SNR of LSS signal.

        Parameters
        ----------
        contamination : float
            Fraction of non-uniformity map that contaminates LSS signal
        metric : str, optional
            Which SNR to maximize. "tot" maximizes the total SNR, "gg" maximizes
            the clustering signal, and "kg" maximizes the cross-correlation with
            CMB lensing. Default is "tot"
        cosmology : ccl.Cosmology or None
            CCL cosmology object. If None, vanilla LCDM is used. Default is None.

        Returns
        -------
        float
            optimized mag_cut
        float
            optimized m5_min
        """
        # Determine the metric index
        if metric == "tot":
            idx = 0
        elif metric == "gg":
            idx = 1
        elif metric == "kg":
            idx = 2
        else:
            raise ValueError(f"Metric {metric} not supported.")

        # Prep the cosmology object
        if cosmology is None:
            cosmology = ccl.CosmologyVanillaLCDM()

        # First optimize mag_cut
        drop_shift = -self.dropout + 2.5 * np.log10(5 / self.snr_min)
        lower = max(23, np.ma.min(self.drop_map) + drop_shift)
        upper = min(np.ma.max(self.det_map), np.ma.max(self.drop_map) + drop_shift)
        res = minimize_scalar(
            fun=lambda mag_cut: -self.calc_lss_snr(
                mag_cut=mag_cut,
                m5_min=mag_cut,
                contamination=contamination,
                cosmology=cosmology,
            )[idx],
            bounds=(lower, upper),
            options=dict(xatol=1e-3),
        )
        if not res.success:
            raise RuntimeError("Optimizing mag_cut failed.")
        if np.isclose(res.x, lower):
            raise RuntimeError("mag_cut hit lower bound.")
        if np.isclose(res.x, upper):
            raise RuntimeError("mag_cut hit upper bound.")
        mag_cut = res.x

        # Now optimize m5_min
        lower = mag_cut
        upper = np.ma.max(self.det_map)
        res = minimize_scalar(
            fun=lambda m5_min: -self.calc_lss_snr(
                mag_cut=mag_cut,
                m5_min=m5_min,
                contamination=contamination,
                cosmology=cosmology,
            )[idx],
            bounds=(lower, upper),
            options=dict(xatol=1e-3),
        )
        if not res.success:
            raise RuntimeError("Optimizing m5_min failed.")
        if np.isclose(res.x, upper):
            raise RuntimeError("m5_min hit upper bound.")
        m5_min = res.x

        return mag_cut, m5_min
