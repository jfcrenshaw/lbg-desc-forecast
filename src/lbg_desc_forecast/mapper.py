"""Class that unifies an LBG tomographic bin, tracer, and depth maps."""

from copy import deepcopy

import numpy as np
import pyccl as ccl
import pymaster as nmt
from lbg_tools import TomographicBin

from .density_interpolator import interpolate_number_density
from .utils import get_lensing_noise, load_m5_map

# Constants for sky area
A_SKY = 41_253  # deg^2
DEG2_PER_STER = A_SKY / (4 * 3.14159)

# Detection bands for each dropout sample
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
        mag_cut: float | None = None,
        m5_min: float | None = None,
        contamination: float | None = 0.1,
        dz: float = 0.0,
        f_interlopers: float = 0.0,
        g_bias: float | None = None,
        g_bias_inter: float | None = None,
        mag_bias: float | None = None,
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
        mag_cut : float or None, optional
            Magnitude cut on LBGs in the detection band. Default is None.
        m5_min : float or None, optional
            The minimum depth in the detection band. Default is None.
        contamination : float, optional
            Fraction of non-uniformity map that contaminates LSS signal.
            Default is 0.1.
        dz : float, optional
            Amount by which to shift the distribution of true LBGs (i.e.
            interlopers are not shifted). This corresponds to the DES delta z
            nuisance parameters. Default is zero.
        f_interlopers : float, optional
            Fraction of low-redshift interlopers. Same p(z) shape is used
            for interlopers, but shifted to the redshift corresponding to
            Lyman-/Balmer-break confusion. Default is zero.
        g_bias : float or None, optional
            Galaxy bias for LBG population. If None, value is pulled from the
            lbg_tools TomographicBin. Default is None.
        g_bias_inter : float or None, optional
            Galaxy bias for the interloper population. If None, value is pulled from the
            lbg_tools TomographicBin. Default is None.
        mag_bias : float or None, optional
            Magnification bias alpha value. If None, value is pulled from the
            lbg_tools TomographicBin. Default is None.
        dropout : float, optional
            Minimum threshold in (dropout band) - (detection band) for LBG
            selection. Default is 1.
        snr_min : float, optional
            Minimum SNR of LBGs in the dropout band. Default is 3.
        """
        # Save params
        self.drop_band = band
        self.det_band = _det_bands[band]
        self.year = year
        self.mag_cut = mag_cut  # type: ignore
        self.m5_min = m5_min  # type: ignore
        self.contamination = contamination
        self.dz = dz
        self.f_interlopers = f_interlopers
        self.g_bias = g_bias
        self.g_bias_inter = g_bias_inter
        self.mag_bias = mag_bias
        self.dropout = dropout
        self.snr_min = snr_min

        # Load maps
        self.drop_map = load_m5_map(self.drop_band, self.year)
        self.det_map = load_m5_map(self.det_band, self.year)

    def copy(self) -> "Mapper":
        """Return copy of self."""
        return deepcopy(self)

    @property
    def mag_cut(self) -> float:
        """Magnitude cut on LBGs in the detection band."""
        if self._mag_cut is None:
            raise ValueError("mag_cut has not been set.")
        return self._mag_cut

    @mag_cut.setter
    def mag_cut(self, value: float | None) -> None:
        """Set magnitude cut on LBGs in the detection band."""
        self._mag_cut = value

    @property
    def m5_min(self) -> float:
        """The minimum depth in the detection band."""
        if self._m5_min is None:
            raise ValueError("m5_min has not been set.")
        return self._m5_min

    @m5_min.setter
    def m5_min(self, value: float | None) -> None:
        """Set the minimum depth in the detection band."""
        self._m5_min = value

    @property
    def mask(self) -> np.ndarray:
        """Construct the mask based on the cuts.

        Returns
        -------
        np.array
            Mask of healpy map
        """
        # m5_min can't be brighter than mag_cut
        m5_min = max(self.m5_min, self.mag_cut)

        # Mask pixels that aren't sufficiently deep
        mask = self.det_map.mask

        mask = mask | (self.det_map < m5_min).data

        # We can't include pixels where the dropout depth isn't
        # sufficiently deeper than mag_cut
        # Cut with reference to dropout threshold
        drop_min = self.mag_cut + self.dropout
        # Convert this to a cut on 5-sigma depth
        m5_drop_min = drop_min - 2.5 * np.log10(5 / self.snr_min)
        # Mask pixels not deep enough in dropout band
        mask = mask | (self.drop_map < m5_drop_min).data

        return mask

    @property
    def f_sky(self) -> float:
        """Calculate f_sky

        Returns
        -------
        float
            f_sky
        """
        return np.mean(~self.mask)

    @property
    def map_depth(self) -> np.ma.MaskedArray:
        """Create map of m5 depths.

        Returns
        -------
        np.ma.MaskedArray
            Healpy map of detection-band number density
        np.ma.MaskedArray
            Healpy map of dropout-band number density
        """
        # Construct the mask
        mask = self.mask

        # Mask the dropout-band depth map
        drop = self.drop_map.copy()
        drop.mask = mask
        drop.data[mask] = np.nan

        # Mask the detection-band depth map
        det = self.det_map.copy()
        det.mask = mask
        det.data[mask] = np.nan

        return drop, det

    @property
    def map_density(self) -> np.ma.MaskedArray:
        """Create map of LBG number density.

        Note this uses the interpolator, so is only accurate at the ~1% level.

        Returns
        -------
        np.ma.MaskedArray
            Healpy map of LBG number densities in deg^-2
        """
        # Interpolate density map
        density = interpolate_number_density(self.drop_band, self.mag_cut, self.det_map)
        density = np.ma.array(density, fill_value=np.nan)
        density.mask = self.mask

        return density

    @property
    def shot_noise(self) -> float:
        """Calculate sample shot noise spectrum.

        Returns
        -------
        float
            Shot noise
        """
        n = np.ma.mean(self.map_density)
        Ngg = 1 / (n * DEG2_PER_STER)
        return Ngg

    @property
    def tomographic_bin(self) -> TomographicBin:
        """Return the lbg_tools tomographic bin."""
        # Get median depth in detection band
        # Create masked depth
        drop_map, det_map = self.map_depth
        m5_det = np.ma.median(det_map)

        return TomographicBin(
            band=self.drop_band,
            mag_cut=self.mag_cut,
            m5_det=m5_det,
            dz=self.dz,
            f_interlopers=self.f_interlopers,
        )

    def get_tracer(
        self,
        cosmology: ccl.Cosmology | None = None,
    ) -> ccl.NumberCountsTracer:
        """Return the CCL tracer.

        Parameters
        ----------
        cosmology : ccl.Cosmology
            CCL cosmology object. If None, vanilla LCDM is used.


        Returns
        -------
        ccl.NumberCountsTracer
            Number counts tracer
        """
        # Get the tomographic bin
        tb = self.tomographic_bin

        # Sample p(z) and bias on dense grid
        z = np.arange(0, 8.01, 0.01)
        pz = np.interp(z, *tb.pz)

        # Determine galaxy bias
        z_interlopers, z_lbg = tb._get_z_grids()
        if self.g_bias_inter is None:
            b_interlopers = tb.g_bias[1][: z_interlopers.size]
        else:
            b_interlopers = [self.g_bias_inter] * len(z_interlopers)
        if self.g_bias is None:
            b_lbg = tb.g_bias[1][-z_lbg.size :]
        else:
            b_lbg = [self.g_bias] * len(z_lbg)
        b = np.interp(  # Resample on denser grid
            z,
            np.concatenate((z_interlopers, z_lbg)),
            np.concatenate((b_interlopers, b_lbg)),
        )

        # Determine magnification bias
        # alpha = 2.5 * d log10(N) / d mag
        # s = alpha / 2.5
        z_interlopers, z_lbg = tb._get_z_grids()
        alpha = tb.mag_bias if self.mag_bias is None else self.mag_bias
        alpha = np.array([0.0] * len(z_interlopers) + [alpha] * len(z_lbg))
        sz = alpha / 2.5
        sz = np.interp(  # Resample on denser grid
            z, np.concatenate((z_interlopers, z_lbg)), sz
        )

        # Prep the cosmology object
        if cosmology is None:
            cosmology = ccl.CosmologyVanillaLCDM()

        # Create the tracer
        tracer = ccl.NumberCountsTracer(
            cosmology,
            has_rsd=False,
            dndz=(z, pz),
            bias=(z, b),
            mag_bias=(z, sz),
        )

        return tracer

    def cross_Cff(self, mapper: "Mapper") -> tuple[np.ndarray, np.ndarray]:
        """Measure cross-correlation of non-uniformity with another mapper.

        Note this is *NOT* scaled by the contamination.

        Parameters
        ----------
        mapper : Mapper
            Other mapper, with which to measure cross-correlation of
            non-uniformity.

        Returns
        -------
        np.ndarray
            Grid of ell values
        np.ndarray
            Cross-spectrum of non-uniformity
        """
        # Create scalar field from fractional density fluctuations
        density = self.map_density
        f = (density - np.ma.mean(density)) / np.ma.mean(density)
        field = nmt.NmtField((~f.mask).astype(int), f.data[None, :])

        # Same for other field
        density2 = mapper.map_density
        f2 = (density2 - np.ma.mean(density2)) / np.ma.mean(density2)
        field2 = nmt.NmtField((~f2.mask).astype(int), f2.data[None, :])

        # Calculate C_ell's from bandpasses
        b = nmt.NmtBin.from_nside_linear(128, 1)
        Cff = nmt.compute_full_master(field, field2, b)[0]

        # Approximate with a running power-law
        # Cff = a * ell ^ (m * ln(ell) + b)
        # fit via ln(Cff) = m * ln(ell)^2 + b * ln(x) + ln(a)
        ell_grid = b.get_effective_ells()
        idx = np.where((Cff > 0))
        x = np.log(ell_grid[idx])
        y = np.log(Cff[idx])
        params = np.polyfit(x, y, deg=2)

        # Evaluate on our ell grid
        ell, _ = get_lensing_noise()
        Cff = np.exp(np.polyval(params, np.log(ell)))

        return ell, Cff

    def auto_Cff(self) -> tuple[np.ndarray, np.ndarray]:
        """Measure auto-spectrum of non-uniformity.

        Note this is *NOT* scaled by the contamination.

        Returns
        -------
        np.ndarray
            Grid of ell values
        np.ndarray
            Auto-spectrum of non-uniformity
        """
        return self.cross_Cff(mapper=self)

    def calc_spectra(self, cosmology: ccl.Cosmology | None = None):
        """Calculate auto-spectra

        Note this actually includes Ckg as well. I just call these auto-spectra
        because only one mapper is involved.

        Parameters
        ----------
        cosmology : ccl.Cosmology
            CCL cosmology object. If None, vanilla LCDM is used.

        Returns
        -------
        np.ndarray
            Set of multipoles, ell
        np.ndarray
            Cgg -- LBG auto-spectrum
        np.ndarray
            Ckg -- LBG x CMB Lensing spectrum
        np.ndarray
            Ckk -- CMB Lensing auto-spectrum
        np.ndarray
            Cff -- non-uniformity auto-spectrum (scaled by contamination)
        """
        # Prep the cosmology object
        if cosmology is None:
            cosmology = ccl.CosmologyVanillaLCDM()

        # Create tracers
        lbg_tracer = self.get_tracer(cosmology)
        cmb_lensing = ccl.CMBLensingTracer(cosmology, z_source=1100)

        # Calculate spectra
        ell = get_lensing_noise()[0]
        Cgg = ccl.angular_cl(cosmology, lbg_tracer, lbg_tracer, ell)
        Ckg = ccl.angular_cl(cosmology, cmb_lensing, lbg_tracer, ell)
        Ckk = ccl.angular_cl(cosmology, cmb_lensing, cmb_lensing, ell)
        Cff = self.auto_Cff()[1] * np.sqrt(self.contamination)

        return ell, Cgg, Ckg, Ckk, Cff

    def cross_spectrum(self, mapper: "Mapper", cosmology: ccl.Cosmology | None = None):
        """Calculate cross-spectra

        Parameters
        ----------
        mapper : Mapper
            Other mapper, with which to calculate cross-spectra.
        cosmology : ccl.Cosmology
            CCL cosmology object. If None, vanilla LCDM is used.

        Returns
        -------
        """
        # Create the two tracers
        tracer0 = self.get_tracer(cosmology)
        tracer1 = mapper.get_tracer(cosmology)

        # Calculate spectra
        ell = get_lensing_noise()[0]
        Cgg_cross = ccl.angular_cl(cosmology, tracer0, tracer1, ell)

        return Cgg_cross

    def calc_lss_snr(
        self,
        cosmology: ccl.Cosmology | None = None,
    ) -> tuple[float, float, float]:
        """Calculate SNR of LSS signals

        Parameters
        ----------
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
        ell, Cgg, Ckg, Ckk, Cff = self.calc_spectra(cosmology)
        Ngg = self.shot_noise + self.contamination * Cff
        Nkk = get_lensing_noise()[1]

        # Assemble signal vector
        mu = np.concatenate((Cgg, Ckg)).reshape(-1, 1)

        # Calculate covariances
        norm = 1 / (2 * ell + 1) / self.f_sky
        Cov_gggg = 2 * norm * (Cgg + Ngg) ** 2
        Cov_ggkg = 2 * norm * Ckg * (Cgg + Ngg)
        Cov_kgkg = norm * ((Ckk + Nkk) * (Cgg + Ngg) + Ckg**2)

        # Inverse covariance matrix
        det = Cov_gggg * Cov_kgkg - Cov_ggkg**2  # Determinant
        Cov_inv = np.block(
            [
                [np.diag(Cov_kgkg / det), -np.diag(Cov_ggkg / det)],
                [-np.diag(Cov_ggkg / det), np.diag(Cov_gggg / det)],
            ]
        )

        # Calculate weighted SNRs
        snr_tot = np.sqrt(mu.T @ Cov_inv @ mu)[0, 0]
        snr_gg = np.sqrt(np.sum(Cgg**2 / Cov_gggg))
        snr_kg = np.sqrt(np.sum(Ckg**2 / Cov_kgkg))

        return snr_tot, snr_gg, snr_kg
