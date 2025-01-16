"""Functions to perform operations with depth maps."""

import healpy as hp
import numpy as np
import pyccl as ccl
import pymaster as nmt
from scipy.optimize import minimize_scalar

from .cosmo_tools import DEG2_PER_STER, calc_lbg_spectra, calc_snr_from_products
from .density_interpolator import interpolate_number_density
from .utils import get_lensing_noise, load_m5_map

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
        self.mag_cut = mag_cut
        self.m5_min = m5_min
        self.contamination = contamination
        self.dz = dz
        self.f_interlopers = f_interlopers
        self.mag_bias = mag_bias
        self.dropout = dropout
        self.snr_min = snr_min

        # Load maps
        self.drop_map = load_m5_map(self.drop_band, self.year)
        self.det_map = load_m5_map(self.det_band, self.year)

    def _get_cuts(
        self, mag_cut: float | None, m5_min: float | None
    ) -> tuple[float, float]:
        """Handle logic associated with cuts values == None.

        Parameters
        ----------
        mag_cut : float, optional
            Magnitude cut on LBGs in the detection band. If provided, overrides
            value set at class instantiation. Default is None.
        m5_min : float, optional
            The minimum depth in the detection band. If provided, overrides
            value set at class instantiation. Default is None.

        Returns
        -------
        float
            mag_cut
        float
            m5_min
        """
        _mag_cut, _m5_min = None, None

        if mag_cut is not None:
            _mag_cut = mag_cut
        elif self.mag_cut is not None:
            _mag_cut = self.mag_cut
        else:
            raise ValueError(
                "mag_cut not set at class instantiation, so it must be provided."
            )

        if m5_min is not None:
            _m5_min = m5_min
        elif self.m5_min is not None:
            _m5_min = self.m5_min
        else:
            raise ValueError(
                "m5_min not set at class instantiation, so it must be provided."
            )

        return _mag_cut, _m5_min

    def construct_mask(
        self,
        mag_cut: float | None = None,
        m5_min: float | None = None,
    ) -> np.ndarray:
        """Construct the mask based on the cuts.

        Parameters
        ----------
        mag_cut : float, optional
            Magnitude cut on LBGs in the detection band. If provided, overrides
            value set at class instantiation. Default is None.
        m5_min : float, optional
            The minimum depth in the detection band. If provided, overrides
            value set at class instantiation. Default is None.

        Returns
        -------
        np.array
            Mask of healpy map
        """
        # Handle logic with cuts == None
        mag_cut, m5_min = self._get_cuts(mag_cut, m5_min)

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
        mag_cut: float | None = None,
        m5_min: float | None = None,
    ) -> np.ma.MaskedArray:
        """Create map of m5 depths.

        Parameters
        ----------
        mag_cut : float, optional
            Magnitude cut on LBGs in the detection band. If provided, overrides
            value set at class instantiation. Default is None.
        m5_min : float, optional
            The minimum depth in the detection band. If provided, overrides
            value set at class instantiation. Default is None.

        Returns
        -------
        np.ma.MaskedArray
            Healpy map of detection-band number density
        np.ma.MaskedArray
            Healpy map of dropout-band number density
        """
        # Construct the mask
        mask = self.construct_mask(mag_cut=mag_cut, m5_min=m5_min)

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
        mag_cut: float | None = None,
        m5_min: float | None = None,
    ) -> np.ma.MaskedArray:
        """Create map of LBG number density.

        Note this uses the interpolator, so is only accurate at the ~1% level.

        Parameters
        ----------
        mag_cut : float, optional
            Magnitude cut on LBGs in the detection band. If provided, overrides
            value set at class instantiation. Default is None.
        m5_min : float, optional
            The minimum depth in the detection band. If provided, overrides
            value set at class instantiation. Default is None.

        Returns
        -------
        np.ma.MaskedArray
            Healpy map of LBG number densities in deg^-2
        """
        # Handle logic with cuts == None
        mag_cut, m5_min = self._get_cuts(mag_cut, m5_min)

        # Construct mask
        mask = self.construct_mask(mag_cut=mag_cut, m5_min=m5_min)

        # Calculate density map
        density = interpolate_number_density(self.drop_band, mag_cut, self.det_map)
        density = np.ma.array(density, fill_value=np.nan)
        density.mask = mask

        return density

    def _measure_Cff(self, mag_cut: float, m5_min: float) -> np.ndarray:
        """Measure auto-spectrum of non-uniformity.

        Parameters
        ----------
        mag_cut : float
            Magnitude cut on LBGs in the detection band
        m5_min : float
            The minimum depth in the detection band

        Returns
        -------
        nmt.NmtBin
            Pymaster bin object
        np.ndarray
            Auto-spectrum of non-uniformity
        """
        # Create scalar field from fractional density fluctuations
        density = self.map_density(mag_cut, m5_min)
        f = (density - np.ma.mean(density)) / np.ma.mean(density)
        field = nmt.NmtField((~f.mask).astype(int), f.data[None, :])

        # Calculate C_ell's from bandpasses
        b = nmt.NmtBin.from_nside_linear(128, 1)
        Cff = nmt.compute_full_master(field, field, b)[0]

        return b, Cff

    def calc_spectra(
        self,
        mag_cut: float | None = None,
        m5_min: float | None = None,
        contamination: float | None = None,
        dz: float | None = None,
        f_interlopers: float | None = None,
        mag_bias: float | None = None,
        cosmology: ccl.Cosmology | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate power spectra associated with this tomographic bin.

        Parameters
        ----------
        mag_cut : float, optional
            Magnitude cut on LBGs in the detection band. If provided, overrides
            value set at class instantiation. Default is None.
        m5_min : float, optional
            The minimum depth in the detection band. If provided, overrides
            value set at class instantiation. Default is None.
        contamination : float or None, optional
            Fraction of non-uniformity map that contaminates LSS signal.
            If provided, overrides value set at class instantiation.
            Default is None.
        dz : float, optional
            Amount by which to shift the distribution of true LBGs (i.e.
            interlopers are not shifted). This corresponds to the DES delta z
            nuisance parameters. If provided, overrides value set at class
            instantiation. Default is None.
        f_interlopers : float, optional
            Fraction of low-redshift interlopers. Same p(z) shape is used
            for interlopers, but shifted to the redshift corresponding to
            Lyman-/Balmer-break confusion. If provided, overrides value set at
            class instantiation. Default is None.
        mag_bias : float or None, optional
            Magnification bias alpha value. If provided, overrides
            value set at class instantiation. Default is None.
        cosmology : ccl.Cosmology
            CCL cosmology object. If None, vanilla LCDM is used. Default is None.

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
            Cff -- non-uniformity auto-spectrum
        """
        # Handle logic with overrides
        mag_cut, m5_min = self._get_cuts(mag_cut, m5_min)
        contamination = self.contamination if contamination is None else contamination
        dz = self.dz if dz is None else dz
        f_interlopers = self.f_interlopers if f_interlopers is None else f_interlopers
        mag_bias = self.mag_bias if mag_bias is None else mag_bias

        # Create masked depth
        drop_map, det_map = self.map_depth(mag_cut, m5_min)

        # Get median m5_det
        m5_det = np.ma.median(det_map)

        # Calculate spectra
        ell, Cgg, Ckg, Ckk = calc_lbg_spectra(
            band=self.drop_band,
            mag_cut=mag_cut,
            m5_det=m5_det,
            dz=dz,
            f_interlopers=f_interlopers,
            mag_bias=mag_bias,
            cosmology=cosmology,
        )

        # Calculate non-uniformity auto-spectrum
        b, Cff = self._measure_Cff(mag_cut, m5_min)

        # Approximate with a running power-law
        # Cff = a * ell ^ (m * ln(ell) + b)
        # fit via ln(Cff) = m * ln(ell)^2 + b * ln(x) + ln(a)
        ell_grid = b.get_effective_ells()
        idx = np.where((Cff > 0))
        x = np.log(ell_grid[idx])
        y = np.log(Cff[idx])
        params = np.polyfit(x, y, deg=2)

        # Evaluate on our ell grid
        Cff = np.exp(np.polyval(params, np.log(ell)))

        # Scale by contamination fraction
        Cff *= np.sqrt(contamination)

        # Return all spectra
        return ell, Cgg, Ckg, Ckk, Cff

    def get_shot_noise(
        self,
        mag_cut: float | None = None,
        m5_min: float | None = None,
    ) -> float:
        """Calculate the sample shot noise.

        Parameters
        ----------
        mag_cut : float, optional
            Magnitude cut on LBGs in the detection band. If provided, overrides
            value set at class instantiation. Default is None.
        m5_min : float, optional
            The minimum depth in the detection band. If provided, overrides
            value set at class instantiation. Default is None.

        Returns
        -------
        float
            Shot noise
        """
        n = np.ma.mean(self.map_density(mag_cut, m5_min))
        Ngg = 1 / (n * DEG2_PER_STER)
        return Ngg

    def calc_lss_snr(
        self,
        mag_cut: float | None = None,
        m5_min: float | None = None,
        contamination: float | None = None,
        dz: float | None = None,
        f_interlopers: float | None = None,
        mag_bias: float | None = None,
        cosmology: ccl.Cosmology | None = None,
    ) -> tuple[float, float, float]:
        """Calculate SNR of LSS signals

        Parameters
        ----------
        mag_cut : float, optional
            Magnitude cut on LBGs in the detection band. If provided, overrides
            value set at class instantiation. Default is None.
        m5_min : float, optional
            The minimum depth in the detection band. If provided, overrides
            value set at class instantiation. Default is None.
        contamination : float or None, optional
            Fraction of non-uniformity map that contaminates LSS signal.
            If provided, overrides value set at class instantiation.
            Default is None.
        dz : float, optional
            Amount by which to shift the distribution of true LBGs (i.e.
            interlopers are not shifted). This corresponds to the DES delta z
            nuisance parameters. If provided, overrides value set at class
            instantiation. Default is None.
        f_interlopers : float, optional
            Fraction of low-redshift interlopers. Same p(z) shape is used
            for interlopers, but shifted to the redshift corresponding to
            Lyman-/Balmer-break confusion. If provided, overrides value set at
            class instantiation. Default is None.
        mag_bias : float or None, optional
            Magnification bias alpha value. If provided, overrides
            value set at class instantiation. Default is None.
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
        _, *spectra = self.calc_spectra(
            mag_cut=mag_cut,
            m5_min=m5_min,
            contamination=contamination,
            dz=dz,
            f_interlopers=f_interlopers,
            mag_bias=mag_bias,
            cosmology=cosmology,
        )

        # Calculate shot noise
        Ngg = self.get_shot_noise(mag_cut, m5_min)

        # Calculate f_sky from the mask
        f_sky = np.mean(~self.construct_mask(mag_cut, m5_min))

        return calc_snr_from_products(*spectra, Ngg, f_sky)  # type: ignore

    def optimize_cuts(
        self,
        contamination: float | None = 0.1,
        dz: float | None = 0.0,
        f_interlopers: float | None = 0.0,
        mag_bias: float | None = 1.0,
        metric: str = "tot",
        cosmology: ccl.Cosmology | None = None,
    ) -> tuple[float, float]:
        """Optimize mag_cut and m5_min by maximizing SNR of LSS signal.

        Parameters
        ----------
        contamination : float or None, optional
            Fraction of non-uniformity map that contaminates LSS signal.
            If provided, overrides value set at class instantiation.
            Default is 0.1.
        dz : float, optional
            Amount by which to shift the distribution of true LBGs (i.e.
            interlopers are not shifted). This corresponds to the DES delta z
            nuisance parameters. If provided, overrides value set at class
            instantiation. Default is zero.
        f_interlopers : float, optional
            Fraction of low-redshift interlopers. Same p(z) shape is used
            for interlopers, but shifted to the redshift corresponding to
            Lyman-/Balmer-break confusion. If provided, overrides value set at
            class instantiation. Default is zero.
        mag_bias : float or None, optional
            Magnification bias alpha value. If provided, overrides
            value set at class instantiation. Default is 1.0, which corresponds
            to no magnification bias.
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
                dz=dz,
                f_interlopers=f_interlopers,
                mag_bias=mag_bias,
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
                dz=dz,
                f_interlopers=f_interlopers,
                mag_bias=mag_bias,
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

    def get_forecast_products(
        self,
        ell_min: float,
        ell_max: float,
        ell_N: int,
        mag_cut: float | None = None,
        m5_min: float | None = None,
        contamination: float | None = None,
        dz: float | None = None,
        f_interlopers: float | None = None,
        mag_bias: float | None = None,
        cosmology: ccl.Cosmology | None = None,
    ) -> dict:
        """Create dictionary containing all the products needed for Fisher forecasts.

        Note the covariances returned in the dict contain the coupling estimated
        by namaster.

        Parameters
        ----------
        ell_min : float
            Minimum ell value.
        ell_max : float
            Maximum ell value.
        ell_N : int
            Number of logarithmic ell bins.
        mag_cut : float, optional
            Magnitude cut on LBGs in the detection band. If provided, overrides
            value set at class instantiation. Default is None.
        m5_min : float, optional
            The minimum depth in the detection band. If provided, overrides
            value set at class instantiation. Default is None.
        contamination : float or None, optional
            Fraction of non-uniformity map that contaminates LSS signal.
            If provided, overrides value set at class instantiation.
            Default is None.
        dz : float, optional
            Amount by which to shift the distribution of true LBGs (i.e.
            interlopers are not shifted). This corresponds to the DES delta z
            nuisance parameters. If provided, overrides value set at class
            instantiation. Default is None.
        f_interlopers : float, optional
            Fraction of low-redshift interlopers. Same p(z) shape is used
            for interlopers, but shifted to the redshift corresponding to
            Lyman-/Balmer-break confusion. If provided, overrides value set at
            class instantiation. Default is None.
        mag_bias : float or None, optional
            Magnification bias alpha value. If provided, overrides
            value set at class instantiation. Default is None.
        cosmology : ccl.Cosmology or None
            CCL cosmology object. If None, vanilla LCDM is used. Default is None.

        Returns
        -------
        dict
            ell: grid of effective ell values
            Cgg: LBG auto-spectrum
            Ckg: LBG x CMB lensing spectrum
            Ckk: CMB lensing auto-spectrum
            Cff: Non-uniformity auto-spectrum
            Cov_gggg: Cov[Cgg, Cgg]
            Cov_kggg: Cov[Ckg, Cgg]
            Cov_kkgg: Cov[Ckk, Cgg]
            Cov_kgkg: Cov[Ckg, Ckg]
            Cov_kkkg: Cov[Ckk, Ckg]
            Cov_kkkk: Cov[Ckk, Ckk]
        """
        # Get theory spectra
        ell, Cgg, Ckg, Ckk, Cff = self.calc_spectra(
            mag_cut=mag_cut,
            m5_min=m5_min,
            contamination=contamination,
            dz=dz,
            f_interlopers=f_interlopers,
            mag_bias=mag_bias,
            cosmology=cosmology,
        )

        # Get nside required to accommodate ell_max
        nside = (ell_max + 1) / 3
        nside = 2 ** np.ceil(np.log2(nside)).astype(int)

        # Get mask and upscale resolution
        mask = self.construct_mask(mag_cut, m5_min)
        mask = hp.pixelfunc.ud_grade((~mask).astype(int), nside)

        # Generate random field realizations
        realized_fields = nmt.utils.synfast_spherical(
            nside,
            np.array([Cgg, Ckg, Ckk]),
            [0, 0],
        )
        g_field = nmt.NmtField(
            mask,
            realized_fields[0, None],
            lmax=ell_max,
            lmax_mask=ell_max,
        )
        k_field = nmt.NmtField(
            mask,
            realized_fields[1, None],
            lmax=ell_max,
            lmax_mask=ell_max,
        )

        # Create logarithmic ell bins
        edges = np.unique(np.geomspace(ell_min, ell_max + 1, ell_N + 1).astype(int))
        b = nmt.NmtBin.from_edges(edges[:-1], edges[1:], is_Dell=False)

        # Create workspaces for each spectrum
        wgg = nmt.NmtWorkspace.from_fields(g_field, g_field, b)
        wkg = nmt.NmtWorkspace.from_fields(k_field, g_field, b)
        wkk = nmt.NmtWorkspace.from_fields(k_field, k_field, b)

        # Add noise to spectra
        Cgg_ = Cgg + self.get_shot_noise(mag_cut, m5_min)
        Ckg_ = Ckg  # No correlated noise
        Ckk_ = Ckk + get_lensing_noise()[1]

        # Calculate covariances
        cw = nmt.NmtCovarianceWorkspace.from_fields(g_field, g_field, g_field, g_field)
        spins = [0, 0, 0, 0]
        Cov_gggg = nmt.gaussian_covariance(
            cw, *spins, [Cgg_], [Cgg_], [Cgg_], [Cgg_], wgg, wb=wgg
        )
        Cov_kggg = nmt.gaussian_covariance(
            cw, *spins, [Ckg_], [Ckg_], [Cgg_], [Cgg_], wkg, wb=wgg
        )
        Cov_kkgg = nmt.gaussian_covariance(
            cw, *spins, [Ckg_], [Ckg_], [Ckg_], [Ckg_], wkk, wb=wkg
        )
        Cov_kgkg = nmt.gaussian_covariance(
            cw, *spins, [Ckk_], [Ckg_], [Ckg_], [Cgg_], wkg, wb=wkg
        )
        Cov_kkkg = nmt.gaussian_covariance(
            cw, *spins, [Ckk_], [Ckg_], [Ckk_], [Ckg_], wkk, wb=wkg
        )
        Cov_kkkk = nmt.gaussian_covariance(
            cw, *spins, [Ckk_], [Ckk_], [Ckk_], [Ckk_], wkk, wb=wkk
        )

        # Interpolate spectra onto new grid
        Cgg = np.interp(b.get_effective_ells(), ell, Cgg)
        Ckg = np.interp(b.get_effective_ells(), ell, Ckg)
        Ckk = np.interp(b.get_effective_ells(), ell, Ckk)
        Cff = np.interp(b.get_effective_ells(), ell, Cff)

        # Return everything in a dictionary
        return dict(
            ell=b.get_effective_ells(),
            Cgg=Cgg,
            Ckg=Ckg,
            Ckk=Ckk,
            Cff=Cff,
            Cov_gggg=Cov_gggg,
            Cov_kggg=Cov_kggg,
            Cov_kkgg=Cov_kkgg,
            Cov_kgkg=Cov_kgkg,
            Cov_kkkg=Cov_kkkg,
            Cov_kkkk=Cov_kkkk,
        )
