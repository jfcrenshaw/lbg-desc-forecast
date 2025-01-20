"""Classes for performing Fisher forecasts for LBG cosmology."""

from copy import deepcopy
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt
from scipy.stats import chi2, multivariate_normal, norm

from .cosmo_factories import CosmoFactory
from .mapper import Mapper
from .utils import get_lensing_noise


class FisherMatrix:
    """Class for storing and manipulating Fisher matrices."""

    def __init__(
        self,
        matrix: np.ndarray,
        center: np.ndarray,
        keys: list[str],
        priors: np.ndarray | None = None,
    ) -> None:
        """Create Fisher matrix.

        Parameters
        ----------
        matrix0 : np.ndarray
            The Fisher matrix
        center : np.ndarray
            Central value for each parameter
        keys : list[str]
            List of parameter names
        priors : np.ndarray or None, optional
            Gaussian prior for each parameter. Default is None.
        """
        self.matrix = matrix
        self.center = center
        self.keys = keys
        self.priors = np.full_like(self.center, np.inf) if priors is None else priors

    @classmethod
    def load(cls, path: Path | str) -> "FisherMatrix":
        """Load Fisher matrix from a file.

        Parameters
        ----------
        path : Path or str
            Path from which to load Fisher matrix

        Returns
        -------
        FisherMatrix
            The Fisher matrix loaded from the file
        """
        data = np.load(path)
        return FisherMatrix(
            matrix=data["matrix"],
            center=data["center"],
            keys=list(data["keys"]),
            priors=data["priors"],
        )

    def save(self, path: Path | str) -> None:
        """Save Fisher matrix to a file.

        Parameters
        ----------
        path : path or str
            Path at which to save the Fisher matrix
        """
        np.savez(
            path,
            matrix=self.matrix,
            center=self.center,
            keys=self.keys,
            priors=self.priors,
        )

    def copy(self) -> "FisherMatrix":
        """Make copy of the fisher matrix.

        Returns
        -------
        FisherMatrix
            Deep copy of the Fisher matrix
        """
        return deepcopy(self)

    def set_prior(self, **kwargs) -> None:
        """Set Gaussian prior for parameter.

        Parameters
        ----------
        **kwargs
            Values to set by keyword. E.g. h=0.05 places a Gaussian prior on
            parameter "h" with a standard deviation of 0.05. Values must be
            floats or None.
        """
        for key, val in kwargs.items():
            idx = self.keys.index(key)
            self.priors[idx] = np.inf if val is None else val

    @property
    def matrix_with_priors(self) -> np.ndarray:
        """Fisher matrix, including the priors."""
        # Get the Fisher matrix without priors
        matrix = self.matrix.copy()

        # Add priors to diagonal
        idx = np.arange(len(matrix))
        matrix[idx, idx] += 1 / self.priors**2

        return matrix

    @property
    def covariance(self) -> np.ndarray:
        """Parameter covariance matrix."""
        return np.linalg.inv(self.matrix_with_priors)

    def __add__(self, other: "FisherMatrix") -> "FisherMatrix":
        """Add two Fisher matrices."""
        # First check that keys all match
        if not set(self.keys) == set(other.keys):
            raise ValueError("Cannot add Fisher matrices whose keys do not match.")

        # Also check that the centers match
        # TODO: determine if you can combine Fisher matrices whose centers don't match
        if not set(self.center) == set(other.center):
            raise ValueError("Cannot add Fisher matrices whose centers do not match.")

        # Now get order of keys in second matrix
        idx = [other.keys.index(key) for key in self.keys]

        # Duplicate Fisher matrix
        matrix = self.copy()

        # Add Fisher matrices
        matrix.matrix = self.matrix + other.matrix[np.ix_(idx, idx)]

        # Add priors
        matrix.priors = 1 / np.sqrt(1 / self.priors**2 + 1 / other.priors[idx] ** 2)

        # Add matrices after re-ordering so keys match
        return matrix

    def fix(self, keys: str | list) -> "FisherMatrix":
        """Fix certain parameters.

        Parameters
        ----------
        keys : str or list
            Name of a parameter to fix, or a list of parameters.

        Returns
        -------
        FisherMatrix
            New Fisher matrix with parameters fixed
        """
        # Cast to an iterable
        keys = np.atleast_1d(keys)

        # Get new matrix
        fisher = self.copy()

        # Loop over keys
        for key in keys:
            # Find index for this key
            idx = fisher.keys.index(key)

            # Remove corresponding row/column in the matrix
            fisher.matrix = np.delete(
                np.delete(fisher.matrix, idx, axis=0), idx, axis=1
            )

            # Remove corresponding center and prior
            fisher.center = np.delete(fisher.center, idx)
            fisher.priors = np.delete(fisher.priors, idx)

            # Delete the key
            del fisher.keys[idx]

        return fisher

    def marginalize(self, keys: str | list) -> "FisherMatrix":
        """Marginalize over certain parameters.

        Parameters
        ----------
        keys : str or list
            Name of a parameter to marginalize over, or a list of parameters.

        Returns
        -------
        FisherMatrix
            New Fisher matrix with parameters marginalized
        """
        # Cast to an iterable
        keys = np.atleast_1d(keys)

        # Get new matrix and corresponding covariance
        fisher = self.copy()
        cov = fisher.covariance

        # Loop over keys
        for key in keys:
            # Find index for this key
            idx = self.keys.index(key)

            # Remove corresponding row/column in the covariance
            cov = np.delete(np.delete(cov, idx, axis=0), idx, axis=1)

            # Remove corresponding center and priro
            fisher.center = np.delete(fisher.center, idx)
            fisher.priors = np.delete(fisher.priors, idx)

            # Delete the key
            del fisher.keys[idx]

        # Invert back to Fisher matrix
        fisher.matrix = np.linalg.inv(cov)

        return fisher

    def contour_1d(self, key: str, grid: np.ndarray) -> np.ndarray:
        """Calculate 1D marginalized distribution of the parameter

        Parameters
        ----------
        key : str
            Name of parameter
        grid : np.ndarray
            Grid along which to evaluate distribution

        Returns
        -------
        np.ndarray
            1D marginalized distribution
        """
        # Get index of parameter
        idx = self.keys.index(key)

        # Get center standard deviation of the parameter
        center = self.center[idx]
        sig = np.sqrt(self.covariance[idx, idx])
        print(center, sig)

        # Calculate contour
        contour = np.exp(-((grid - center) ** 2) / (2 * sig**2))
        contour /= np.sqrt(2 * np.pi * sig**2)

        return contour

    def contour_2d(
        self,
        keys: list[str],
        nsig: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate 2D N-sigma Gaussian contour.

        Parameters
        ----------
        keys : list[str]
            List of 2 parameters for which to create the contour
        nsig : float, optional
            Number of sigma. Default is 1.0

        Returns
        -------
        np.ndarray
            N-sigma contour
        """
        # Get sub-covariance
        if len(keys) != 2:
            raise ValueError("Must provide two keys.")
        idx = [self.keys.index(key) for key in keys]
        cov = self.covariance[np.ix_(idx, idx)]

        # Eigen-decomposition to enable rotation
        val, rot = np.linalg.eig(cov)

        # Scale by ... TODO fix this comment
        q = 2 * norm.cdf(nsig) - 1
        r2 = chi2.ppf(q, 2)
        val = np.sqrt(val * r2)

        # 1000 points along ellipse
        t = np.linspace(0, 2 * np.pi, 1000)
        xy = np.stack((np.cos(t), np.sin(t)), axis=-1)

        # Calculate ellipse points
        center = self.center[idx]
        x, y = nsig * rot @ (val * xy).T + np.array(center).reshape(2, 1)

        return x, y

    def sample(
        self,
        N: int,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample parameters from the Gaussian covariance.

        Parameters
        ----------
        N : int
            Number of samples to return
        random_state : int, np.random.Generator, or None, optional
            Value to set the random state. Default is None.

        Returns
        -------
        np.ndarray
            2D array of samples, of shape (N x len(self.keys))
        """
        # Resolve the random generator
        if not isinstance(random_state, np.random.Generator):
            random_state = np.random.default_rng(random_state)

        # Draw samples
        samples = multivariate_normal.rvs(
            self.center,
            self.covariance,
            size=N,
            random_state=random_state,
        )

        return samples


class Forecaster:
    """Class to perform Fisher forecasts."""

    def __init__(
        self,
        mappers: list[Mapper],
        cosmo: CosmoFactory,
        ell_min: int = 50,
        ell_max: int = 2000,
        ell_N: int = 20,
        clustering: bool = True,
        xcorr: bool = True,
        lensing: bool = True,
        step_frac: float = 0.05,
        step_min: float = 0.01,
    ) -> None:
        """Create forecaster.

        Parameters
        ----------
        mappers : list[Mapper]
            List of mappers for LBG samples to use in forecast.
        cosmo : CosmoFactory
            Factory for the cosmology object
        ell_min : int, optional
            Minimum ell value. Default is 50
        ell_max : int, optional
            Maximum ell value. Default is 2000
        ell_N : int, optional
            Number of logarithmic ell bins. Default is 20
        cosmology : Cosmology, optional
            Cosmology object from cosmo_tools.
        clustering : bool, optional
            Whether to use the clustering signal in the forecast.
            Default is True.
        xcorr : bool, optional
            Whether to use the LBG x CMB lensing signal in the forecast.
            Default is True.
        lensing : bool, optional
            Whether to use CMB lensing auto-correlation in the forecast.
            Default is True.
        step_frac : float, optional
            Fraction by which to step each parameter. Default is 0.05.
        step_min : float, optional
            Minimum value by which to step each parameter. Default is 0.01.
            Size by which to step each parameter. Default is 0.01.
        """
        # Save params
        self.mappers = mappers
        self.cosmo = cosmo
        self.ell_min = ell_min
        self.ell_max = ell_max
        self.ell_N = ell_N

        self.clustering = clustering
        self.xcorr = xcorr
        self.lensing = lensing

        self.step_frac = step_frac
        self.step_min = step_min

        # Create NaMaster ell bins
        edges = np.unique(np.geomspace(ell_min, ell_max + 1, ell_N + 1).astype(int))
        self.bins = nmt.NmtBin.from_edges(edges[:-1], edges[1:], is_Dell=False)

        # nside required to accommodate ell_max
        nside = (ell_max + 1) / 3
        self.nside = 2 ** np.ceil(np.log2(nside)).astype(int)

        # Empty covariance
        self.cov: np.ndarray | None = None
        self.cov_inv: np.ndarray | None = None

    @property
    def cosmo_params(self) -> dict:
        """Parameters associated with the cosmological model"""
        return self.cosmo.params.copy()

    @property
    def lbg_params(self) -> dict:
        """Parameters associated with the LBG tracers"""
        params = dict()
        for mapper in self.mappers:
            params[mapper.drop_band] = dict(
                dz=mapper.dz,
                f_interlopers=mapper.f_interlopers,
                g_bias=mapper.g_bias,
                g_bias_inter=mapper.g_bias_inter,
                mag_bias=mapper.mag_bias,
            )
        return params

    @property
    def params(self) -> dict:
        """All parameters"""
        all_params = self.cosmo_params
        for band in self.lbg_params:
            all_params |= {
                f"{band}_{key}": val  # type: ignore
                for key, val in self.lbg_params[band].items()
            }

        return all_params

    @property
    def keys(self) -> list[str]:
        """Keys for all parameters"""
        return list(self.params.keys())

    @property
    def values(self) -> np.ndarray:
        """Values for all parameters"""
        return np.array(list(self.params.values()))

    @property
    def dparams(self) -> np.ndarray:
        return np.clip(self.step_frac * self.values, self.step_min, None)

    def _step_by_key(
        self,
        key: str | None,
        sign: int,
    ) -> tuple[CosmoFactory, list[Mapper]]:
        """Return new param dicts with parameter incremented

        Parameters
        ----------
        key : str
            Name of parameter to increment
        sign : int
            +1 or -1, multiplied by the step size

        Returns
        -------
        CosmoFactory
            Cosmology factory, with a parameter (possibly) incremented
        list[Mapper]
            List of LBG mappers, with a parameter (possibly) incremented
        """
        # New copies of cosmology factory and mapper, to avoid in-place changes
        cosmo = self.cosmo.copy()
        mappers = {mapper.drop_band: mapper.copy() for mapper in self.mappers}

        # If we are not incrementing any parameters, just return these
        if key is None:
            return cosmo, list(mappers.values())

        # Otherwise, check this is a valid key
        if key not in self.keys:
            raise ValueError("key not in list of parameters.")

        # Increment the parameter
        idx = self.keys.index(key)
        new_val = self.params[key] + sign * self.dparams[idx]

        # Set the new value
        if key in cosmo.params:
            setattr(cosmo, key, new_val)
        else:
            # Split dropout band from key name
            band, subkey = key.split("_", maxsplit=1)

            # And increment parameter in corresponding mapper
            setattr(mappers[band], subkey, new_val)

        return cosmo, list(mappers.values())

    def create_signal(
        self,
        param_to_step: str | None = None,
        sign: int = +1,
    ) -> np.ndarray:
        """Create signal vector.

        Order is (here assuming 2 LBG samples):

        gg kg gg kg kk

        However some of these may be missing if self.clustering,
        self.xcorr, or self.lensing is False.

        Parameters
        ----------
        param_to_step : str or None, optional
            Name of parameter to increment. Default is None.
        sign : int, optional
            Sign of the increment. Can be +/- 1. Default is +1.

        Returns
        -------
        np.ndarray
            Signal vector
        """
        if sign not in [+1, -1]:
            raise ValueError("sign must be +/- 1")

        # Increment parameters
        cosmo, mappers = self._step_by_key(param_to_step, sign)

        # Get the NaMaster bins object
        signal = []
        for mapper in mappers:
            # Get spectra
            ell, Cgg, Ckg, Ckk, Cff = mapper.calc_spectra(cosmo.cosmology)
            Cgg += mapper.shot_noise
            Cgg += mapper.contamination * Cff
            Ckk += get_lensing_noise()[1]

            # Bin the C_ells
            if self.clustering:
                signal.append(self.bins.bin_cell(Cgg[: self.ell_max + 1]))
            if self.xcorr:
                signal.append(self.bins.bin_cell(Ckg[: self.ell_max + 1]))

        # Add single lensing signal
        if self.lensing:
            signal.append(self.bins.bin_cell(Ckk[: self.ell_max + 1]))

        signal = np.concatenate(signal)

        return signal

    def _create_cov_products(self, mapper: Mapper) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        nmt.NmtField,
        nmt.NmtField,
    ]:
        """Create products required to calculate covariance.

        Parameters
        ----------
        mapper : Mapper
            The mapper object to create fields for.

        Returns
        -------
        np.ndarray
            ell grid
        np.ndarray
            Cgg -- LBG auto-correlation
        np.ndarray
            Ckg -- cross-correlation between LBGs and CMB lensing
        np.ndarray
            Ckk -- CMB lensing auto-correlation
        np.ndarray
            Cff -- non-uniformity auto-correlation
        nmt.NmtField
            Galaxy tracer field
        nmt.NmtField
            CMB lensing tracer field
        """
        # Get spectra and noise
        ell, Cgg, Ckg, Ckk, Cff = mapper.calc_spectra(self.cosmo.cosmology)
        Cgg += mapper.shot_noise
        Ckk += get_lensing_noise()[1]

        # Upscale mask
        mask = hp.pixelfunc.ud_grade((~mapper.mask).astype(int), self.nside)

        # Generate random field realizations
        realized_fields = nmt.utils.synfast_spherical(
            self.nside,
            np.array([Cgg, Ckg, Ckk]),
            [0, 0],
        )
        g_field = nmt.NmtField(
            mask,
            realized_fields[0, None],
            lmax=self.ell_max,
            lmax_mask=self.ell_max,
        )
        k_field = nmt.NmtField(
            mask,
            realized_fields[1, None],
            lmax=self.ell_max,
            lmax_mask=self.ell_max,
        )

        return ell, Cgg, Ckg, Ckk, Cff, g_field, k_field

    def create_cov(self) -> None:
        """Create full covariance matrix.

        The structure of the covariance matrix is (here assuming 2 LBG samples):

            gg_gg gg_kg             gg_kk
            kg_gg kg_kg             kg_kk
                        gg_gg gg_kg gg_kk
                        kg_gg kg_kg kg_kk
            kk_gg kk_kg kk_gg kk_kg kk_kk

        However some of these blocks may be removed if self.clustering,
        self.xcorr, or self.lensing is False.

        sets self.cov and self.cov_inv attributes
        """
        # Building block of zeros
        Z = np.zeros((self.ell_N, self.ell_N))

        # Constants used to determine indexing below
        A = 2 if (self.clustering and self.xcorr) else 1
        B = A - 1
        C = 1 if self.lensing else 0

        # Create initial list of zero blocks
        N = A * len(self.mappers) + C
        Cov = [[Z for j in range(N)] for i in range(N)]

        # All fields are spin-zero
        spins = [0, 0, 0, 0]

        # Blocks for clustering and cross-correlation for each LBG sample
        for i, mapper in enumerate(self.mappers):
            # Get spectra and noise
            ell, Cgg, Ckg, Ckk, Cff, g_field, k_field = self._create_cov_products(
                mapper
            )

            # Create NaMaster workspaces
            cw = nmt.NmtCovarianceWorkspace.from_fields(
                g_field, g_field, g_field, g_field
            )
            wgg = nmt.NmtWorkspace.from_fields(g_field, g_field, self.bins)
            wkg = nmt.NmtWorkspace.from_fields(k_field, g_field, self.bins)
            wkk = nmt.NmtWorkspace.from_fields(k_field, k_field, self.bins)

            # First we will handle correlations within the same redshift bin
            if self.clustering:
                Cov_gggg = nmt.gaussian_covariance(
                    cw, *spins, [Cgg], [Cgg], [Cgg], [Cgg], wgg, wb=wgg
                )
                Cff_ = np.interp(self.bins.get_effective_ells(), ell, Cff)
                Cov_gggg += np.diag(Cff_**2)
                Cov[A * i][A * i] = Cov_gggg

            if self.clustering and self.xcorr:
                Cov_kggg = nmt.gaussian_covariance(
                    cw, *spins, [Ckg], [Ckg], [Cgg], [Cgg], wkg, wb=wgg
                )
                Cov[A * i][A * i + B] = Cov_kggg
                Cov[A * i + B][A * i] = Cov_kggg

            if self.xcorr:
                Cov_kgkg = nmt.gaussian_covariance(
                    cw, *spins, [Ckk], [Ckg], [Ckg], [Cgg], wkg, wb=wkg
                )
                Cov[A * i + B][A * i + B] = Cov_kgkg

            if self.lensing:
                if self.clustering:
                    Cov_kkgg = nmt.gaussian_covariance(
                        cw, *spins, [Ckg], [Ckg], [Ckg], [Ckg], wkk, wb=wkg
                    )
                    Cov[-1][A * i] = Cov_kkgg
                    Cov[A * i][-1] = Cov_kkgg

                if self.xcorr:
                    Cov_kkkg = nmt.gaussian_covariance(
                        cw, *spins, [Ckk], [Ckg], [Ckk], [Ckg], wkk, wb=wkg
                    )
                    Cov[-1][A * i + B] = Cov_kkkg
                    Cov[A * i + B][-1] = Cov_kkkg

        # Now lensing autocorrelation
        if self.lensing:
            # Finally add the lensing block
            Cov_kkkk = nmt.gaussian_covariance(
                cw, *spins, [Ckk], [Ckk], [Ckk], [Ckk], wkk, wb=wkk
            )
            Cov[-1][-1] = Cov_kkkk

        # Finally, cross-correlations between different redshift bins
        for i in range(len(self.mappers)):
            for j in range(i + 1, len(self.mappers)):
                # Get mappers for each tomographic bin
                mapper_i = self.mappers[i]
                mapper_j = self.mappers[j]

                # Get products associated with auto-spectra
                _, Cgg_i, Ckg_i, Ckk_i, Cff_i, g_field_i, k_field_i = (
                    self._create_cov_products(mapper_i)
                )
                _, Cgg_j, Ckg_j, Ckk_j, Cff_j, g_field_j, k_field_j = (
                    self._create_cov_products(mapper_j)
                )

                # Calculate cross-spectrum
                Cgg_ij = mapper_i.cross_spectrum(mapper_j, self.cosmo.cosmology)

                # Create workspaces
                cw_ij = nmt.NmtCovarianceWorkspace.from_fields(
                    g_field_i, g_field_i, g_field_j, g_field_j
                )
                wgg_i = nmt.NmtWorkspace.from_fields(g_field_i, g_field_i, self.bins)
                wkg_i = nmt.NmtWorkspace.from_fields(k_field_i, g_field_i, self.bins)
                wgg_j = nmt.NmtWorkspace.from_fields(g_field_j, g_field_j, self.bins)
                wkg_j = nmt.NmtWorkspace.from_fields(k_field_j, g_field_j, self.bins)

                if self.clustering:
                    Cov_gggg_ij = nmt.gaussian_covariance(
                        cw_ij,
                        *spins,
                        [Cgg_ij],
                        [Cgg_ij],
                        [Cgg_ij],
                        [Cgg_ij],
                        wa=wgg_i,
                        wb=wgg_j,
                    )
                    # Upper triangle
                    Cov[A * i][A * j] = Cov_gggg_ij
                    # Lower triangle
                    Cov[A * j][A * i] = Cov_gggg_ij

                if self.clustering and self.xcorr:
                    Cov_kggg_ij = nmt.gaussian_covariance(
                        cw_ij,
                        *spins,
                        [Ckg_j],
                        [Ckg_j],
                        [Cgg_ij],
                        [Cgg_ij],
                        wa=wkg_i,
                        wb=wgg_j,
                    )
                    # Upper triangle
                    Cov[A * i][A * j + B] = Cov_kggg_ij
                    Cov[A * i + B][A * j] = Cov_kggg_ij
                    # Lower triangle
                    Cov[A * j + B][A * i] = Cov_kggg_ij
                    Cov[A * j][A * i + B] = Cov_kggg_ij

                if self.xcorr:
                    Cov_kgkg_ij = nmt.gaussian_covariance(
                        cw_ij,
                        *spins,
                        [Ckk_i],
                        [Ckg_j],
                        [Ckg_i],
                        [Cgg_ij],
                        wa=wkg_i,
                        wb=wkg_j,
                    )
                    # Upper triangle
                    Cov[A * i + B][A * j + B] = Cov_kgkg_ij
                    # Lower triangle
                    Cov[A * j + B][A * i + B] = Cov_kgkg_ij

        # Assemble matrix from blocks
        Cov = np.block(Cov)

        # Save covariance and inverse
        self.cov = Cov
        self.cov_inv = np.linalg.inv(Cov)

    def create_fisher_matrix(self) -> None:
        """Create Fisher matrix for forecast.

        sets self.fisher_matrix attribute
        """
        if self.cov is None:
            raise ValueError(
                "Covariance matrix is still None. "
                "You must first run self.create_cov()."
            )

        # Create empty fisher matrix
        keys = self.keys
        fisher = np.zeros((len(keys), len(keys)))

        # Loop over parameter pairs and calculate fisher entry
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                dmu_i = (
                    self.create_signal(keys[i], +1) - self.create_signal(keys[i], -1)
                ) / (2 * self.dparams[i])
                dmu_j = (
                    self.create_signal(keys[j], +1) - self.create_signal(keys[j], -1)
                ) / (2 * self.dparams[j])
                fisher[i, j] = np.dot(
                    dmu_i, np.dot(self.cov_inv, dmu_j.reshape(-1, 1))
                )[0]

        # Make symmetric
        fisher += np.triu(fisher, k=1).T

        self.fisher_matrix = FisherMatrix(
            matrix=fisher,
            center=self.values,
            keys=self.keys,
        )
