"""Class to perform Fisher forecasting for LBG cosmology."""

import numpy as np
import pyccl as ccl

from .mapper import Mapper
from .utils import data_dir


class Forecaster:
    def __init__(
        self,
        mappers: list[Mapper],
        ell_min: float,
        ell_max: float,
        ell_N: int,
        contamination: float,
        cosmology: ccl.Cosmology,
        clustering: bool = True,
        xcorr: bool = True,
        lensing: bool = True,
    ) -> None:
        """Create forecaster class.

        Parameters
        ----------
        mappers : list[Mapper]
            List of mappers for LBG samples to use in forecast.
        ell_min : float
            Minimum ell value.
        ell_max : float
            Maximum ell value.
        ell_N : int
            Number of logarithmic ell bins.
        contamination : float
            Fraction of non-uniformity map that contaminates LSS signal.
        cosmology : ccl.Cosmology
            CCL cosmology object.
        clustering : bool, optional
            Whether to use the clustering signal in the forecast.
            Default is True.
        xcorr : bool, optional
            Whether to use the LBG x CMB lensing signal in the forecast.
            Default is True.
        lensing : bool, optional
            Whether to use CMB lensing auto-correlation in the forecast.
            Default is True.
        """
        # Check that all the years match in the mappers
        years = [mapper.year for mapper in mappers]
        if len(set(years)) > 1:
            raise ValueError("Mappers have different years")

        # Save params
        self.mappers = mappers
        self.ell_min = ell_min
        self.ell_max = ell_max
        self.ell_N = ell_N
        self.contamination = contamination
        self.cosmology = cosmology

        self.clustering = clustering
        self.xcorr = xcorr
        self.lensing = lensing

        self._products: list[dict[str, np.ndarray]] | None = None

        # Load the 3x2pt Fisher matrix
        file = data_dir / f"srd_y{years[0]}_3x2pt_fisher_matrix_all.npy"
        self.fisher_3x2pt = np.load(file)

    @property
    def products(self) -> list[dict[str, np.ndarray]]:
        """Return list of forecast products for each mapper in self.mappers.

        This is a lazy property that creates the products the first time they're
        requested.
        """
        if self._products is not None:
            return self._products

        self._products = [
            mapper.get_forecast_products(
                self.ell_min,
                self.ell_max,
                self.ell_N,
                cosmology=self.cosmology,
            )
            for mapper in self.mappers
        ]

        return self._products

    def create_signal(self) -> np.ndarray:
        """Create signal vector.

        Order is (here assuming 2 LBG samples):

        gg kg gg kg kk

        However some of these may be missing if self.clustering,
        self.xcorr, or self.lensing is False.

        Returns
        -------
        np.ndarray
            Signal vector
        """
        signal = []
        for product in self.products:
            if self.clustering:
                signal.append(product["Cgg"])
            if self.xcorr:
                signal.append(product["Ckg"])
        if self.lensing:
            signal.append(product["Ckk"])

        signal = np.concatenate(signal)

        return signal

    def create_cov(self) -> np.ndarray:
        """Create full covariance matrix.

        The structure of the covariance matrix is (here assuming 2 LBG samples):

            gg_gg gg_kg             gg_kk
            kg_gg kg_kg             kg_kk
                        gg_gg gg_kg gg_kk
                        kg_gg kg_kg kg_kk
            kk_gg kk_kg kk_gg kk_kg kk_kk

        However some of these blocks may be removed if self.clustering,
        self.xcorr, or self.lensing is False.

        Returns
        -------
        np.ndarray
            Full covariance matrix.
        """
        # Building block of zeros
        Z = np.zeros((self.ell_N, self.ell_N))

        # Constants used to determine indexing below
        a = 2 if (self.clustering and self.xcorr) else 1
        b = a - 1
        c = 1 if self.lensing else 0

        # Create initial list of zero blocks
        n = a * len(self.mappers) + c
        Cov = [[Z for j in range(n)] for i in range(n)]

        # Blocks for clustering and cross-correlation for each LBG sample
        for i, products in enumerate(self.products):
            if self.clustering:
                nonuniformity_var = np.diag(self.contamination * products["Cff"] ** 2)
                Cov[a * i][a * i] = products["Cov_gggg"] + nonuniformity_var
            if self.clustering and self.xcorr:
                Cov[a * i][a * i + b] = products["Cov_kggg"]
                Cov[a * i + b][a * i] = products["Cov_kggg"]
            if self.xcorr:
                Cov[a * i + b][a * i + b] = products["Cov_kgkg"]
            if self.lensing:
                if self.clustering:
                    Cov[-1][a * i] = products["Cov_kkgg"]
                    Cov[a * i][-1] = products["Cov_kkgg"]
                if self.xcorr:
                    Cov[-1][a * i + b] = products["Cov_kkkg"]
                    Cov[a * i + b][-1] = products["Cov_kkkg"]

        if self.lensing:
            # Finally add the lensing block
            Cov[-1][-1] = products["Cov_kkkk"]

        # Assemble matrix from blocks
        Cov = np.block(Cov)

        return Cov
