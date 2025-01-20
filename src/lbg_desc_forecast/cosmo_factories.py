"""Tools for cosmology analyses"""

from abc import ABC
from copy import deepcopy
from dataclasses import dataclass

import pyccl as ccl

default_cosmo_params = dict(
    Omega_M=0.3111,
    Omega_b=0.02242 / 0.6766**2,
    h=0.6766,
    n_s=0.9665,
    sigma8=0.8102,
    w0=-1,
    wa=0,
    m_nu=0.06,
    mu_0=0,
    sigma_0=0,
)


@dataclass
class CosmoFactory(ABC):
    """Base class for cosmology factories.

    Sub-classes must define sigma8, As, or equivalent.
    """

    Omega_M: float = 0.3111
    Omega_b: float = 0.02242 / 0.6766**2
    h: float = 0.6766
    n_s: float = 0.9665

    def copy(self) -> "CosmoFactory":
        """Return deep copy of self."""
        return deepcopy(self)

    @property
    def params(self) -> dict:
        return self.__dict__

    @property
    def _params_with_omega_c(self) -> dict:
        params = self.params.copy()
        params["Omega_c"] = params.pop("Omega_M") - params["Omega_b"]
        return params

    @property
    def cosmology(self) -> ccl.Cosmology:
        return ccl.Cosmology(**self._params_with_omega_c)


@dataclass
class MainCosmology(CosmoFactory):
    """Main factory for cosmology forecasts"""

    sigma8: float = 0.8102
    w0: float = -1
    wa: float = 0
    m_nu: float = 0.06
    mu_0: float = 0
    sigma_0: float = 0

    @property
    def cosmology(self) -> ccl.Cosmology:
        params = self._params_with_omega_c
        params["mg_parametrization"] = ccl.modified_gravity.MuSigmaMG(
            mu_0=params.pop("mu_0"),
            sigma_0=params.pop("sigma_0"),
        )

        return ccl.Cosmology(**params)


@dataclass
class sig8Cosmo(CosmoFactory): ...  # TODO: define this cosmology
