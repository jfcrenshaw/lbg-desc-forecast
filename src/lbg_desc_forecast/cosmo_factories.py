"""Tools for cosmology analyses"""

from abc import ABC
from copy import deepcopy
from dataclasses import dataclass

import pyccl as ccl


@dataclass
class CosmoFactory(ABC):
    """Base class for cosmology factories.

    Sub-classes must define sigma8, As, or equivalent.
    """

    Omega_M: float = 0.3156
    Omega_b: float = 0.0491685
    h: float = 0.6727
    n_s: float = 0.9645

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

    sigma8: float = 0.831
    w0: float = -1
    wa: float = 0
    m_nu: float = 0.06

    @property
    def cosmology(self) -> ccl.Cosmology:
        return ccl.Cosmology(
            **self._params_with_omega_c,
            mass_split="single",
            extra_parameters=dict(camb=dict(dark_energy_model="ppf")),
        )


@dataclass
class sig8Cosmo(CosmoFactory): ...  # TODO: define this cosmology
