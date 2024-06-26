"""
This module provides a simple harmonic solid calculator for ASE.
The potential is defined by a list of pairs of atoms and their
equilibrium bond length and spring constant.
Note that this is different from the  HarmonicLatticeCalculator
which uses a reference lattice structure.

A utility function get_harmonic_solid is provided to extract the
harmonic potentials from the system.

"""
import typing as typing

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic
from matscipy.neighbours import neighbour_list
from matscipy.numpy_tricks import mabincount

__all__ = [
    "HarmonicSolidPotential",
    "HarmonicSolidCalculator",
    "get_harmonic_solid",
]


class HarmonicSolidPotential:
    def __init__(
        self,
        pairs: typing.Sequence[tuple[int, int]],
        bond_length: float,
        k_spring: float,
    ) -> None:
        # Check the input
        for i, j in pairs:
            if i == j:
                raise ValueError("The bond list should not contain self-interactions")
            if (j, i) in pairs:
                raise ValueError(
                    "The bond list should not contain repeated interactions"
                )
        self.i, self.j = np.array(pairs).T
        self.lij = bond_length
        self.kij = k_spring

    def get_results(
        self,
        atoms: Atoms,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        # find_mic returns the mic vectors (D) and the mic distances (d)
        Dij, dij = find_mic(
            atoms.positions[self.i] - atoms.positions[self.j],
            cell=atoms.cell,
            pbc=atoms.pbc,
        )

        # energy
        xij = dij - self.lij
        energy = 0.5 * np.sum(self.kij * xij**2)

        # forces
        fij = -(self.kij * xij / dij)[:, None] * Dij
        N = len(atoms)
        forces = mabincount(self.i, fij, N) - mabincount(self.j, fij, N)
        virial = -np.einsum("ij,ik->jk", fij, Dij)
        return energy, forces, virial


class HarmonicSolidCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress", "free_energy"]

    def __init__(
        self,
        potentials: HarmonicSolidPotential | typing.Sequence[HarmonicSolidPotential],
    ) -> None:
        Calculator.__init__(self)
        if isinstance(potentials, HarmonicSolidPotential):
            potentials = (potentials,)
        self.potentials = potentials

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        # No typing maybe ok since it is used internally by ASE
        Calculator.calculate(self, atoms, properties, system_changes)

        # Initialize the results
        energy = 0.0
        forces = np.zeros((len(self.atoms), 3))
        virial = np.zeros((3, 3))

        # Loop over the potentials
        for pot in self.potentials:
            e, f, v = pot.get_results(self.atoms)
            energy += e
            forces += f
            virial += v

        # Store the results
        self.results["energy"] = energy
        self.results["forces"] = forces
        if all(atoms.pbc):
            self.results["stress"] = virial / atoms.get_volume()
        self.results["free_energy"] = energy


def get_harmonic_solid(
    atoms: Atoms,
    cutoff: float,
    k_springs: typing.Sequence[float],
    prec: float,
) -> tuple[HarmonicSolidPotential, ...]:
    """
    Extract the harmonic potentials from the system.

    Parameters
    ----------
    atoms : Atoms
        The system.

    cutoff : float
        The cutoff distance to consider interactions.

    k_springs : typing.Sequence[float]
        The spring constants for [1st, 2nd, ...] neighbors.

    Returns
    -------
    tuple[HarmonicSolidPotential, ...]
        The list of potentials.

    """

    i, j, d = neighbour_list("ijd", atoms, cutoff)
    mask = i < j
    i, j, d = i[mask], j[mask], d[mask]
    arg = np.argsort(d)
    i, j, d = i[arg], j[arg], d[arg]
    steps = (np.argwhere(np.diff(d) > prec) + 1).flatten()

    pieces = zip(
        np.split(i, steps),
        np.split(j, steps),
        np.split(d, steps),
        k_springs,
    )

    potentials = tuple(
        HarmonicSolidPotential(
            pairs=[(a, b) for a, b in zip(ii, jj)],
            bond_length=dd,
            k_spring=kk,
        )
        for ii, jj, dd, kk in pieces
    )
    return potentials


def test_pair(verbose: bool = False) -> None:
    # A pair of atoms
    atoms = Atoms(
        symbols=["H", "H"],
        positions=[[0, 0, 0], [1.1, 0, 0]],
    )

    pot = HarmonicSolidPotential(
        pairs=[(0, 1)],
        bond_length=1.0,
        k_spring=1.0,
    )

    calc = HarmonicSolidCalculator(
        potentials=pot,
    )

    atoms.calc = calc
    assert np.isclose(atoms.get_potential_energy(), 0.5 * 0.1**2)
    f = atoms.get_forces()
    fn = calc.calculate_numerical_forces(atoms)
    result = np.allclose(f, fn, atol=1e-6)
    if not result:
        print("Forces do not match (test_pair)")
        print(f"Analytical forces: \n {f}")
        print(f"Numerical forces: \n {fn}")

    assert result


def test_get_harmonic_potentials() -> None:
    from ase.build import bulk

    # sc lattice
    atoms = bulk("X", "sc", 1.0, cubic=True).repeat(4)
    potentials = get_harmonic_solid(atoms, 2.0, [1.0, 0.3], 0.01)

    # - test number of nearest neighbor pairs
    assert potentials[0].i.size / len(atoms) == 6 / 2

    # - test number of next nearest neighbor pairs
    assert potentials[1].i.size / len(atoms) == 12 / 2

    # bcc lattice
    atoms = bulk("X", "bcc", 1.0, cubic=True).repeat(4)
    potentials = get_harmonic_solid(atoms, 2.0, [1.0, 0.3], 0.01)

    # - test number of nearest neighbor pairs
    assert potentials[0].i.size / len(atoms) == 8 / 2

    # - test number of next nearest neighbor pairs
    assert potentials[1].i.size / len(atoms) == 6 / 2


def _test_bulk(lattice: str) -> bool:
    from ase.build import bulk

    atoms = bulk("X", lattice, 1.0, cubic=True).repeat(4)
    potentials = get_harmonic_solid(atoms, 2.0, [1.0, 0.3], 0.01)

    calc = HarmonicSolidCalculator(
        potentials=potentials,
    )

    # test the forces
    atoms.rattle(0.1)
    atoms.calc = calc
    f = atoms.get_forces()
    fn = calc.calculate_numerical_forces(atoms)
    result1 = np.allclose(f, fn, atol=1e-6)
    if not result1:
        print("Forces do not match (test_bulk)")
        print(f"Analytical forces: \n {f}")
        print(f"Numerical forces: \n {fn}")

    # test the stress
    s = atoms.get_stress()
    sn = calc.calculate_numerical_stress(atoms)
    result2 = np.allclose(s, sn, atol=1e-6)
    if not result2:
        print("Stress do not match (test_bulk)")
        print(f"Analytical stress: \n {s}")
        print(f"Numerical stress: \n {sn}")
    return result1 and result2


def test_bulk() -> None:
    for lattice in ["sc", "bcc", "fcc"]:
        assert _test_bulk(lattice)


if __name__ == "__main__":
    test_pair()
    test_get_harmonic_potentials()
    test_bulk()
