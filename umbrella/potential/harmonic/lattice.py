"""
This module provides a calculator for harmonic lattices.
Displacements from a reference lattice are used to calculate
the energy and forces.

"""

import typing as typing

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic
from matscipy.neighbours import neighbour_list
from matscipy.numpy_tricks import mabincount

__all__ = [
    "HarmonicLatticePotential",
    "HarmonicLatticeCalculator",
    "get_harmonic_lattice",
]


class HarmonicLatticePotential:
    def __init__(
        self,
        pairs: typing.Sequence[tuple[int, int]],
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
        self.kij = k_spring

    def get_results(
        self,
        displacemens: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        Dij = displacemens[self.i] - displacemens[self.j]
        energy = 0.5 * np.sum(self.kij * Dij**2)

        # forces
        fij = -self.kij * Dij
        N = displacemens.shape[0]
        forces = mabincount(self.i, fij, N) - mabincount(self.j, fij, N)

        # TODO: implement the virial
        return energy, forces


class HarmonicLatticeCalculator(Calculator):
    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(
        self,
        potentials: HarmonicLatticePotential
        | typing.Sequence[HarmonicLatticePotential],  # noqa: W503
        ref_pos: np.ndarray,
    ) -> None:
        Calculator.__init__(self)
        if isinstance(potentials, HarmonicLatticePotential):
            potentials = (potentials,)
        self.potentials = potentials
        self.ref_pos = ref_pos.copy()

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        # No typing maybe ok since it is used internally by ASE
        Calculator.calculate(self, atoms, properties, system_changes)
        vec = self.atoms.positions - self.ref_pos
        vec, _ = find_mic(vec, cell=self.atoms.cell, pbc=self.atoms.pbc)

        # Initialize the results
        energy = 0.0
        forces = np.zeros((len(self.atoms), 3))

        # Loop over the potentials
        for pot in self.potentials:
            e, f = pot.get_results(vec)
            energy += e
            forces += f

        # make sure forces add up to zero
        assert np.allclose(forces.sum(axis=0), 0.0)

        # Store the results
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

        # TODO: implement the stress tensor


def get_harmonic_lattice(
    atoms: Atoms,
    cutoff: float,
    k_springs: typing.Sequence[float],
    prec: float,
) -> tuple[HarmonicLatticePotential, ...]:
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
    tuple[HarmonicLatticePotential, ...]
        The list of potentials.

    """

    i, j, d, D = neighbour_list("ijdD", atoms, cutoff)
    mask = i < j
    i, j, d, D = i[mask], j[mask], d[mask], D[mask]
    arg = np.argsort(d)
    i, j, d, D = i[arg], j[arg], d[arg], D[arg]
    steps = (np.argwhere(np.diff(d) > prec) + 1).flatten()

    pieces = zip(
        np.split(i, steps),
        np.split(j, steps),
        np.split(d, steps),
        np.split(D, steps, axis=0),
        k_springs,
    )

    potentials = tuple(
        HarmonicLatticePotential(
            pairs=[(a, b) for a, b in zip(ii, jj)],
            k_spring=kk,
        )
        for ii, jj, dd, DD, kk in pieces
    )
    return potentials


def test_pair(verbose: bool = False) -> None:
    # A pair of atoms
    atoms = Atoms(
        symbols=["H", "H"],
        positions=[[0, 0, 0], [1.0, 0, 0]],
    )

    pot = HarmonicLatticePotential(
        pairs=[(0, 1)],
        k_spring=1.0,
    )

    calc = HarmonicLatticeCalculator(
        potentials=pot,
        ref_pos=atoms.positions,
    )

    #
    atoms.positions[1, 0] += 0.1
    atoms.calc = calc
    e = atoms.get_potential_energy()
    assert np.isclose(e, 0.5 * 0.1**2), e
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
    potentials = get_harmonic_lattice(atoms, 2.0, [1.0, 0.3], 0.01)

    # - test number of nearest neighbor pairs
    assert potentials[0].i.size / len(atoms) == 6 / 2

    # - test number of next nearest neighbor pairs
    assert potentials[1].i.size / len(atoms) == 12 / 2

    # bcc lattice
    atoms = bulk("X", "bcc", 1.0, cubic=True).repeat(4)
    potentials = get_harmonic_lattice(atoms, 2.0, [1.0, 0.3], 0.01)

    # - test number of nearest neighbor pairs
    assert potentials[0].i.size / len(atoms) == 8 / 2

    # - test number of next nearest neighbor pairs
    assert potentials[1].i.size / len(atoms) == 6 / 2


def _test_bulk(lattice: str) -> bool:
    from ase.build import bulk

    atoms = bulk("X", lattice, 1.0, cubic=True).repeat(4)
    potentials = get_harmonic_lattice(atoms, 2.0, [1.0, 0.3], 0.01)

    calc = HarmonicLatticeCalculator(
        potentials=potentials,
        ref_pos=atoms.positions,
    )

    # test the forces
    atoms.rattle(0.1)
    atoms.calc = calc
    f = atoms.get_forces()
    fn = calc.calculate_numerical_forces(atoms)
    result1 = np.allclose(f, fn, atol=1e-6)
    if not result1:
        print(f"lattice: {lattice}")
        print("Forces do not match (test_bulk)")
        print(f"Analytical forces: \n {f}")
        print(f"Numerical forces: \n {fn}")

    # test translational invariance
    atoms.positions -= 3.0
    atoms.wrap()
    f_trans = atoms.get_forces()
    result2 = np.allclose(f, f_trans)
    return result1 and result2


def test_bulk() -> None:
    for lattice in ["sc", "bcc", "fcc"]:
        assert _test_bulk(lattice)


if __name__ == "__main__":
    test_pair()
    test_get_harmonic_potentials()
    test_bulk()
