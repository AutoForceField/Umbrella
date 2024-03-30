import typing as typing

import numpy as np
from ase.atoms import Atoms
from ase.build import bulk, make_supercell
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic
from matscipy.neighbours import neighbour_list
from matscipy.numpy_tricks import mabincount

__all__ = [
    "HarmonicSolidCalculator",
    "find_bonds",
]


class HarmonicSolidCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress", "free_energy"]

    def __init__(
        self,
        bonds: typing.Sequence[tuple[int, int]],
        lengths: float | typing.Sequence[float],
        k_springs: float | typing.Sequence[float],
        **kwargs: typing.Any,
    ) -> None:
        """
        Parameters
        ----------
        bonds : Sequence[tuple[int, int]]
            The list of bonds.

        lengths : float | Sequence[float]
            The equilibrium bond length.

        k_springs : float | Sequence[float]
            The spring constant (eV/A^2).

        """
        Calculator.__init__(self, **kwargs)
        self.i, self.j = np.array(bonds).T
        self.lij = np.array(lengths)
        self.kij = np.array(k_springs)

        # Check the input
        if any(self.i == self.j):
            raise ValueError("The bond list should not contain self-interactions")

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        # No typing maybe ok since it is used internally by ASE
        Calculator.calculate(self, atoms, properties, system_changes)

        # find_mic returns the mic vectors (D) and the mic distances (d)
        Dij, dij = find_mic(
            self.atoms.positions[self.i] - self.atoms.positions[self.j],
            cell=self.atoms.cell,
            pbc=self.atoms.pbc,
        )

        # energy
        xij = dij - self.lij
        energy = 0.5 * np.sum(self.kij * xij**2)
        self.results["energy"] = energy

        # forces
        fij = -(self.kij * xij / dij)[:, None] * Dij
        N = len(self.atoms)
        forces = mabincount(self.i, fij, N) - mabincount(self.j, fij, N)
        self.results["forces"] = forces

        # stress
        if all(atoms.pbc):
            stress = -np.einsum("ij,ik->jk", fij, Dij) / atoms.get_volume()
            self.results["stress"] = stress
        self.results["free_energy"] = energy


def find_bonds(atoms: Atoms, length: float, toll: float) -> tuple[tuple[int, int], ...]:
    """
    Find the bonds in the system.

    Parameters
    ----------
    atoms : Atoms
        The system.

    length : float
        The bond length.

    toll : float
        The tolerance to consider two atoms bonded.

    Returns
    -------
    tuple[tuple[int, int], ...]
        The list of bonds.

    """
    i, j, d = neighbour_list("ijd", atoms, length + toll)
    mask = d > length - toll

    assert np.all(
        i[mask] != j[mask]
    ), "The bond list contains self-interactions, probably the pbc cell is too small."
    return tuple(zip(i[mask], j[mask]))


def test_pair(verbose: bool = False) -> None:
    # A pair of atoms
    atoms = Atoms(
        symbols=["H", "H"],
        positions=[[0, 0, 0], [1.1, 0, 0]],
    )

    calc = HarmonicSolidCalculator(
        atoms=atoms,
        bonds=[(0, 1)],
        lengths=[1.0],
        k_springs=[1.0],
    )

    atoms.calc = calc
    f = atoms.get_forces()
    fn = calc.calculate_numerical_forces(atoms)
    result = np.allclose(f, fn, atol=1e-6)
    if not result:
        print("Forces do not match (test_pair)")
        print(f"Analytical forces: \n {f}")
        print(f"Numerical forces: \n {fn}")

    assert result


def test_bulk() -> None:
    from ase.build import bulk

    atoms = bulk("X", "sc", 1.0, cubic=True).repeat(4)
    calc = HarmonicSolidCalculator(
        atoms=atoms,
        bonds=find_bonds(atoms, 1.0, 0.05),
        lengths=[1.0],
        k_springs=[1.0],
    )
    atoms.calc = calc

    # test the forces
    f = atoms.get_forces()
    fn = calc.calculate_numerical_forces(atoms)
    result = np.allclose(f, fn, atol=1e-6)
    if not result:
        print("Forces do not match (test_bulk)")
        print(f"Analytical forces: \n {f}")
        print(f"Numerical forces: \n {fn}")
    assert result

    # test the stress
    s = atoms.get_stress()
    sn = calc.calculate_numerical_stress(atoms)
    result = np.allclose(s, sn, atol=1e-6)
    if not result:
        print("Stress do not match (test_bulk)")
        print(f"Analytical stress: \n {s}")
        print(f"Numerical stress: \n {sn}")
    assert result


def get_simple_cubic(
    atom: str,
    a: float,
    k1_spring: float,
    k2_spring: float,
    super_cell: int | tuple[int, int, int] | np.ndarray,
) -> tuple[Atoms, Atoms, Calculator]:
    prim = bulk(
        atom,
        "sc",
        a=a,
        cubic=True,
    )

    # make a transformation matrix
    if isinstance(super_cell, int):
        trans = np.eye(3) * super_cell
    else:
        super_cell = np.array(super_cell)
        if super_cell.size == 3:
            trans = np.diag(super_cell)
        elif super_cell.size == 9:
            trans = super_cell
        else:
            raise ValueError("super_cell must be an int, a 3-tuple, or a 3x3 array")

    atoms = make_supercell(prim, trans)

    # nearest neighbor bonds
    nn_bonds = find_bonds(
        atoms,
        a,
        toll=0.05,
    )

    # next nearest neighbor bonds
    b = a * np.sqrt(2)
    nnn_bonds = find_bonds(
        atoms,
        b,
        toll=0.05,
    )

    bonds = (*nn_bonds, *nnn_bonds)
    lengths = len(nn_bonds) * [a] + len(nnn_bonds) * [b]
    k_springs = len(nn_bonds) * [k1_spring] + len(nnn_bonds) * [k2_spring]
    calc = HarmonicSolidCalculator(
        bonds=bonds,
        lengths=lengths,
        k_springs=k_springs,
    )

    assert len(nn_bonds) / len(atoms) == 6
    assert len(nnn_bonds) / len(atoms) == 12

    return prim, atoms, calc


def get_bcc(
    atom: str,
    a: float,
    k1_spring: float,
    k2_spring: float,
    super_cell: int | tuple[int, int, int] | np.ndarray,
) -> tuple[Atoms, Atoms, Calculator]:
    prim = bulk(
        atom,
        "bcc",
        a=a,
        cubic=True,
    )

    # make a transformation matrix
    if isinstance(super_cell, int):
        trans = np.eye(3) * super_cell
    else:
        super_cell = np.array(super_cell)
        if super_cell.size == 3:
            trans = np.diag(super_cell)
        elif super_cell.size == 9:
            trans = super_cell
        else:
            raise ValueError("super_cell must be an int, a 3-tuple, or a 3x3 array")

    atoms = make_supercell(prim, trans)

    # nearest neighbor bonds
    nn_distance = a * np.sqrt(3) / 2
    nn_bonds = find_bonds(
        atoms,
        nn_distance,
        toll=0.05,
    )

    # next nearest neighbor bonds
    nnn_distance = a
    nnn_bonds = find_bonds(
        atoms,
        nnn_distance,
        toll=0.05,
    )

    bonds = (*nn_bonds, *nnn_bonds)
    lengths = len(nn_bonds) * [nn_distance] + len(nnn_bonds) * [nnn_distance]
    k_springs = len(nn_bonds) * [k1_spring] + len(nnn_bonds) * [k2_spring]
    calc = HarmonicSolidCalculator(
        bonds=bonds,
        lengths=lengths,
        k_springs=k_springs,
    )

    assert len(nn_bonds) / len(atoms) == 8
    assert len(nnn_bonds) / len(atoms) == 6

    return prim, atoms, calc


def test_lattices():
    prim, atoms, calc = get_simple_cubic("X", 4.6, 0.3, 0.1, 4)
    prim, atoms, calc = get_bcc("X", 4.6, 0.3, 0.1, 4)


if __name__ == "__main__":
    test_pair()
    test_bulk()
    test_lattices()
