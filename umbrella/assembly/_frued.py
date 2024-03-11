"""
Utility functions for utilizing 'frued' package, available at:
    https://github.com/glotzerlab/freud

For utilizing the 'frued' package, atomic structures must be converted to
'freud' Box objects. This module provides functions to convert atomic structures
to 'freud' Box objects.
"""
import freud as _freud
import numpy as _np
from ase.atoms import Atoms as _Atoms
from ase.cell import Cell as _Cell

import umbrella.assembly.rotation as _rotation  # import PrismRotation as _PrismRotation


def get_freud_box(
    cell: _np.ndarray | _Cell,
) -> _freud.box.Box:
    """
    Get 'freud' Box object from an ASE cell object or arbitrary 3x3 array.

    Args:
        cell (numpy.ndarray or ase.Cell): ASE cell object or arbitrary 3x3 array.

    Returns:
        freud.box.Box: 'freud' Box object.
    """

    # frued Box is defined by Lx, Ly, Lz, xy, xz, yz
    # Lx, Ly, Lz are the lengths of the box edges
    # PrismRotation rotates the cell such that upper
    # triangular part of the array is zero.
    rot = _rotation.get_rotation_to_prism_basis(cell)
    prism_cell = _rotation.apply_rotation(cell, rot)
    assert _np.allclose(prism_cell.flat[[1, 2, 5]], 0)

    Lx, Ly, Lz, xyLy, xzLz, yzLz = prism_cell.flat[[0, 4, 8, 3, 6, 7]]
    xy = xyLy / Ly
    xz = xzLz / Lz
    yz = yzLz / Lz
    box = _freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz)
    return box


def get_freud_system(
    atoms: _Atoms, wrap: bool = False
) -> tuple[_freud.box.Box, _np.ndarray]:
    """
    Get 'freud' system (Box, positions) from ase ASE atomic structure.

    Args:
        atoms (ase.Atoms): Atomic structure; must be periodic.
        wrap (bool, optional): Whether to wrap atomic positions to the box.

    Returns:
        tuple[freud.box.Box, numpy.ndarray]: 'freud' Box object and corresponding
            atomic positions.
    """

    assert all(atoms.get_pbc()), "Atomic structure must be periodic."

    box = get_freud_box(atoms.cell)

    # Since the box maybe rotated, the atomic positions
    # must be rotated as well. Although, scaled positions
    # are not affected by the rotation.
    # Fortunately, 'freud' Box object can be created from
    # scaled positions (which are called fractional in frued).
    scaled_positions = atoms.get_scaled_positions(wrap=wrap)
    positions = box.make_absolute(scaled_positions)
    return box, positions
