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

from umbrella.assembly.rotation import PrismRotation as _PrismRotation


def get_freud_box(
    atoms: _Atoms, wrap: bool = False
) -> tuple[_freud.box.Box, _np.ndarray]:
    """
    Get 'freud' Box object from atomic structure.

    Args:
        atoms (ase.Atoms): Atomic structure; must be periodic.
        wrap (bool, optional): Whether to wrap atomic positions to the box.

    Returns:
        tuple[freud.box.Box, numpy.ndarray]: 'freud' Box object and corresponding
            atomic positions.
    """

    assert all(atoms.get_pbc()), "Atomic structure must be periodic."

    # frued Box is defined by Lx, Ly, Lz, xy, xz, yz
    # Lx, Ly, Lz are the lengths of the box edges
    # PrismRotation rotates the cell such that upper
    # triangular part of the array is zero.
    prism_cell = _PrismRotation(atoms.cell)(atoms.cell)
    assert _np.allclose(prism_cell.flat[[1, 2, 5]], 0)

    Lx, Ly, Lz, xyLy, xzLz, yzLz = prism_cell.flat[[0, 4, 8, 3, 6, 7]]
    xy = xyLy / Ly
    xz = xzLz / Lz
    yz = yzLz / Lz
    box = _freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz)

    # Since the box is rotated, the atomic positions
    # must be rotated as well. Although, scaled positions
    # are not affected by the rotation.
    # Fortunately, 'freud' Box object can be created from
    # scaled positions (which are called fractional in frued).
    scaled_positions = atoms.get_scaled_positions(wrap=wrap)
    positions = box.make_absolute(scaled_positions)
    return box, positions