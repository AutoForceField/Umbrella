"""
Utility functions for utilizing 'frued' package, available at:
    https://github.com/glotzerlab/freud
"""
import freud

import umbrella.assembly.rotation as rotation

__all__ = [
    "get_freud_box",
    "get_frued_system",
]


def get_freud_box(
    basis: rotation.TypeBasis,
) -> freud.box.Box:
    """
    Convert a basis to a freud Box object.
    The basis must be oriented in a prismic parallelepiped.
    See 'assembly/rotation.py' for more information.

    Parameters
    ----------
    basis : (a, b, c) base vectors of a prismic parallelepiped.

    Returns
    -------
    box : freud.box.Box object.

    """

    if not rotation.is_prism(basis):
        raise ValueError("basis must be prismic.")

    Lx, Ly, Lz, xyLy, xzLz, yzLz = rotation.get_prism_from_basis(basis)
    xy = xyLy / Ly
    xz = xzLz / Lz
    yz = yzLz / Lz
    box = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz)
    return box


def get_frued_system(
    basis: rotation.TypeBasis,
    positions: rotation.TypePositions,
    rotate_to_prism: bool = False,
) -> tuple[freud.box.Box, rotation.TypePositions]:
    """
    Create a freud Box object and positions from a basis and positions.

    Parameters
    ----------
    basis : (a, b, c) base vectors of a prismic parallelepiped.
    positions : (N, 3) array of particle positions.
    rotate_to_prism : bool, optional
        If True, rotate the basis/positions to the prismic parallelepiped.

    Returns
    -------
    box : freud.box.Box object.
    positions : (N, 3) array of particle positions.

    """
    if rotate_to_prism and not rotation.is_prism(basis):
        rot = rotation.get_rotation_to_prism_basis(basis)
        basis = rotation.apply_rotation(rot, basis)
        positions = rotation.apply_rotation(rot, positions)
    box = get_freud_box(basis)
    return box, positions
