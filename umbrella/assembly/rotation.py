"""
Coordinate conventions:
    The (x, y, z) coordinates of vectors in 3d
    are assumed to be stored in rows of a numpy
    array with the following layout:
    R = (
            (x1, y1, z1),
            (x2, y2, z2),
            ...
        )
    This is chosen for consistency with other
    related python packages such as ase, etc.
    But one has to keep in mind that in algebraic
    formulas the vector are usually assumed as
    columns. Therefore the code will look slightly
    different from algebraic formulas. For instance
    rotation of coordinates R by a matrix M becomes:
        rotated-R = (M @ R.T).T = R @ M.T

    For consistency with the above, we will use
    the function `apply_rotation` to apply a
    rotation to a sequence of coordinates.
    A rotation is typically obtained from
    `get_rotation_...` functions.

Basis conventions:
    The (a, b, c) base vectors of a generic
    parallelepiped is assumed to be stored as
    abc = (
            (ax, ay, az),
            (bx, by, bz),
            (cx, cy, cz)
        )
    While e.g. ASE has no constraints on the
    base vectors, other packages such as LAMMPS,
    frued, etc. have certain constraints.
    We define the following conventions:

        Right-handedness:
            a . (b x c) > 0

        Prism:
            ay = az = bz = 0

        Transposed Prism:
            bx = cx = cy = 0

"""
from __future__ import annotations

import typing
from math import pi

import numpy as np

__all__ = [
    "apply_rotation",
    "get_rotation_about_axis",
    "get_rotation_to_prism_basis",
    "get_volume_of_parallelepiped",
    "get_prism_from_basis",
    "get_basis_from_prism",
    "is_right_handed",
    "is_prism",
]

TypeVector = tuple[float, float, float]
TypeBasis = typing.Union[tuple[TypeVector, TypeVector, TypeVector], np.ndarray]
TypePrism = tuple[float, float, float, float, float, float]
TypePositions = typing.Union[typing.Sequence[TypeVector], np.ndarray]


def apply_rotation(
    rot: np.ndarray,
    coordinates: TypePositions,
) -> np.ndarray:
    """
    Apply a rotation to a sequence of coordinates.

    parameters
    ----------
    rot: a 3x3 rotation matrix
    coordinates: a sequence of coordinates

    returns
    -------
    a sequence of rotated coordinates
    """
    return np.asarray(coordinates) @ rot.T


def get_rotation_about_axis(axis: TypeVector, angle: float) -> np.ndarray:
    """
    Get a rotation matrix about an axis by an angle.

    parameters
    ----------
    axis: a rotation axis
    angle: rotaion angle (Radian)

    returns
    -------
    a 3x3 rotation matrix

    """
    n = np.asarray(axis)
    n = n / np.linalg.norm(n)
    a = np.cos(angle / 2)
    b, c, d = -n * np.sin(angle / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )
    return rot


def get_rotation_to_prism_basis(abc: TypeBasis) -> np.ndarray:
    """
    Creates a rotation matrix which maps 3 right-handed
    generic base vectors a, b, c into new ones
    a', b', c' where
        a' = (x1,  0,  0)
        b' = (x2, y2,  0)
        c' = (x3, y3, z3)
    We will refer to the above as a "prism" basis; inspired
    by the LAMMPS documentation:
        https://docs.lammps.org/Howto_triclinic.html

    parameters
    ----------
    abc: (a, b, c) base vectors

    returns
    -------
    a 3x3 rotation matrix

    """

    # raise error if not right-handed
    # because the following algorithm
    # assumes right-handedness
    if not is_right_handed(abc):
        raise NotImplementedError("Only right-handed base vectors are supported")
    a, b, c = np.asarray(abc)
    a_ = a / np.linalg.norm(a)
    ab = np.cross(a, b)
    ab_ = ab / np.linalg.norm(ab)
    prism = np.zeros((3, 3))
    prism.flat[[0, 4, 8, 1, 2, 5]] = [
        np.linalg.norm(a),  # ax -> 0
        np.linalg.norm(np.cross(a_, b)),  # by -> 4
        np.dot(ab_, c),  # cz -> 8
        np.dot(a_, b),  # bx -> 1
        np.dot(a_, c),  # cx -> 2
        np.dot(np.cross(ab_, a_), c),  # cy -> 5
    ]
    recip = np.array([np.cross(b, c), np.cross(c, a), np.cross(a, b)]) / np.linalg.det(
        prism
    )
    rot = prism @ recip
    return rot


def get_volume_of_parallelepiped(
    abc: TypeBasis,
) -> float:
    """
    Get the volume of a parallelepiped defined by 3 vectors.
    The volume is positive only if the vectors are right-handed.

    parameters
    ----------
    abc: (a, b, c) the base vectors

    returns
    -------
    the volume of the parallelepiped

    """
    a, b, c = np.asarray(abc)
    return np.dot(a, np.cross(b, c))


def is_right_handed(
    abc: TypeBasis,
) -> bool:
    """
    Check if the base vectors are right-handed.

    parameters
    ----------
    abc: (a, b, c) the base vectors

    returns
    -------
    True if right-handed, False otherwise

    """
    return get_volume_of_parallelepiped(abc) > 0


def is_prism(abc: TypeBasis) -> bool:
    """
    Check if the base vectors define a prism.

    parameters
    ----------
    abc: (a, b, c) the base vectors

    returns
    -------
    True if the base vectors define a prism, False otherwise

    """
    abc = np.asarray(abc)
    return np.allclose(abc.flat[[1, 2, 5]], 0)


def get_prism_from_basis(
    abc: TypeBasis,
) -> TypePrism:
    """
    Get the parameters of a prism defined by 3 base vectors.

    parameters
    ----------
    abc: (a, b, c) the base vectors

    returns
    -------
    lx, ly, lz, xy, xz, yz
    """
    assert is_prism(abc)
    abc = np.asarray(abc)
    return abc.flat[[0, 4, 8, 3, 6, 7]]


def get_basis_from_prism(
    param: TypePrism,
) -> TypeBasis:
    """
    Get the base vectors of a prism defined by its parameters.

    parameters
    ----------
    lx, ly, lz, xy, xz, yz

    returns
    -------
    (a, b, c) the base vectors
    """
    lx, ly, lz, xy, xz, yz = param
    a = (lx, 0, 0)
    b = (xy, ly, 0)
    c = (xz, yz, lz)
    return a, b, c


def test_rotation_about_axis() -> bool:
    axis = (0.0, 0.0, 1.0)
    angle = pi / 2
    rot = get_rotation_about_axis(axis, angle)
    xyz_in = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    xyz_out = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    # one at a time
    for a, b in zip(xyz_in, xyz_out):
        bb = apply_rotation(rot, a)
        assert np.allclose(bb, b)
    # collective
    out = apply_rotation(rot, xyz_in)
    assert np.allclose(out, xyz_out)
    return True


def test_rotation_to_prism_basis() -> bool:
    for _ in range(100):
        basis = np.random.uniform(size=(3, 3))
        basis.flat[[1, 2, 5]] = 0
        assert is_right_handed(basis)
        assert is_prism(basis)
        rot = get_rotation_to_prism_basis(basis)
        new = apply_rotation(rot, basis)
        assert np.allclose(new.flat[[1, 2, 5]], 0)
        assert np.allclose(basis @ basis.T, new @ new.T)
    return True


if __name__ == "__main__":
    test_rotation_about_axis()
    test_rotation_to_prism_basis()
