"""
Rubik's Cube Emulator (3x3) — Facelet and Cubie representations.

Provides:
  - RubiksCube: facelet-based 3x3 cube with apply(move), is_solved(), load_from_dict(), to_dict(), as_cubie_cube()
  - CubieCube: cubie-based representation with corner/edge permutation + orientation
  - Coordinate encoding/decoding utilities for building solvers
  - Move tables (_CORNER_PERM, _CORNER_ORIENT, _EDGE_PERM, _EDGE_ORIENT)
  - MOVE_NAMES, P2_MOVE_NAMES, FACE_COLORS constants

Corner numbering (Kociemba standard):
  0:URF  1:UFL  2:ULB  3:UBR  4:DFR  5:DLF  6:DBL  7:DRB

Edge numbering (Kociemba standard):
  0:UR  1:UF  2:UL  3:UB  4:DR  5:DF  6:DL  7:DB  8:FR  9:FL  10:BL  11:BR
"""

import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACE_COLORS = {'U': 'W', 'D': 'Y', 'F': 'G', 'B': 'B', 'L': 'O', 'R': 'R'}

# ---------------------------------------------------------------------------
# Move definitions
# ---------------------------------------------------------------------------
# Convention: new[i] = old[perm[i]]  (pull convention)
# new_co[i] = (old_co[perm[i]] + orient[i]) % 3
# new_eo[i] = (old_eo[perm[i]] + orient[i]) % 2

_CORNER_PERM = {
    'U': [1, 2, 3, 0, 4, 5, 6, 7],
    'R': [4, 1, 2, 0, 7, 5, 6, 3],
    'F': [1, 5, 2, 3, 0, 4, 6, 7],
    'D': [0, 1, 2, 3, 7, 4, 5, 6],
    'L': [0, 5, 1, 3, 4, 6, 2, 7],
    'B': [0, 1, 3, 7, 4, 5, 2, 6],
}

_CORNER_ORIENT = {
    'U': [0, 0, 0, 0, 0, 0, 0, 0],
    'R': [2, 0, 0, 1, 1, 0, 0, 2],
    'F': [1, 2, 0, 0, 2, 1, 0, 0],
    'D': [0, 0, 0, 0, 0, 0, 0, 0],
    'L': [0, 1, 2, 0, 0, 2, 1, 0],
    'B': [0, 0, 1, 2, 0, 0, 2, 1],
}

_EDGE_PERM = {
    'U': [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11],
    'R': [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0],
    'F': [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
    'D': [0, 1, 2, 3, 7, 4, 5, 6, 8, 9, 10, 11],
    'L': [0, 1, 9, 3, 4, 5, 10, 7, 8, 6, 2, 11],
    'B': [0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7],
}

_EDGE_ORIENT = {
    'U': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'F': [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    'D': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'B': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
}

MOVE_NAMES = ['U', "U'", 'U2', 'R', "R'", 'R2', 'F', "F'", 'F2',
              'D', "D'", 'D2', 'L', "L'", 'L2', 'B', "B'", 'B2']
P2_MOVE_NAMES = ['U', "U'", 'U2', 'D', "D'", 'D2', 'R2', 'L2', 'F2', 'B2']


# ---------------------------------------------------------------------------
# Core Emulator Logic (facelet based, for display)
# ---------------------------------------------------------------------------
class RubiksCube:
    """Facelet-based 3x3 Rubik's Cube.

    Each face is a 3x3 numpy array of single-character color strings.
    Supports applying moves, checking solved state, and converting to CubieCube.
    """

    def __init__(self) -> None:
        self.faces: Dict[str, np.ndarray] = {
            f: np.full((3, 3), c, dtype='U1') for f, c in FACE_COLORS.items()
        }

    def _cycle_rows(self, a: str, ra: int, b: str, rb: int,
                    c: str, rc: int, d: str, rd: int) -> None:
        f = self.faces
        tmp = f[d][rd].copy()
        f[d][rd], f[c][rc], f[b][rb], f[a][ra] = (
            f[c][rc], f[b][rb], f[a][ra], tmp
        )

    def apply(self, move: str) -> None:
        """Apply a move string (e.g. 'R', "R'", 'R2') to the cube."""
        m = move.strip()
        face = m[0]
        suffix = m[1:]
        times = 3 if suffix == "'" else (2 if suffix == "2" else 1)
        for _ in range(times):
            if face == 'U':
                self.faces['U'] = np.rot90(self.faces['U'], k=1)
                self._cycle_rows('L', 0, 'F', 0, 'R', 0, 'B', 0)
            elif face == 'D':
                self.faces['D'] = np.rot90(self.faces['D'], k=1)
                self._cycle_rows('F', 2, 'L', 2, 'B', 2, 'R', 2)
            elif face == 'F':
                self.faces['F'] = np.rot90(self.faces['F'], k=-1)
                u2 = self.faces['U'][2].copy()
                rc0 = self.faces['R'][:, 0].copy()
                d0 = self.faces['D'][0].copy()
                lc2 = self.faces['L'][:, 2].copy()
                self.faces['U'][2] = lc2[::-1]
                self.faces['R'][:, 0] = u2
                self.faces['D'][0] = rc0[::-1]
                self.faces['L'][:, 2] = d0
            elif face == 'B':
                self.faces['B'] = np.rot90(self.faces['B'], k=-1)
                u0 = self.faces['U'][0].copy()
                lc0 = self.faces['L'][:, 0].copy()
                d2 = self.faces['D'][2].copy()
                rc2 = self.faces['R'][:, 2].copy()
                self.faces['L'][:, 0] = u0[::-1]
                self.faces['D'][2] = lc0
                self.faces['R'][:, 2] = d2[::-1]
                self.faces['U'][0] = rc2
            elif face == 'R':
                self.faces['R'] = np.rot90(self.faces['R'], k=-1)
                fc2 = self.faces['F'][:, 2].copy()
                uc2 = self.faces['U'][:, 2].copy()
                bc0 = self.faces['B'][:, 0].copy()
                dc2 = self.faces['D'][:, 2].copy()
                self.faces['U'][:, 2] = fc2
                self.faces['B'][:, 0] = uc2[::-1]
                self.faces['D'][:, 2] = bc0[::-1]
                self.faces['F'][:, 2] = dc2
            elif face == 'L':
                self.faces['L'] = np.rot90(self.faces['L'], k=1)
                fc0 = self.faces['F'][:, 0].copy()
                uc0 = self.faces['U'][:, 0].copy()
                bc2 = self.faces['B'][:, 2].copy()
                dc0 = self.faces['D'][:, 0].copy()
                self.faces['U'][:, 0] = fc0
                self.faces['B'][:, 2] = uc0[::-1]
                self.faces['D'][:, 0] = bc2[::-1]
                self.faces['F'][:, 0] = dc0

    def is_solved(self) -> bool:
        """Return True if every face has a single uniform color."""
        return all(np.all(f == f[1, 1]) for f in self.faces.values())

    def load_from_dict(self, state: Dict[str, Any]) -> None:
        """Load cube state from a dict of face-name -> 3x3 color lists."""
        for fn in self.faces:
            if fn in state:
                self.faces[fn] = np.array(state[fn], dtype='U1')

    def to_dict(self) -> Dict[str, List[List[str]]]:
        """Export cube state as a dict of face-name -> 3x3 color lists."""
        return {fn: self.faces[fn].tolist() for fn in self.faces}

    def as_cubie_cube(self) -> 'CubieCube':
        """Convert current facelet state to a CubieCube representation."""
        cc = CubieCube()
        u_col = self.faces['U'][1, 1]
        d_col = self.faces['D'][1, 1]

        # Corners
        for pos in range(8):
            facets = _CORNER_FACETS[pos]
            actual_colors = [self.faces[f][r][c] for f, r, c in facets]
            actual_set = set(actual_colors)
            found = False
            for piece in range(8):
                piece_set = set(_CORNER_COLORS[piece])
                if actual_set == piece_set:
                    cc.cp[pos] = piece
                    for k in range(3):
                        if actual_colors[k] in (u_col, d_col):
                            cc.co[pos] = k
                            break
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Could not identify corner at position {pos}: {actual_colors}"
                )

        # Edges
        for pos in range(12):
            facets = _EDGE_FACETS[pos]
            actual_colors = [self.faces[f][r][c] for f, r, c in facets]
            actual_set = set(actual_colors)
            found = False
            for piece in range(12):
                piece_set = set(_EDGE_COLORS[piece])
                if actual_set == piece_set:
                    cc.ep[pos] = piece
                    ref_color = _EDGE_COLORS[piece][0]
                    cc.eo[pos] = 0 if actual_colors[0] == ref_color else 1
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Could not identify edge at position {pos}: {actual_colors}"
                )
        return cc


# ---------------------------------------------------------------------------
# Piece-based cube state (for solver)
# ---------------------------------------------------------------------------
class CubieCube:
    """Stores the cube as corner/edge permutation + orientation arrays.

    Corner permutation (cp): list of 8 ints, each 0-7
    Corner orientation (co): list of 8 ints, each 0-2
    Edge permutation (ep): list of 12 ints, each 0-11
    Edge orientation (eo): list of 12 ints, each 0-1
    """

    def __init__(self, cp: Optional[List[int]] = None,
                 co: Optional[List[int]] = None,
                 ep: Optional[List[int]] = None,
                 eo: Optional[List[int]] = None) -> None:
        self.cp = cp if cp is not None else list(range(8))
        self.co = co if co is not None else [0] * 8
        self.ep = ep if ep is not None else list(range(12))
        self.eo = eo if eo is not None else [0] * 12

    def copy(self) -> 'CubieCube':
        return CubieCube(self.cp[:], self.co[:], self.ep[:], self.eo[:])

    def apply(self, move_name: str) -> None:
        """Apply a named move (e.g. 'R', "R'", 'R2')."""
        face = move_name[0]
        suffix = move_name[1:] if len(move_name) > 1 else ''
        times = 3 if suffix == "'" else (2 if suffix == '2' else 1)
        cp_f = _CORNER_PERM[face]
        co_f = _CORNER_ORIENT[face]
        ep_f = _EDGE_PERM[face]
        eo_f = _EDGE_ORIENT[face]
        for _ in range(times):
            new_cp = [self.cp[cp_f[i]] for i in range(8)]
            new_co = [(self.co[cp_f[i]] + co_f[i]) % 3 for i in range(8)]
            new_ep = [self.ep[ep_f[i]] for i in range(12)]
            new_eo = [(self.eo[ep_f[i]] + eo_f[i]) % 2 for i in range(12)]
            self.cp, self.co, self.ep, self.eo = new_cp, new_co, new_ep, new_eo

    def is_solved(self) -> bool:
        return (self.cp == list(range(8)) and self.co == [0] * 8 and
                self.ep == list(range(12)) and self.eo == [0] * 12)


# ---------------------------------------------------------------------------
# Facelet <-> CubieCube conversion tables
# ---------------------------------------------------------------------------
_CORNER_FACETS = [
    [('U', 2, 2), ('R', 0, 0), ('F', 0, 2)],  # 0: URF
    [('U', 2, 0), ('F', 0, 0), ('L', 0, 2)],  # 1: UFL
    [('U', 0, 0), ('L', 0, 0), ('B', 0, 2)],  # 2: ULB
    [('U', 0, 2), ('B', 0, 0), ('R', 0, 2)],  # 3: UBR
    [('D', 0, 2), ('F', 2, 2), ('R', 2, 0)],  # 4: DFR
    [('D', 0, 0), ('L', 2, 2), ('F', 2, 0)],  # 5: DLF
    [('D', 2, 0), ('B', 2, 2), ('L', 2, 0)],  # 6: DBL
    [('D', 2, 2), ('R', 2, 2), ('B', 2, 0)],  # 7: DRB
]

_EDGE_FACETS = [
    [('U', 1, 2), ('R', 0, 1)],   # 0: UR
    [('U', 2, 1), ('F', 0, 1)],   # 1: UF
    [('U', 1, 0), ('L', 0, 1)],   # 2: UL
    [('U', 0, 1), ('B', 0, 1)],   # 3: UB
    [('D', 1, 2), ('R', 2, 1)],   # 4: DR
    [('D', 0, 1), ('F', 2, 1)],   # 5: DF
    [('D', 1, 0), ('L', 2, 1)],   # 6: DL
    [('D', 2, 1), ('B', 2, 1)],   # 7: DB
    [('F', 1, 2), ('R', 1, 0)],   # 8: FR
    [('F', 1, 0), ('L', 1, 2)],   # 9: FL
    [('B', 1, 2), ('L', 1, 0)],   # 10: BL
    [('B', 1, 0), ('R', 1, 2)],   # 11: BR
]

_CORNER_COLORS: List[Tuple[str, ...]] = []
_EDGE_COLORS: List[Tuple[str, ...]] = []


def _init_color_tables() -> None:
    global _CORNER_COLORS, _EDGE_COLORS
    _CORNER_COLORS = []
    for facets in _CORNER_FACETS:
        colors = tuple(FACE_COLORS[f] for f, r, c in facets)
        _CORNER_COLORS.append(colors)
    _EDGE_COLORS = []
    for facets in _EDGE_FACETS:
        colors = tuple(FACE_COLORS[f] for f, r, c in facets)
        _EDGE_COLORS.append(colors)


_init_color_tables()


# ---------------------------------------------------------------------------
# Coordinate encoding/decoding utilities
# ---------------------------------------------------------------------------
def encode_co(co: List[int]) -> int:
    """Encode corner orientations (first 7) as base-3 number. Range: 0..2186."""
    n = 0
    for i in range(7):
        n = n * 3 + co[i]
    return n


def decode_co(n: int) -> List[int]:
    """Decode corner orientation index to 8-element list."""
    co = [0] * 8
    s = 0
    for i in range(6, -1, -1):
        co[i] = n % 3
        n //= 3
        s += co[i]
    co[7] = (3 - s % 3) % 3
    return co


def encode_eo(eo: List[int]) -> int:
    """Encode edge orientations (first 11) as base-2 number. Range: 0..2047."""
    n = 0
    for i in range(11):
        n = n * 2 + eo[i]
    return n


def decode_eo(n: int) -> List[int]:
    """Decode edge orientation index to 12-element list."""
    eo = [0] * 12
    s = 0
    for i in range(10, -1, -1):
        eo[i] = n % 2
        n //= 2
        s += eo[i]
    eo[11] = s % 2
    return eo


_SLICE_COMBOS = list(combinations(range(12), 4))
_SLICE_TO_IDX = {c: i for i, c in enumerate(_SLICE_COMBOS)}


def encode_slice(ep: List[int]) -> int:
    """Given edge permutation, find which positions contain slice edges (pieces 8-11).
    Returns index 0..494."""
    positions = tuple(sorted(i for i in range(12) if ep[i] >= 8))
    return _SLICE_TO_IDX[positions]


def decode_slice(idx: int) -> Tuple[int, ...]:
    """Return the 4 positions that contain slice edges for this index."""
    return _SLICE_COMBOS[idx]


SOLVED_SLICE_IDX = _SLICE_TO_IDX[(8, 9, 10, 11)]


def encode_cp(cp: List[int]) -> int:
    """Encode corner permutation using Lehmer code. Range: 0..40319."""
    n = 0
    for i in range(8):
        count = 0
        for j in range(i + 1, 8):
            if cp[j] < cp[i]:
                count += 1
        n = n * (8 - i) + count
    return n


def encode_ep8(ep: List[int]) -> int:
    """Encode the permutation of the first 8 edges using Lehmer code. Range: 0..40319."""
    sub = ep[:8]
    n = 0
    for i in range(8):
        count = 0
        for j in range(i + 1, 8):
            if sub[j] < sub[i]:
                count += 1
        n = n * (8 - i) + count
    return n


def encode_eslice(ep: List[int]) -> int:
    """Encode the permutation of slice edges (positions 8-11). Range: 0..23."""
    sub = [ep[i] - 8 for i in range(8, 12)]
    n = 0
    for i in range(4):
        count = 0
        for j in range(i + 1, 4):
            if sub[j] < sub[i]:
                count += 1
        n = n * (4 - i) + count
    return n


def get_permutation_parity(p: List[int]) -> int:
    """Calculate the parity of a permutation (0 for even, 1 for odd)."""
    p = list(p)
    parity = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                parity += 1
    return parity % 2
