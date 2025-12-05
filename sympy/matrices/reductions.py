from typing import Callable, Optional, Tuple, List

from sympy.simplify.simplify import (
    simplify as _simplify, dotprodsimp as _dotprodsimp)

from .utilities import _get_intermediate_simp, _iszero
from .determinant import _find_reasonable_pivot


def _row_reduce_list(mat: list,
                     rows: int,
                     cols: int,
                     one,
                     iszerofunc: Callable,
                     simpfunc: Callable,
                     normalize_last: bool = True,
                     normalize: bool = True,
                     zero_above: bool = True) -> Tuple[list, Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
    """Row reduce a flat list representation of a matrix and return a tuple
    (rref_matrix, pivot_cols, swaps) where ``rref_matrix`` is a flat list,
    ``pivot_cols`` are the pivot columns and ``swaps`` are any row swaps that
    were used in the process of row reduction.

    Parameters
    ==========
    mat : list
        list of matrix elements, must be ``rows`` * ``cols`` in length
    rows, cols : integer
        number of rows and columns in flat list representation
    one : SymPy object
        represents the value one, from ``Matrix.one``
    iszerofunc : callable
        determines if an entry can be used as a pivot. Should return True,
        False or None.
    simpfunc : callable
        used to simplify elements and test if they are zero if ``iszerofunc``
        returns `None`
    normalize_last : indicates where all row reduction should
        happen in a fraction-free manner and then the rows are
        normalized (so that the pivots are 1), or whether
        rows should be normalized along the way (like the naive
        row reduction algorithm)
    normalize : whether pivot rows should be normalized so that
        the pivot value is 1
    zero_above : whether entries above the pivot should be zeroed.
        If ``zero_above=False``, an echelon matrix will be returned.
    """

    def get_col(i: int) -> List:
        # Returns a slice view of column i; this creates a list but kept
        # localized to avoid repeated large allocations elsewhere.
        return mat[i::cols]

    def row_swap(i: int, j: int) -> None:
        i0, j0 = i * cols, j * cols
        mat[i0:(i0 + cols)], mat[j0:(j0 + cols)] = \
            mat[j0:(j0 + cols)], mat[i0:(i0 + cols)]

    def cross_cancel(a, i_row: int, b, j_row: int) -> None:
        """Row op row[i_row] = a*row[i_row] - b*row[j_row]"""
        i_base = i_row * cols
        j_base = j_row * cols
        # compute stride once and avoid repeated index arithmetic
        for k in range(cols):
            p_i = i_base + k
            p_j = j_base + k
            mat[p_i] = isimp(a * mat[p_i] - b * mat[p_j])

    isimp = _get_intermediate_simp(_dotprodsimp)
    piv_row = piv_col = 0
    pivot_cols: List[int] = []
    swaps: List[Tuple[int, int]] = []

    # use a fraction free method to zero above and below each pivot
    while piv_col < cols and piv_row < rows:
        # create a column slice starting at piv_row
        col_slice = get_col(piv_col)[piv_row:]
        pivot_offset, pivot_val, assumed_nonzero, newly_determined = \
            _find_reasonable_pivot(col_slice, iszerofunc, simpfunc)

        # _find_reasonable_pivot may have simplified some things
        for (offset, val) in newly_determined:
            offset += piv_row
            mat[offset * cols + piv_col] = val

        if pivot_offset is None:
            piv_col += 1
            continue

        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            swap_with = pivot_offset + piv_row
            row_swap(piv_row, swap_with)
            swaps.append((piv_row, swap_with))

        # if we aren't normalizing last, we normalize
        # before we zero the other rows
        if normalize_last is False:
            i = piv_row
            j = piv_col
            base = i * cols
            mat[base + j] = one
            inv_pivot = pivot_val
            for k in range(j + 1, cols):
                mat[base + k] = isimp(mat[base + k] / inv_pivot)
            # after normalizing, the pivot value is 1
            pivot_val = one

        # zero above and below the pivot
        for row in range(rows):
            # don't zero our current row
            if row == piv_row:
                continue
            # don't zero above the pivot unless we're told.
            if zero_above is False and row < piv_row:
                continue
            # if we're already a zero, don't do anything
            val = mat[row * cols + piv_col]
            if iszerofunc(val):
                continue

            cross_cancel(pivot_val, row, val, piv_row)
        piv_row += 1

    # normalize each row (if requested)
    if normalize_last is True and normalize is True:
        for piv_i, piv_j in enumerate(pivot_cols):
            base = piv_i * cols
            pivot_val = mat[base + piv_j]
            mat[base + piv_j] = one
            for k in range(piv_j + 1, cols):
                mat[base + k] = isimp(mat[base + k] / pivot_val)

    return mat, tuple(pivot_cols), tuple(swaps)


# This function is a candidate for caching if it gets implemented for matrices.
def _row_reduce(M, iszerofunc, simpfunc, normalize_last: bool = True,
                normalize: bool = True, zero_above: bool = True):
    mat, pivot_cols, swaps = _row_reduce_list(list(M), M.rows, M.cols, M.one,
                                              iszerofunc, simpfunc,
                                              normalize_last=normalize_last,
                                              normalize=normalize, zero_above=zero_above)

    return M._new(M.rows, M.cols, mat), pivot_cols, swaps


def _is_echelon(M, iszerofunc=_iszero):
    """Returns `True` if the matrix is in echelon form. That is, all rows of
    zeros are at the bottom, and below each leading non-zero in a row are
    exclusively zeros."""
    if M.rows <= 0 or M.cols <= 0:
        return True

    zeros_below = all(iszerofunc(t) for t in M[1:, 0])

    if iszerofunc(M[0, 0]):
        return zeros_below and _is_echelon(M[:, 1:], iszerofunc)

    return zeros_below and _is_echelon(M[1:, 1:], iszerofunc)


def _echelon_form(M, iszerofunc=_iszero, simplify: Optional[Callable] = False, with_pivots: bool = False):
    """Returns a matrix row-equivalent to ``M`` that is in echelon form.

    If ``with_pivots`` is True the function returns (matrix_in_echelon_form, pivots).
    """
    # preserve legacy behaviour: non-callable simplify => use SymPy's _simplify
    simpfunc = simplify if callable(simplify) else _simplify

    mat, pivots, _ = _row_reduce(M, iszerofunc, simpfunc,
                                 normalize_last=True, normalize=False, zero_above=False)

    if with_pivots:
        return mat, pivots

    return mat


# This function is a candidate for caching if it gets implemented for matrices.
def _rank(M, iszerofunc=_iszero, simplify: Optional[Callable] = False) -> int:
    """Returns the rank of a matrix."""
    def _permute_complexity_right(M, iszerofunc):
        """Permute columns with complicated elements as far right as they can go."""
        def complexity(i):
            # the complexity of a column will be judged by how many
            # element's zero-ness cannot be determined
            return sum(1 if iszerofunc(e) is None else 0 for e in M[:, i])

        complex_list = [(complexity(i), i) for i in range(M.cols)]
        perm = [j for (i, j) in sorted(complex_list)]

        return (M.permute(perm, orientation='cols'), perm)

    # preserve legacy behaviour: non-callable simplify => use SymPy's _simplify
    simpfunc = simplify if callable(simplify) else _simplify

    # for small matrices, we compute the rank explicitly
    # if is_zero on elements doesn't answer the question
    # for small matrices, we fall back to the full routine.
    if M.rows <= 0 or M.cols <= 0:
        return 0

    if M.rows <= 1 or M.cols <= 1:
        zeros = [iszerofunc(x) for x in M]

        if False in zeros:
            return 1

    if M.rows == 2 and M.cols == 2:
        zeros = [iszerofunc(x) for x in M]

        if False not in zeros and None not in zeros:
            return 0

        d = M.det()

        if iszerofunc(d) and False in zeros:
            return 1
        if iszerofunc(d) is False:
            return 2

    mat, _ = _permute_complexity_right(M, iszerofunc=iszerofunc)
    _, pivots, _ = _row_reduce(mat, iszerofunc, simpfunc, normalize_last=True,
                              normalize=False, zero_above=False)

    return len(pivots)


def _rref(M, iszerofunc=_iszero, simplify: Optional[Callable] = False, pivots: bool = True,
          normalize_last: bool = True):
    """Return reduced row-echelon form of matrix and indices of pivot vars.

    Parameters
    ----------
    iszerofunc : callable
        function used to detect exact-zero (should return True/False/None)
    simplify : callable or False
        If callable, used as simplifier; otherwise, for historical/backward
        compatibility, SymPy's default `_simplify` is used.
    pivots : bool
        If True (default) return (matrix, pivot_cols) else return only matrix
    normalize_last : bool
        Whether to perform fraction-free elimination and normalize pivots at end
    """
    # preserve legacy behaviour: non-callable simplify => use SymPy's _simplify
    simpfunc = simplify if callable(simplify) else _simplify

    mat, pivot_cols, _ = _row_reduce(M, iszerofunc, simpfunc,
                                     normalize_last, normalize=True, zero_above=True)

    if pivots:
        return mat, pivot_cols

    return mat
