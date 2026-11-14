"""
Batched L-BFGS-B driver.

Runs many independent L-BFGS-B problems of the same dimension side by side.
Each problem keeps its own reverse-communication state; all problems that
need a new function/gradient evaluation are evaluated with a single call to
the (vectorized) objective. Problems drop out of the batch as they terminate.

The iterations of each problem are those of
``scipy.optimize.minimize(method='L-BFGS-B')``: the same compiled L-BFGS-B
core is used, and the stopping rules of `_minimize_lbfgsb` are applied
(in compiled code, see ``setulb_batch`` in ``src/lbfgsb.c``).
"""
import numpy as np

from scipy._lib._util import HAS_ILP64
from scipy._lib._elementwise_iterative_method import (
    _ECONVERGED, _ECONVERR, _EVALUEERR, _ECALLBACK, _EINPUTERR, _EINPROGRESS)
from . import _lbfgsb
from ._lbfgsb_py import status_messages, task_messages


# L-BFGS-B specific status code: the line search could not make progress
# (``ABNORMAL`` termination of L-BFGS-B).
_ELINESEARCH = -1

# task[:, 0] values of the compiled L-BFGS-B code
_FG, _CONVERGENCE, _STOP, _WARNING, _ERROR, _ABNORMAL = 3, 4, 5, 6, 7, 8
_STOP_CALLB = 505


def _status_from_task(task):
    """Map L-BFGS-B task codes of terminated problems to status codes."""
    t0 = task[:, 0]
    status = np.full(t0.shape, _EINPUTERR, dtype=np.int32)
    status[t0 == _CONVERGENCE] = _ECONVERGED
    status[(t0 == _ABNORMAL) | (t0 == _WARNING)] = _ELINESEARCH
    status[t0 == _STOP] = _ECONVERR
    return status


def _lbfgsb_batched(fg, x0, lb, ub, *, maxcor=10, ftol=2.2204460492503131e-09,
                    gtol=1e-5, maxls=20, maxiter=15000, maxfun=15000,
                    callback=None):
    """Minimize ``B`` independent functions of ``n`` variables with L-BFGS-B.

    Parameters
    ----------
    fg : callable
        ``fg(idx, x) -> (f, g, nfev)``. ``idx`` is an integer array of shape
        ``(k,)`` with the indices of the problems to evaluate, and ``x`` is a
        float64 array of shape ``(k, n)`` (a copy) with the points at which to
        evaluate them. Must return the function values ``f`` (shape ``(k,)``),
        the gradients ``g`` (shape ``(k, n)``), and the number of function
        evaluations used for each problem.
    x0 : ndarray, shape (B, n)
        Initial guesses.
    lb, ub : ndarray, shape (B, n)
        Lower and upper bounds (use ``-inf``/``inf`` for no bound).
    maxcor, ftol, gtol, maxls, maxiter, maxfun
        See `scipy.optimize.minimize` with ``method='L-BFGS-B'``.
    callback : callable, optional
        ``callback(state) -> bool`` is called after each round of batched
        evaluations. ``state`` is a dict with the current arrays (``x``,
        ``fun``, ``jac``: the last accepted iterates; ``nit``, ``nfev``,
        ``njev``, ``status``; status is ``1`` for problems still in progress).
        The arrays must not be modified. Returning ``True`` halts all problems
        that are still in progress (status ``-4``).

    Returns
    -------
    res : dict
        Arrays ``x``, ``fun``, ``jac``, ``nit``, ``nfev``, ``njev``,
        ``status``, ``message`` with leading dimension ``B``.
    """
    B, n = x0.shape
    m = int(maxcor)
    factr = ftol / np.finfo(float).eps
    int_dtype = np.int64 if HAS_ILP64 else np.int32

    # bound types: 0 - unbounded, 1 - only lower, 2 - both, 3 - only upper
    has_lb, has_ub = np.isfinite(lb), np.isfinite(ub)
    nbd = np.where(has_lb & has_ub, 2,
                   np.where(has_lb, 1, np.where(has_ub, 3, 0)))
    nbd = np.ascontiguousarray(nbd, dtype=int_dtype)
    low_bnd = np.ascontiguousarray(np.where(has_lb, lb, 0.0), dtype=np.float64)
    upper_bnd = np.ascontiguousarray(np.where(has_ub, ub, 0.0), dtype=np.float64)

    # Initial vectors must lie within the bounds; this matches what
    # `_minimize_lbfgsb` does.
    x = np.array(np.clip(x0, lb, ub), dtype=np.float64, order='C')
    f = np.zeros(B, dtype=np.float64)
    g = np.zeros((B, n), dtype=np.float64)
    wa = np.zeros((B, 2*m*n + 5*n + 11*m*m + 8*m), dtype=np.float64)
    iwa = np.zeros((B, 3*n), dtype=int_dtype)
    task = np.zeros((B, 2), dtype=int_dtype)
    ln_task = np.zeros((B, 2), dtype=int_dtype)
    lsave = np.zeros((B, 4), dtype=int_dtype)
    isave = np.zeros((B, 44), dtype=int_dtype)
    dsave = np.zeros((B, 29), dtype=np.float64)

    # last accepted iterate of each problem
    xk = x.copy()
    fk = np.full(B, np.nan)
    gk = np.full((B, n), np.nan)

    nit = np.zeros(B, dtype=np.int64)
    nfev = np.zeros(B, dtype=np.int64)
    njev = np.zeros(B, dtype=np.int64)
    status = np.full(B, _EINPROGRESS, dtype=np.int32)

    def step(idx):
        _lbfgsb.setulb_batch(m, x, low_bnd, upper_bnd, nbd, f, g, factr, gtol,
                             wa, iwa, task, lsave, isave, dsave, maxls, ln_task,
                             xk, fk, gk, nit, nfev, maxiter, maxfun, idx)

    def finalize(idx):
        # problems that do not request an evaluation have terminated
        needs_eval = task[idx, 0] == _FG
        if needs_eval.all():  # fast path: nothing terminated in this round
            return idx
        done = idx[~needs_eval]
        status[done] = _status_from_task(task[done])
        return idx[needs_eval]

    state = dict(x=xk, fun=fk, jac=gk, nit=nit, nfev=nfev, njev=njev,
                 status=status)

    # START: checks input, projects x0 onto the bounds and requests f and g
    active = np.arange(B, dtype=np.intp)
    step(active)
    active = finalize(active)

    first_round = True
    while active.size:
        # `x[active]` is a copy: the objective may keep a reference to it
        f_new, g_new, nfev_new = fg(active, x[active])
        # while no problem has terminated, plain slices are cheaper than
        # fancy indexing
        sel = slice(None) if active.size == B else active
        f[sel] = f_new
        g[sel] = g_new
        nfev[sel] += nfev_new
        njev[sel] += 1

        # Non-finite values are passed on to L-BFGS-B like in `minimize`
        # (e.g. the line search backtracks from points where f is inf).
        # Problems with a NaN objective or a non-finite gradient at the
        # initial point cannot make progress, however, and are stopped.
        # All problems are evaluated at their initial point in the first round.
        if first_round:
            first_round = False
            bad = np.isnan(f[active]) | ~np.all(np.isfinite(g[active]), axis=-1)
            if np.any(bad):
                b = active[bad]
                xk[b], fk[b], gk[b] = x[b], f[b], g[b]
                status[b] = _EVALUEERR
                active = active[~bad]

        step(active)
        active = finalize(active)

        if callback is not None and callback(state):
            status[active] = _ECALLBACK
            task[active] = (_STOP, _STOP_CALLB)
            break

    # A non-finite objective or gradient at the reported point is flagged as
    # such, whatever the reason the compiled code stopped.
    non_finite = ~(np.isfinite(fk) & np.all(np.isfinite(gk), axis=-1))
    status[non_finite & (status != _EINPUTERR)] = _EVALUEERR

    message = np.empty(B, dtype=object)
    for i in range(B):
        if status[i] == _EVALUEERR:
            message[i] = "ABNORMAL: NON-FINITE FUNCTION VALUE OR GRADIENT"
        else:
            message[i] = (status_messages[int(task[i, 0])] + ": "
                          + task_messages[int(task[i, 1])])
    message = message.astype(str) if B else np.empty(0, dtype=str)

    return dict(x=xk, fun=fk, jac=gk, nit=nit, nfev=nfev, njev=njev,
                status=status, message=message)
