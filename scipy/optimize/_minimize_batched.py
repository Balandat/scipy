"""
Batched minimization of many independent multivariate functions.
"""
import math
import operator
import os

import numpy as np

from scipy._lib._array_api import (
    array_namespace, xp_capabilities, xp_result_device, xp_result_type,
    is_numpy, is_cupy, is_torch, is_array_api_strict)
from scipy._lib._elementwise_iterative_method import _EINPROGRESS, _ECONVERGED
from scipy._lib._util import MapWrapper
from ._optimize import OptimizeResult
from ._constraints import Bounds
from ._numdiff import _approx_gradient_batched
from ._lbfgsb_batched import _lbfgsb_batched

__all__ = ['minimize_batched']


_FD_METHODS = ('2-point', '3-point', 'cs')

_DEFAULT_TOLERANCES = {
    'l-bfgs-b': {'ftol': 2.2204460492503131e-09, 'gtol': 1e-5},
}

_DEFAULT_OPTIONS = {
    'l-bfgs-b': {'maxcor': 10, 'maxls': 20, 'maxfun': 15000, 'eps': 1e-8,
                 'finite_diff_rel_step': None},
}

_RESULT_KEYS = ['success', 'status', 'message', 'x', 'fun', 'jac', 'nit',
                'nfev', 'njev']


def _to_numpy(a):
    """Copy an array of any supported backend to a NumPy array on the host.

    The objective function is evaluated in the array namespace (and on the
    device) of `x0`, while the optimizer state lives on the host; the function
    values and gradients therefore have to be transferred to the host.
    """
    if isinstance(a, np.ndarray | np.generic | float | int | complex):
        return np.asarray(a)
    xp = array_namespace(a)
    if is_numpy(xp):
        return np.asarray(a)
    if is_torch(xp):
        # `detach` is required for tensors that are part of an autograd graph
        return a.detach().cpu().numpy()
    if is_cupy(xp):
        return a.get()
    if is_array_api_strict(xp):
        return np.asarray(xp.asarray(a, device=xp.Device("CPU_DEVICE")))
    return np.asarray(a)


def _np_float_dtype(xp, dtype):
    return {16: np.float16, 32: np.float32, 64: np.float64}[xp.finfo(dtype).bits]


def _validate_options(method, tolerances, options, maxiter, n):
    tol = dict(_DEFAULT_TOLERANCES[method])
    if tolerances is not None:
        unknown = set(tolerances) - set(tol)
        if unknown:
            raise ValueError(f"Unknown tolerances {sorted(unknown)}; valid keys "
                             f"for method '{method}' are {sorted(tol)}.")
        tol.update(tolerances)
    for key, val in tol.items():
        val = float(val)
        if not (val >= 0):
            raise ValueError(f"Tolerance `{key}` must be non-negative.")
        tol[key] = val

    opts = dict(_DEFAULT_OPTIONS[method])
    if options is not None:
        unknown = set(options) - set(opts)
        if unknown:
            raise ValueError(f"Unknown options {sorted(unknown)}; valid options "
                             f"for method '{method}' are {sorted(opts)}.")
        opts.update(options)

    for key in ('maxcor', 'maxls', 'maxfun'):
        try:
            opts[key] = operator.index(opts[key])
        except TypeError as e:
            raise ValueError(f"Option `{key}` must be an integer.") from e
    if opts['maxcor'] < 1 or opts['maxls'] < 1 or opts['maxfun'] < 0:
        raise ValueError("Options `maxcor` and `maxls` must be positive and "
                         "`maxfun` must be non-negative.")
    try:
        maxiter = operator.index(maxiter)
    except TypeError as e:
        raise ValueError("`maxiter` must be a non-negative integer.") from e
    if maxiter < 0:
        raise ValueError("`maxiter` must be a non-negative integer.")

    for key in ('eps', 'finite_diff_rel_step'):
        val = opts[key]
        if val is not None:
            val = np.asarray(val, dtype=np.float64)
            try:
                np.broadcast_to(val, (n,))
            except ValueError as e:
                raise ValueError(f"Option `{key}` must be a scalar or have "
                                 f"shape ({n},).") from e
            opts[key] = val
    return tol, opts, maxiter


def _prepare_bounds(bounds, n, batch_shape):
    """Return host arrays `lb`, `ub` of shape ``(B, n)``."""
    B = math.prod(batch_shape)
    if bounds is None:
        return np.full((B, n), -np.inf), np.full((B, n), np.inf)
    if not isinstance(bounds, Bounds):
        raise TypeError("`bounds` must be None or an instance of "
                        "`scipy.optimize.Bounds`.")

    out = []
    for name, b in (('lb', bounds.lb), ('ub', bounds.ub)):
        b = np.asarray(_to_numpy(b), dtype=np.float64)
        if b.ndim == 0:
            b = b[np.newaxis]
        if b.shape[0] not in (1, n):
            raise ValueError(f"The first axis of `bounds.{name}` must have "
                             f"length 1 or n={n} (the number of variables).")
        # The first axis of the bounds corresponds with the variables, the
        # remaining axes are broadcast against the batch shape.
        b = np.moveaxis(b, 0, -1)
        try:
            b = np.broadcast_to(b, batch_shape + (n,))
        except ValueError as e:
            raise ValueError(f"`bounds.{name}` with shape {b.shape} cannot be "
                             "broadcast to the shape of the problem "
                             f"{(n,) + batch_shape}.") from e
        out.append(np.reshape(b, (B, n)))

    lb, ub = out
    if np.any(lb > ub):
        raise ValueError("One of the lower bounds is greater than an upper "
                         "bound.")
    return lb, ub


def _n_workers(workers):
    """Number of chunks to create for `workers` (None: no parallelism)."""
    if workers is None:
        return 1
    if callable(workers):
        # `Pool.map` and `concurrent.futures` executors' `map`: one chunk per
        # worker of the pool; other map-like callables: one chunk per CPU
        pool = getattr(workers, "__self__", None)
        for attr in ("_processes", "_max_workers"):
            k = getattr(pool, attr, None)
            if isinstance(k, int) and k >= 1:
                return k
        n_cpus = getattr(os, "process_cpu_count", os.cpu_count)()
        return n_cpus or 1
    try:
        k = operator.index(workers)
    except TypeError as e:
        raise ValueError("`workers` must be None, an integer, or a map-like "
                         "callable.") from e
    if k == -1:
        return getattr(os, "process_cpu_count", os.cpu_count)() or 1
    if k < 1:
        raise ValueError("`workers` must be -1, a positive integer, or a "
                         "map-like callable.")
    return k


_CHUNK_RESULT_KEYS = ('x', 'fun', 'jac', 'nit', 'nfev', 'njev', 'status', 'message')


def _minimize_batched_chunk(task):
    """Solve one chunk of problems (run by the workers; must be pickleable)."""
    fun, x0, kwds = task
    res = minimize_batched(fun, x0, **kwds)
    # results are returned as NumPy arrays (also for other array namespaces)
    return {key: _to_numpy(res[key]) for key in _CHUNK_RESULT_KEYS}


@xp_capabilities(
    cpu_only=True, exceptions=["cupy", "torch", "jax.numpy"], jax_jit=False,
    skip_backends=[("dask.array", "the iterations run eagerly on the host")],
    extra_note=(
        "The iterations of the optimizer run in NumPy on the host. `fun` and "
        "`jac` are called with arrays of the namespace and on the device of "
        "`x0`; their outputs are copied to the host in every iteration."))
def minimize_batched(fun, x0, /, *, method='L-BFGS-B', jac=None, bounds=None,
                     args=(), kwargs=None, tolerances=None, maxiter=15000,
                     callback=None, options=None, workers=None):
    r"""Minimize many independent multivariate functions simultaneously.

    Solves a batch of independent problems
    :math:`\min_x f_i(x)`, :math:`x \in \mathbb{R}^n` (optionally subject to
    bounds), with one solver instance per problem. In every iteration, all
    problems that need a function evaluation are evaluated with a *single*
    call to a vectorized `fun`. Problems drop out of the batch as they
    terminate.

    This is useful when evaluating `fun` at many points at once is not much
    more expensive than evaluating it at a single point: for instance when
    `fun` is vectorized with NumPy, or when it is evaluated with an array
    library that runs on a GPU. Typical applications are multi-start local
    optimization (the same function from many starting points) and solving
    many related problems that differ only in some parameters (see `args`).

    Parameters
    ----------
    fun : callable
        The objective functions to be minimized. The signature must be::

            fun(x: array, *args) -> array

        where ``x`` has shape ``(n, ...)``: the first axis corresponds with the
        ``n`` variables, and each remaining element along the trailing axes
        corresponds with a separate problem. `fun` must return an array of
        shape ``x.shape[1:]`` with the value of the objective function of each
        problem; that is, ``fun(x)[j]`` must equal ``f_j(x[:, j])``. Each
        element of each array in `args` (after broadcasting with the batch
        shape) is a parameter of the corresponding problem.

        `fun` is called only for problems that have not yet terminated, so
        the number of problems (the size of the trailing axes of ``x``) as
        well as the parameters in `args` change during the run. If `jac` is
        one of the finite difference methods, `fun` is also called with
        ``x`` of shape ``(n, k, s)`` to evaluate the ``s`` finite difference
        points of each of the ``k`` problems at once; in this case, the arrays
        in `args` have shape ``(k, 1)``. `fun` must not mutate its arguments.
    x0 : array_like, shape (n, ...)
        Initial guesses. The first axis corresponds with the ``n`` variables;
        the remaining axes (the "batch shape") index the problems. `x0` must
        be broadcastable with the arrays in `args` (and `kwargs`), excluding
        its first axis.
    method : str, optional
        Optimization method. Currently, only ``'L-BFGS-B'`` is supported.
        The iterations of each problem are those of
        ``minimize(..., method='L-BFGS-B')``.
    jac : {callable, True, '2-point', '3-point', 'cs', None}, optional
        Method for computing the gradients.

        - If callable, it must have the same signature as `fun` and return an
          array of shape ``x.shape`` (the gradient of each problem along the
          first axis).
        - If ``True``, `fun` returns a tuple ``(f, g)`` with the function
          values and the gradients.
        - If one of ``{'2-point', '3-point', 'cs'}``, the gradients are
          estimated with finite differences using a relative step size
          (see ``options['finite_diff_rel_step']``).
        - If ``None`` (default), the gradients are estimated with forward
          differences using the absolute step size ``options['eps']``.

        As in `minimize`, the finite difference steps respect the bounds.
    bounds : `Bounds`, optional
        Bounds on the variables. The first axis of ``bounds.lb`` and
        ``bounds.ub`` corresponds with the variables and must have length
        ``n`` or ``1``; any additional axes are broadcast against the batch
        shape, so that different problems may have different bounds. Use
        ``-np.inf``/``np.inf`` to indicate no bound. Variables with equal
        lower and upper bounds are fixed. Initial guesses outside the bounds
        are clipped to the bounds. ``bounds.keep_feasible`` is ignored.
    args : tuple of array_like, optional
        Additional positional arguments to be passed to `fun` (and `jac`).
        The arrays must be broadcastable with each other and with the batch
        shape of `x0`; they are passed to `fun` for the active problems only.
        Arguments that are not broadcastable with the batch shape must be
        wrapped into `fun` instead.
    kwargs : dict of array_like, optional
        Additional keyword arguments to be passed to `fun` (and `jac`). The
        same broadcasting rules as for `args` apply.
    tolerances : dictionary of floats, optional
        Termination tolerances. Valid keys of the dictionary are:

        - ``ftol``: the iteration of a problem stops when
          ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
        - ``gtol``: the iteration of a problem stops when
          ``max{|proj g_i | i = 1, ..., n} <= gtol``, where ``proj g_i`` is
          the i-th component of the projected gradient.

        See Notes for the default values.
    maxiter : int, default: 15000
        Maximum number of iterations per problem.
    callback : callable, optional
        A function to be called before the first iteration and after each
        round of batched function evaluations as ``callback(res)``, where
        ``res`` is an `OptimizeResult` like the one returned by
        `minimize_batched` (but without ``message``), reporting the current
        (last accepted) iterate of every problem. The ``status`` of problems
        that are still in progress is ``1``. If `callback` raises a
        ``StopIteration``, all problems still in progress are terminated with
        status ``-4``. `callback` must not mutate `res` or its attributes.
    options : dict, optional
        Method-specific options. For ``method='L-BFGS-B'``:

        - ``maxcor`` (int, default 10): maximum number of variable metric
          corrections used to define the limited memory matrix.
        - ``maxls`` (int, default 20): maximum number of line search steps
          per iteration.
        - ``maxfun`` (int, default 15000): maximum number of function
          evaluations per problem. Function evaluations used for finite
          difference gradients count, too. As in `minimize`, the limit is
          checked between iterations, so it can be exceeded.
        - ``eps`` (float or array_like, default 1e-8): absolute step size
          for ``jac=None``.
        - ``finite_diff_rel_step`` (None or array_like): relative step size
          for ``jac`` in ``{'2-point', '3-point', 'cs'}``. If None,
          an appropriate step size is chosen automatically.

    workers : int or map-like callable, optional
        Solve the problems in parallel. The batch is split into chunks, one
        per worker (with at least two problems per chunk), and each chunk is
        solved with a separate call of `minimize_batched`; the results are
        assembled into the result of the whole batch.

        - ``None`` or ``1`` (default): no parallelism; the whole batch is
          solved in the current process.
        - An integer ``k > 1``: a `multiprocessing.Pool <multiprocessing.pool.Pool>`
          with ``k`` processes is created for the call; ``-1`` uses all
          available CPUs. `fun`, `jac`, and the arrays in `args` and `kwargs`
          must be pickleable.
        - A map-like callable such as ``multiprocessing.Pool.map`` or
          ``concurrent.futures.ThreadPoolExecutor.map``: chunks are evaluated
          as ``workers(func, iterable)``. For the ``map`` method of a
          `multiprocessing.pool.Pool` or of a `concurrent.futures` executor,
          the batch is split into one chunk per worker of the pool; for other
          map-like callables, into one chunk per CPU. Reusing a pool across
          calls avoids the cost of starting processes, and a thread pool does
          not require pickleable functions (and avoids sending the data that
          `fun` refers to to other processes).

        `callback` cannot be combined with `workers`. See Notes.

    Returns
    -------
    res : OptimizeResult
        An object similar to an instance of `OptimizeResult` with the
        following attributes. The descriptions are written as though the
        values will be scalars; however, if `x0` and `args` describe a batch
        of problems, each attribute is an array with the batch shape (or,
        for ``x`` and ``jac``, with shape ``(n, *batch_shape)``).

        success : bool array
            ``True`` when the algorithm terminated successfully (status ``0``);
            ``False`` otherwise.
        status : int array
            An integer representing the exit status of the algorithm.

            - ``0`` : Converged (the ``ftol`` or ``gtol`` criterion is met).
            - ``-1`` : The line search could not find a point with
              sufficiently lower objective value (abnormal termination).
            - ``-2`` : The maximum number of iterations or function
              evaluations was reached.
            - ``-3`` : A non-finite objective function value or gradient was
              encountered.
            - ``-4`` : Iteration was terminated by `callback`.
            - ``1`` : The algorithm is proceeding normally (in `callback`
              only).

        message : str array
            A description of the cause of the termination; for problems
            terminated by the L-BFGS-B algorithm, the same message as that
            of `minimize`.
        x : float array
            The solution of the optimization problem.
        fun : float array
            The values of the objective functions at the solution.
        jac : float array
            The gradients of the objective functions at the solution.
        nit : int array
            The number of iterations performed.
        nfev : int array
            The number of points at which the objective function of each
            problem was evaluated (including finite difference points). This
            is distinct from the number of times `fun` was called.
        njev : int array
            The number of gradients evaluated for each problem.

    See Also
    --------
    minimize : Minimization of a single function of one or more variables.
    scipy.optimize.elementwise.find_minimum :
        Elementwise minimization of scalar functions of one variable.

    Notes
    -----
    For ``method='L-BFGS-B'``, each problem is solved with the same compiled
    L-BFGS-B code [1]_, [2]_, [3]_ as ``minimize(..., method='L-BFGS-B')``,
    applying the same stopping rules: given identical function values and
    gradients, the iterates of each problem are identical to those of
    `minimize`. The differences are:

    - `callback` is called once per round of batched function evaluations
      with the state of all problems (not once per iteration of a single
      problem).
    - If `fun` is NaN or the gradient is not finite at the initial guess of a
      problem, that problem is not iterated (status ``-3``).
    - Fixed variables (equal lower and upper bounds) are not removed from
      the problem; their finite difference derivatives are set to zero.
    - The inverse Hessian approximation is not returned.

    The default tolerances are ``ftol=2.2204460492503131e-09`` and
    ``gtol=1e-5``.

    `fun` (and `jac`) are called with arrays of the array namespace and on
    the device of `x0` (and `args`), in the floating point type of `x0`.
    The state of the optimizer is held in double precision NumPy arrays on
    the host; points are copied to the device before each call to `fun`, and
    the function values and gradients are copied back to the host.

    Because problems drop out of the batch as they terminate, `fun` is called
    with arrays of many different shapes. Libraries that compile a function
    for each input shape, such as JAX with ``jax.jit``, may therefore compile
    `fun` many times. To limit the number of compilations, `fun` can pad its
    argument to one of a few sizes (e.g. the next power of two) before calling
    the compiled function, and discard the padded entries of the result; see
    the :ref:`tutorial <tutorial-optimize_batched>` for an example.

    When finite differences are used, all ``s`` finite difference points
    of the ``k`` active problems are evaluated in a single call of `fun`
    with an argument of shape ``(n, k, s)``, where ``s`` is ``n`` for
    ``'2-point'`` and ``'cs'`` and ``2n`` for ``'3-point'``.

    On the CPU, a single call of `minimize_batched` uses one core (unless the
    objective function itself is multithreaded). With `workers`, the problems
    are distributed over several cores, and each worker still evaluates its
    problems in batches; for large batches, this combines the benefits of
    batching and of parallelism. Parallel evaluation is not beneficial for
    objective functions that are evaluated on a GPU, nor for small batches of
    cheap problems, where the cost of distributing the work dominates. Each
    worker process evaluates `fun` with its own BLAS library; to avoid
    oversubscription, limit the number of BLAS threads per process (e.g. with
    ``threadpoolctl``) when `fun` uses multithreaded linear algebra. With
    process-based `workers`, `fun` (including any data it refers to) is
    pickled and sent with every chunk; if `fun` refers to large data, a
    thread pool (e.g. ``workers=ThreadPoolExecutor(4).map``) is usually
    faster than processes. Results are identical to those without
    `workers` if `fun` returns the same values for a problem regardless of
    which other problems are evaluated in the same call.

    References
    ----------
    .. [1] R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for
       Bound Constrained Optimization, (1995), SIAM Journal on Scientific and
       Statistical Computing, 16, 5, pp. 1190-1208.
    .. [2] C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778:
       L-BFGS-B, FORTRAN routines for large scale bound constrained
       optimization (1997), ACM Transactions on Mathematical Software, 23, 4,
       pp. 550 - 560.
    .. [3] J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778:
       L-BFGS-B, FORTRAN routines for large scale bound constrained
       optimization (2011), ACM Transactions on Mathematical Software, 38, 1.

    Examples
    --------
    The Rosenbrock function `rosen` and its gradient `rosen_der` are
    vectorized: the first axis of the argument corresponds with the
    variables. Minimize the 4-dimensional Rosenbrock function from five
    random starting points at once.

    >>> import numpy as np
    >>> from scipy.optimize import minimize_batched, rosen, rosen_der
    >>> rng = np.random.default_rng(1638083107694713882823079058616272161)
    >>> x0 = rng.uniform(-2, 2, size=(4, 5))  # 4 variables, 5 problems
    >>> res = minimize_batched(rosen, x0, jac=rosen_der)
    >>> res.success
    array([ True,  True,  True,  True,  True])
    >>> res.x.shape
    (4, 5)
    >>> np.round(res.fun, 4)
    array([3.7014, 0.    , 0.    , 0.    , 0.    ])

    All runs converged, but the first one converged to a local minimum. The
    best solution is

    >>> np.round(res.x[:, np.argmin(res.fun)], 6)
    array([1., 1., 1., 1.])

    The number of iterations differs between problems; each problem was
    evaluated only until it converged.

    >>> res.nit  # may vary
    array([34, 20, 29, 32, 33], dtype=int32)

    The problems may also differ in their parameters, which are passed
    via `args`. Minimize :math:`f(x) = \|x - c\|^2 + p \sum_i x_i^4` for
    five values of :math:`p`, subject to bounds :math:`x_i \geq 0.5`.

    >>> from scipy.optimize import Bounds
    >>> c = np.array([1., 2.])[:, np.newaxis]
    >>> def f(x, p):
    ...     return np.sum((x - c)**2, axis=0) + p*np.sum(x**4, axis=0)
    >>> def df(x, p):
    ...     return 2*(x - c) + 4*p*x**3
    >>> p = np.array([0., 0.1, 1., 10., 100.])
    >>> res = minimize_batched(f, np.zeros(2), jac=df, args=(p,),
    ...                        bounds=Bounds(0.5, np.inf))
    >>> res.x.shape
    (2, 5)
    >>> np.round(res.x, 3)
    array([[1.   , 0.869, 0.59 , 0.5  , 0.5  ],
           [2.   , 1.423, 0.835, 0.5  , 0.5  ]])
    """
    # --- input validation ---------------------------------------------------
    if not callable(fun):
        raise ValueError("`fun` must be callable.")
    if callback is not None and not callable(callback):
        raise ValueError("`callback` must be callable.")
    n_workers = _n_workers(workers)
    if n_workers > 1 and callback is not None:
        raise ValueError("`callback` cannot be used together with `workers`.")
    meth = str(method).lower()
    if meth not in _DEFAULT_OPTIONS:
        raise ValueError(f"Unknown method '{method}'; supported methods are "
                         "['L-BFGS-B'].")

    if callable(jac):
        jac_mode = 'callable'
    elif jac is True:
        jac_mode = 'fun'
    elif jac is None or jac is False:
        jac_mode = 'absolute'
    elif isinstance(jac, str) and jac in _FD_METHODS:
        jac_mode = 'relative'
    else:
        raise ValueError("`jac` must be callable, True, None, or one of "
                         f"{list(_FD_METHODS)}.")

    if not isinstance(args, tuple):
        args = (args,)
    kwargs = {} if kwargs is None else dict(kwargs)
    kw_keys = list(kwargs)

    xp = array_namespace(x0, *args, *kwargs.values())
    device = xp_result_device(x0, *args, *kwargs.values())
    x0 = xp.asarray(x0, device=device)
    dtype = xp_result_type(x0, force_floating=True, xp=xp)
    if not xp.isdtype(dtype, 'real floating'):
        raise ValueError("`x0` must be real.")
    if x0.ndim < 1 or x0.shape[0] < 1:
        raise ValueError("`x0` must have at least one dimension, and its first "
                         "axis (the variables) must have positive length.")
    np_dtype = _np_float_dtype(xp, dtype)
    np_cdtype = np.result_type(np_dtype, np.complex64)

    params = [xp.asarray(a, device=device) for a in (*args, *kwargs.values())]
    n = x0.shape[0]
    try:
        batch_shape = tuple(np.broadcast_shapes(
            tuple(x0.shape[1:]), *(tuple(p.shape) for p in params)))
    except ValueError as e:
        raise ValueError("The shape of `x0` (excluding the first axis) must be "
                         "broadcastable with the shapes of `args` and "
                         "`kwargs`.") from e
    B = math.prod(batch_shape)

    tol, opts, maxiter = _validate_options(meth, tolerances, options, maxiter, n)
    lb, ub = _prepare_bounds(bounds, n, batch_shape)

    x0_np = np.asarray(_to_numpy(x0), dtype=np.float64)
    # the batch axes of `x0` (all but the first) are aligned to the right
    x0_np = np.reshape(x0_np, (n,) + (1,)*(len(batch_shape) - x0_np.ndim + 1)
                       + x0_np.shape[1:])
    x0_np = np.broadcast_to(x0_np, (n,) + batch_shape).reshape(n, B).T
    # parameters of each problem, flattened along the batch dimensions
    params = [xp.reshape(xp.broadcast_to(p, batch_shape), (B,)) for p in params]

    if is_numpy(xp):
        # fast paths: avoid the overhead of the compat layer in every round
        def to_xp(a, dt=np_dtype):
            return np.asarray(a, dtype=dt)
        permute_dims = np.transpose
    else:
        def to_xp(a, dt=np_dtype):
            return xp.asarray(np.asarray(a, dtype=dt), device=device)
        permute_dims = xp.permute_dims

    # --- function evaluation -------------------------------------------------
    def call(f, x, idx, expand=False):
        if not params:
            return f(x)
        idx_xp = xp.asarray(idx, device=device)
        ps = [xp.take(p, idx_xp, axis=0) for p in params]
        if expand:
            ps = [xp.expand_dims(p, axis=-1) for p in ps]
        a, kw = ps[:len(args)], dict(zip(kw_keys, ps[len(args):]))
        return f(x, *a, **kw)

    def check_shape(val, shape, what):
        val = _to_numpy(val)
        if val.shape != shape:
            raise ValueError(f"{what} must return an array of shape {shape} "
                             f"for an argument of shape {(n,) + shape[-1:]}, "
                             f"but returned an array of shape {val.shape}.")
        return val

    def fg(idx, X):
        k = idx.shape[0]
        x = permute_dims(to_xp(X), (1, 0))  # shape (n, k)

        if jac_mode == 'fun':
            out = call(fun, x, idx)
            try:
                f, g = out
            except (TypeError, ValueError) as e:
                raise ValueError("With `jac=True`, `fun` must return a tuple "
                                 "`(f, g)`.") from e
            f = check_shape(f, (k,), "`fun`")
            g = check_shape(g, (n, k), "The gradient (`jac=True`)")
            return f, g.T, 1

        f = check_shape(call(fun, x, idx), (k,), "`fun`")
        if jac_mode == 'callable':
            g = check_shape(call(jac, x, idx), (n, k), "`jac`")
            return f, g.T, 1

        # Finite differences, in the floating point type of `x0`. All finite
        # difference points of all active problems are evaluated at once.
        def evaluate(S):  # S: (k, s, n) -> (k, s)
            dt = np_cdtype if np.iscomplexobj(S) else np_dtype
            xs = permute_dims(to_xp(S, dt), (2, 0, 1))  # (n, k, s)
            return check_shape(call(fun, xs, idx, expand=True),
                               S.shape[:2], "`fun`")

        X_e = X.astype(np_dtype)
        if jac_mode == 'absolute':
            fd_method, rel_step, abs_step = '2-point', None, opts['eps']
        else:
            fd_method, rel_step = jac, opts['finite_diff_rel_step']
            abs_step = None
        g, nev = _approx_gradient_batched(
            evaluate, X_e, f, method=fd_method, rel_step=rel_step,
            abs_step=abs_step, lb=lb[idx], ub=ub[idx])
        return f, g, 1 + nev

    # --- results ---------------------------------------------------------------
    def prepare_result(d, final):
        # `d` holds the optimizer's working arrays; always return copies, so
        # that results passed to `callback` are not modified later on.
        def batch(a, dt):
            return to_xp(np.array(np.reshape(a, batch_shape), dtype=dt), dt)

        def core(a):
            return to_xp(np.array(np.reshape(a.T, (n,) + batch_shape),
                                  dtype=np_dtype))

        status = d['status']
        res = OptimizeResult(
            success=batch(status == _ECONVERGED, np.bool_),
            status=batch(status, np.int32),
            x=core(d['x']),
            fun=batch(d['fun'], np_dtype),
            jac=core(d['jac']),
            nit=batch(d['nit'], np.int32),
            nfev=batch(d['nfev'], np.int32),
            njev=batch(d['njev'], np.int32),
        )
        keys = list(_RESULT_KEYS)
        if final:
            res['message'] = np.reshape(d['message'], batch_shape)
        else:
            keys.remove('message')
        res._order_keys = keys
        return res

    cb = None
    if callback is not None:
        def cb(state):
            try:
                callback(prepare_result(state, final=False))
            except StopIteration:
                return True
            return False

    # --- parallel evaluation of chunks of problems ------------------------------
    n_chunks = min(n_workers, B // 2)  # at least two problems per chunk
    if n_chunks > 1:
        chunks = np.array_split(np.arange(B), n_chunks)
        tasks = []
        for idx in chunks:
            idx_xp = xp.asarray(idx, device=device)
            ps = [xp.take(p, idx_xp, axis=0) for p in params]
            kwds = dict(method=method, jac=jac, args=tuple(ps[:len(args)]),
                        kwargs=dict(zip(kw_keys, ps[len(args):])),
                        tolerances=tol, maxiter=maxiter, options=options)
            if bounds is not None:
                kwds['bounds'] = Bounds(lb[idx].T, ub[idx].T)
            tasks.append((fun, to_xp(x0_np[idx].T), kwds))
        with MapWrapper(workers) as mapper:
            parts = list(mapper(_minimize_batched_chunk, tasks))
        d = {key: np.concatenate([part[key] for part in parts], axis=-1)
             for key in _CHUNK_RESULT_KEYS}
        d['x'], d['jac'] = d['x'].T, d['jac'].T  # internal layout: (B, n)
        return prepare_result(d, final=True)

    if B == 0:
        empty = dict(x=np.empty((0, n)), fun=np.empty(0), jac=np.empty((0, n)),
                     nit=np.empty(0), nfev=np.empty(0), njev=np.empty(0),
                     status=np.empty(0, dtype=np.int32),
                     message=np.empty(0, dtype=str))
        return prepare_result(empty, final=True)

    res = _lbfgsb_batched(fg, x0_np, lb, ub, maxcor=opts['maxcor'],
                          ftol=tol['ftol'], gtol=tol['gtol'],
                          maxls=opts['maxls'], maxiter=maxiter,
                          maxfun=opts['maxfun'], callback=cb)
    assert not np.any(res['status'] == _EINPROGRESS)
    return prepare_result(res, final=True)
