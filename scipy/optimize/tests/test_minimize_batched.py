import functools
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest

from scipy.optimize import (minimize_batched, minimize, rosen, rosen_der,
                            Bounds)
from scipy._lib._array_api import (array_namespace, make_xp_test_case,
                                   xp_device, is_torch)
from scipy._lib._array_api_no_0d import xp_assert_close, xp_assert_equal


# A smooth, nonconvex test function with a per-problem parameter `a`. It is
# written with operations that also work for complex input (for 'cs').
def f_scalar(x, a=0.):
    return np.sum((x - a)**2) + 0.1*np.sum(x**4) + np.sin(x[0]*x[1])


def g_scalar(x, a=0.):
    g = 2*(x - a) + 0.4*x**3
    g[0] += np.cos(x[0]*x[1])*x[1]
    g[1] += np.cos(x[0]*x[1])*x[0]
    return g


def fg_scalar(x, a=0.):
    return f_scalar(x, a), g_scalar(x, a)


def _loop_vectorized(f, out, x, *args):
    x = np.asarray(x)
    args = [np.broadcast_to(a, x.shape[1:]) for a in args]
    batch = x.shape[1:]
    res = [f(x[(slice(None),) + j], *(a[j] for a in args))
           for j in np.ndindex(batch)]
    if out == 'f':
        return np.reshape(np.asarray(res), batch)
    if out == 'g':
        return np.reshape(np.stack(res, axis=-1), x.shape)
    fs = np.reshape(np.asarray([r[0] for r in res]), batch)
    gs = np.reshape(np.stack([r[1] for r in res], axis=-1), x.shape)
    return fs, gs


def loop_vectorize(f, out='f'):
    """Vectorize a function of a single problem by looping over problems.

    Function values are bitwise identical to those of the scalar function, so
    results can be compared exactly to those of `minimize`. The result is
    pickleable if `f` is (for tests with process-based `workers`).
    """
    return functools.partial(_loop_vectorized, f, out)


f_vec = loop_vectorize(f_scalar)
g_vec = loop_vectorize(g_scalar, out='g')
fg_vec = loop_vectorize(fg_scalar, out='fg')


def assert_same_as_minimize(res, x0, args=(), jac=None, bounds=None,
                            exact=True, **kwds):
    """Compare each problem of `res` with the result of `minimize`."""
    x0 = np.asarray(x0)
    batch = res.fun.shape
    # the batch axes of x0 (all but the first) are aligned to the right
    x0 = np.reshape(x0, x0.shape[:1] + (1,)*(len(batch) - x0.ndim + 1)
                    + x0.shape[1:])
    x0 = np.broadcast_to(x0, x0.shape[:1] + batch)
    args = [np.broadcast_to(a, batch) for a in args]
    for j in np.ndindex(batch):
        jx = (slice(None),) + j
        b = None
        if bounds is not None:
            lb = np.broadcast_to(np.moveaxis(np.atleast_1d(bounds.lb), 0, -1),
                                 batch + x0.shape[:1])
            ub = np.broadcast_to(np.moveaxis(np.atleast_1d(bounds.ub), 0, -1),
                                 batch + x0.shape[:1])
            b = Bounds(lb[j], ub[j])
        ref = minimize(f_scalar if jac is not True else fg_scalar,
                       x0[jx], args=tuple(a[j] for a in args),
                       jac=g_scalar if jac == 'callable' else jac, bounds=b,
                       method='L-BFGS-B', **kwds)
        if exact:
            assert_equal(res.x[jx], ref.x)
            assert_equal(res.fun[j], ref.fun)
            assert_equal(res.jac[jx], ref.jac)
            assert_equal(res.nit[j], ref.nit)
            assert_equal(res.nfev[j], ref.nfev)
            assert_equal(res.njev[j], ref.njev)
            assert_equal(res.message[j], ref.message)
        else:
            assert_allclose(res.x[jx], ref.x, rtol=1e-6, atol=1e-8)
            assert_allclose(res.fun[j], ref.fun, rtol=1e-6, atol=1e-12)
        assert_equal(res.success[j], ref.success)


class TestMinimizeBatched:

    @pytest.mark.parametrize('jac', [True, 'callable', None,
                                     '2-point', '3-point', 'cs'])
    @pytest.mark.parametrize('bounds', [None, Bounds(-0.5, [0.7, 0.3, 1.0])])
    def test_equivalence_with_minimize(self, jac, bounds):
        # Each problem must follow exactly the iterations of `minimize`
        rng = np.random.default_rng(3249823498)
        x0 = rng.uniform(-2, 2, size=(3, 6))
        a = np.linspace(-1, 1, 6)
        fun = fg_vec if jac is True else f_vec
        res = minimize_batched(fun, x0, args=(a,), bounds=bounds,
                               jac=g_vec if jac == 'callable' else jac)
        assert np.all(res.success)
        assert_equal(res.status, 0)
        assert_same_as_minimize(res, x0, args=(a,), jac=jac, bounds=bounds)

    def test_vectorized_rosen(self):
        # Truly vectorized function; values may differ from the single-problem
        # function by rounding, so compare with tolerance.
        rng = np.random.default_rng(2398420398)
        x0 = rng.uniform(-2, 2, size=(5, 2, 3))
        res = minimize_batched(rosen, x0, jac=rosen_der)
        assert res.x.shape == (5, 2, 3)
        assert res.jac.shape == (5, 2, 3)
        for key in ['fun', 'nit', 'nfev', 'njev', 'status', 'success',
                    'message']:
            assert res[key].shape == (2, 3)
        assert res.nit.dtype == np.int32
        assert res.success.dtype == np.bool_
        for j in np.ndindex(2, 3):
            ref = minimize(rosen, x0[(slice(None),) + j], jac=rosen_der,
                           method='L-BFGS-B')
            assert_allclose(res.x[(slice(None),) + j], ref.x, atol=1e-4)
            assert_allclose(res.fun[j], ref.fun, atol=1e-8)

    def test_single_problem(self):
        # x0 of shape (n,) is a single problem; results are 0-d
        x0 = np.array([1.5, -0.5, 0.25])
        res = minimize_batched(f_vec, x0, jac=g_vec)
        assert res.x.shape == (3,)
        assert res.fun.shape == ()
        assert res.status.shape == ()
        ref = minimize(f_scalar, x0, jac=g_scalar, method='L-BFGS-B')
        assert_equal(res.x, ref.x)
        assert_equal(res.nit, ref.nit)

    def test_args_compression(self):
        # `fun` is called for the active problems only, and `args` are
        # compressed accordingly; `args` can be used to identify the problems.
        B = 7
        scale = np.arange(1, B + 1)
        ids = np.arange(B)
        seen = []

        def fun(x, scale, ids):
            assert x.shape[1:] == scale.shape == ids.shape
            seen.append(ids.copy())
            return scale * np.sum((x - ids)**2, axis=0), 2*scale*(x - ids)

        x0 = np.zeros((2, B))
        res = minimize_batched(fun, x0, jac=True, args=(scale, ids),
                               tolerances=dict(gtol=1e-10))
        assert_allclose(res.x, np.broadcast_to(ids, (2, B)), atol=1e-8)
        # The first call evaluates all problems, later calls only some of them
        assert_equal(seen[0], ids)
        assert min(len(s) for s in seen) < B
        # Each problem is evaluated exactly `nfev` times
        counts = np.bincount(np.concatenate(seen), minlength=B)
        assert_equal(counts, res.nfev)

    def test_kwargs(self):
        rng = np.random.default_rng(23498723)
        x0 = rng.uniform(-2, 2, size=(3, 4))
        a = np.linspace(-1, 1, 4)

        def fun(x, *, a):
            return f_vec(x, a)

        res = minimize_batched(fun, x0, jac=None, kwargs=dict(a=a))
        assert_same_as_minimize(res, x0, args=(a,))

    @pytest.mark.parametrize('x0_shape, a_shape, batch',
                             [((3,), (5,), (5,)),
                              ((3, 1), (2, 4), (2, 4)),
                              ((3, 4), (2, 1), (2, 4))])
    def test_broadcasting(self, x0_shape, a_shape, batch):
        rng = np.random.default_rng(4598734)
        x0 = rng.uniform(-2, 2, size=x0_shape)
        a = rng.uniform(-1, 1, size=a_shape)
        res = minimize_batched(f_vec, x0, jac=g_vec, args=(a,))
        assert res.x.shape == (3,) + batch
        assert res.fun.shape == batch
        assert_same_as_minimize(res, x0, args=(a,), jac='callable')

    def test_bounds_per_problem(self):
        # Different bounds for each problem
        rng = np.random.default_rng(9834598)
        x0 = rng.uniform(-1, 1, size=(3, 4))
        lb = rng.uniform(-1, -0.2, size=(3, 4))
        ub = rng.uniform(0.1, 0.5, size=(3, 4))
        ub[0, 0] = np.inf
        lb[1, 1] = -np.inf
        bounds = Bounds(lb, ub)
        res = minimize_batched(f_vec, x0, jac=g_vec, bounds=bounds)
        assert np.all((res.x >= lb) & (res.x <= ub))
        assert_same_as_minimize(res, x0, jac='callable', bounds=bounds)

        # a single bound for all variables of each problem
        bounds = Bounds(-0.1*np.arange(1, 5)[np.newaxis], np.inf)
        res = minimize_batched(f_vec, x0, jac=g_vec, bounds=bounds)
        assert_same_as_minimize(res, x0, jac='callable', bounds=bounds)

    def test_x0_outside_bounds_is_clipped(self):
        x0 = np.array([[3., -3.], [0.5, 0.5]])
        bounds = Bounds([-1, -1], [1, 1])
        x0_seen = []

        def fun(x):
            x0_seen.append(x.copy())
            return f_vec(x)

        res = minimize_batched(fun, x0, jac=None, bounds=bounds)
        assert_equal(x0_seen[0], np.clip(x0, -1, 1))
        assert_same_as_minimize(res, x0, bounds=bounds)

    def test_fixed_variables(self):
        rng = np.random.default_rng(5098345)
        x0 = rng.uniform(-1, 1, size=(3, 4))
        bounds = Bounds([0.25, -1, -1], [0.25, 1, 1])
        # With an analytical gradient, the result is identical to `minimize`
        res = minimize_batched(f_vec, x0, jac=g_vec, bounds=bounds)
        assert_equal(res.x[0], 0.25)
        assert_same_as_minimize(res, x0, jac='callable', bounds=bounds)
        # `minimize` removes fixed variables when finite differences are used
        res = minimize_batched(f_vec, x0, jac='3-point', bounds=bounds)
        assert_equal(res.x[0], 0.25)
        assert_equal(res.jac[0], 0)
        assert_same_as_minimize(res, x0, jac='3-point', bounds=bounds,
                                exact=False)

    def test_maxiter(self):
        rng = np.random.default_rng(34985734)
        x0 = rng.uniform(-2, 2, size=(4, 5))
        for maxiter in [0, 1, 5]:
            res = minimize_batched(rosen, x0, jac=rosen_der, maxiter=maxiter)
            assert_equal(res.status, -2)
            assert_equal(res.nit, max(maxiter, 1))
            assert np.all(res.message ==
                          "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT")

        x0 = rng.uniform(-2, 2, size=(3, 5))
        res = minimize_batched(f_vec, x0, jac=g_vec, maxiter=3)
        assert_same_as_minimize(res, x0, jac='callable', options={'maxiter': 3})

    def test_maxfun(self):
        rng = np.random.default_rng(239847)
        x0 = rng.uniform(-2, 2, size=(3, 5))
        res = minimize_batched(f_vec, x0, jac=None, options={'maxfun': 20})
        assert_equal(res.status, -2)
        assert np.all(res.message ==
                      "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT")
        assert_same_as_minimize(res, x0, options={'maxfun': 20})

    def test_abnormal_termination(self):
        # A wrong gradient makes the line search fail
        def fun(x):
            return np.sum(x**2, axis=0), -2*x

        x0 = np.array([[1., 2.], [2., 3.]])
        res = minimize_batched(fun, x0, jac=True, options={'maxls': 1})
        assert_equal(res.status, -1)
        assert not np.any(res.success)
        assert np.all(np.char.startswith(res.message, "ABNORMAL"))
        # the reported function value and gradient belong to `res.x`
        f, g = fun(res.x)
        assert_equal(res.fun, f)
        assert_equal(res.jac, g)

    def test_non_finite(self):
        # Problems with NaN at the initial guess stop immediately; inf is
        # handled by the line search just like in `minimize`.
        def f1(x, a):
            g = 2*(x - 3)
            if a == 0:
                return np.nan, np.full_like(x, np.nan)
            if a == 1 and np.any(x > 1.5):
                return np.inf, g
            return np.sum((x - 3)**2), g

        f = loop_vectorize(f1, out='fg')
        x0 = np.zeros((2, 3))
        a = np.array([0, 1, 2])
        res = minimize_batched(f, x0, args=(a,), jac=True)
        assert_equal(res.status, [-3, 0, 0])
        assert_equal(res.nfev[0], 1)
        assert np.isnan(res.fun[0])
        assert "NON-FINITE" in res.message[0]
        assert_equal(res.x[:, 0], x0[:, 0])
        for j in [1, 2]:
            ref = minimize(f1, x0[:, j], args=(a[j],), jac=True,
                           method='L-BFGS-B')
            assert_equal(res.x[:, j], ref.x)
            assert_equal(res.nfev[j], ref.nfev)
        # problem 1 backtracked from points where the function is infinite
        assert res.fun[1] > res.fun[2]

    def test_callback(self):
        rng = np.random.default_rng(9823745)
        x0 = rng.uniform(-2, 2, size=(4, 6))
        x0[:, 0] = 1  # the minimum: converges right away
        calls = []

        def callback(res):
            calls.append(res)
            assert 'message' not in res
            assert res.x.shape == (4, 6)
            if len(calls) == 1:
                # called after the evaluation at the initial guesses
                assert_equal(res.x, x0)
                assert_equal(res.nit, 0)
                assert_equal(res.fun, rosen(x0))
            in_progress = res.status == 1
            assert_equal(res.success, res.status == 0)
            assert np.all(in_progress | (res.status == 0))
            if len(calls) == 5:
                raise StopIteration

        res = minimize_batched(rosen, x0, jac=rosen_der, callback=callback)
        assert len(calls) == 5
        assert_equal(res.status[0], 0)
        assert_equal(res.status[1:], -4)
        assert np.all(res.message[1:] == "STOP: CALLBACK REQUESTED HALT")
        # the result reports the last accepted iterate, as did the callback
        assert_equal(res.x, calls[-1].x)
        assert_equal(res.fun, calls[-1].fun)
        assert_equal(res.nit, calls[-1].nit)

    def test_callback_results_are_snapshots(self):
        # results passed to the callback must not be modified by later
        # iterations (they must not share memory with the optimizer's state)
        rng = np.random.default_rng(2398457)
        x0 = rng.uniform(-2, 2, size=(3, 4))
        stored, copies = [], []

        def callback(res):
            stored.append(res)
            copies.append({k: np.copy(v) for k, v in res.items()})

        res = minimize_batched(rosen, x0, jac=rosen_der, callback=callback)
        assert len(stored) > 2
        for r, c in zip(stored, copies):
            for k in c:
                assert_equal(r[k], c[k])
        # and the final result does not alias the last intermediate result
        assert not np.shares_memory(res.x, stored[-1].x)

    def test_callback_iterates_match_minimize(self):
        x0 = np.array([-1.2, 1.0, 0.3])
        xs_batched, xs_ref = [], []

        def callback(res):
            if not xs_batched or not np.array_equal(res.x, xs_batched[-1]):
                xs_batched.append(res.x.copy())

        def callback_ref(intermediate_result):
            xs_ref.append(intermediate_result.x.copy())

        minimize_batched(f_vec, x0, jac=g_vec, callback=callback)
        minimize(f_scalar, x0, jac=g_scalar, method='L-BFGS-B',
                 callback=callback_ref)
        # the first callback reports x0, `minimize` reports iterates only
        assert_equal(np.asarray(xs_batched[1:]), np.asarray(xs_ref))

    @pytest.mark.parametrize('jac, s', [(None, 3), ('2-point', 3),
                                        ('3-point', 6), ('cs', 3)])
    def test_finite_difference_calls(self, jac, s):
        # All finite difference points are evaluated in one call
        shapes = []

        def fun(x, a):
            shapes.append((x.shape, a.shape))
            return f_vec(x, a)

        rng = np.random.default_rng(3450987)
        x0 = rng.uniform(-2, 2, size=(3, 5))
        a = np.linspace(-1, 1, 5)
        res = minimize_batched(fun, x0, jac=jac, args=(a,))
        # calls alternate between function values and finite differences
        for (sx, sa), (sx2, sa2) in zip(shapes[::2], shapes[1::2]):
            k = sx[1]
            assert sx == (3, k) and sa == (k,)
            assert sx2 == (3, k, s) and sa2 == (k, 1)
        assert len(shapes) == 2*np.max(res.njev)
        assert_equal(res.nfev, res.njev*(1 + s))

    def test_float32(self):
        dtypes = []

        def fun(x):
            dtypes.append(x.dtype)
            return rosen(x), rosen_der(x)

        x0 = np.asarray([[-1.2, 0.5], [1.0, 0.5]], dtype=np.float32)
        res = minimize_batched(fun, x0, jac=True)
        assert set(dtypes) == {np.dtype(np.float32)}
        assert res.x.dtype == np.float32
        assert res.fun.dtype == np.float32
        assert_allclose(res.x, 1, atol=1e-2)

        # finite differences use steps appropriate for float32
        res = minimize_batched(rosen, x0, jac='3-point')
        assert res.x.dtype == np.float32
        assert_allclose(res.x, 1, atol=2e-2)

    def test_integer_x0(self):
        res = minimize_batched(rosen, np.asarray([[0, 2], [0, 2]]),
                               jac=rosen_der)
        assert res.x.dtype == np.float64
        assert_allclose(res.x, 1, atol=1e-4)

    def test_empty_batch(self):
        def fun(x):
            raise AssertionError("must not be called")

        res = minimize_batched(fun, np.ones((3, 0)))
        assert res.x.shape == (3, 0)
        assert res.fun.shape == (0,)
        assert res.status.shape == (0,)

    def test_input_validation(self):
        x0 = np.ones((2, 3))
        with pytest.raises(ValueError, match="`fun` must be callable"):
            minimize_batched(None, x0)
        with pytest.raises(ValueError, match="Unknown method"):
            minimize_batched(rosen, x0, method='BFGS')
        with pytest.raises(ValueError, match="`jac` must be"):
            minimize_batched(rosen, x0, jac='4-point')
        with pytest.raises(ValueError, match="`callback` must be callable"):
            minimize_batched(rosen, x0, callback='yes')
        with pytest.raises(ValueError, match="at least one dimension"):
            minimize_batched(rosen, np.float64(1.))
        with pytest.raises(ValueError, match="at least one dimension"):
            minimize_batched(rosen, np.ones((0, 3)))
        with pytest.raises(ValueError, match="must be real"):
            minimize_batched(rosen, x0 + 1j)
        with pytest.raises(ValueError, match="broadcastable"):
            minimize_batched(rosen, x0, args=(np.ones(4),))
        with pytest.raises(ValueError, match="Unknown tolerances"):
            minimize_batched(rosen, x0, tolerances={'xtol': 1e-3})
        with pytest.raises(ValueError, match="must be non-negative"):
            minimize_batched(rosen, x0, tolerances={'gtol': -1})
        with pytest.raises(ValueError, match="Unknown options"):
            minimize_batched(rosen, x0, options={'disp': True})
        with pytest.raises(ValueError, match="must be positive"):
            minimize_batched(rosen, x0, options={'maxcor': 0})
        with pytest.raises(ValueError, match="must be an integer"):
            minimize_batched(rosen, x0, options={'maxls': 1.5})
        with pytest.raises(ValueError, match="`maxiter` must be"):
            minimize_batched(rosen, x0, maxiter=-1)
        with pytest.raises(ValueError, match="Option `eps`"):
            minimize_batched(rosen, x0, options={'eps': [1e-8]*3})
        with pytest.raises(TypeError, match="`bounds` must be"):
            minimize_batched(rosen, x0, bounds=[(0, 1), (0, 1)])
        with pytest.raises(ValueError, match="lower bounds is greater"):
            minimize_batched(rosen, x0, bounds=Bounds([0, 1], [1, 0]))
        with pytest.raises(ValueError, match="first axis of `bounds.lb`"):
            minimize_batched(rosen, x0, bounds=Bounds(np.zeros(3), 1))
        with pytest.raises(ValueError, match="cannot be broadcast"):
            minimize_batched(rosen, x0, bounds=Bounds(np.zeros((2, 4)), 1))

        # invalid outputs of `fun` and `jac`
        with pytest.raises(ValueError, match="`fun` must return an array"):
            minimize_batched(lambda x: np.sum(x), x0, jac=rosen_der)
        with pytest.raises(ValueError, match="`jac` must return an array"):
            minimize_batched(rosen, x0, jac=lambda x: rosen_der(x)[0])
        with pytest.raises(ValueError, match="must return a tuple"):
            minimize_batched(rosen, x0, jac=True)

    def test_thread_safety(self):
        # Independent calls may run concurrently in several threads
        rng = np.random.default_rng(23458734)
        x0s = [rng.uniform(-2, 2, size=(3, 4)) for _ in range(4)]
        refs = [minimize_batched(f_vec, x0, jac=g_vec) for x0 in x0s]
        barrier = threading.Barrier(4)

        def run(x0):
            barrier.wait()
            return minimize_batched(f_vec, x0, jac=g_vec)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(run, x0s))
        for res, ref in zip(results, refs):
            assert_equal(res.x, ref.x)
            assert_equal(res.nfev, ref.nfev)


def _thread_map(func, iterable):
    # a map-like callable based on threads (no pickling required)
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(func, iterable))


class TestWorkers:

    def assert_same(self, res, ref):
        for key in ['x', 'fun', 'jac', 'nit', 'nfev', 'njev', 'status',
                    'success', 'message']:
            assert_equal(res[key], ref[key])
            assert res[key].dtype == ref[key].dtype
            assert res[key].shape == ref[key].shape

    @pytest.mark.parametrize('workers', [2, _thread_map])
    @pytest.mark.parametrize('batch', [(7,), (2, 5)])
    def test_equal_to_serial(self, workers, batch):
        # chunks are solved independently; results equal the serial ones
        rng = np.random.default_rng(29384756)
        x0 = rng.uniform(-1, 1, size=(3,) + batch)
        a = rng.uniform(-1, 1, size=batch)
        lb = rng.uniform(-1, -0.2, size=(3,) + batch)
        bounds = Bounds(lb, 0.8)
        kwds = dict(jac=g_vec, args=(a,), bounds=bounds, maxiter=50)
        ref = minimize_batched(f_vec, x0, **kwds)
        res = minimize_batched(f_vec, x0, workers=workers, **kwds)
        self.assert_same(res, ref)

    def test_finite_differences(self):
        rng = np.random.default_rng(2938475)
        x0 = rng.uniform(-2, 2, size=(3, 9))
        a = np.linspace(-1, 1, 9)
        ref = minimize_batched(f_vec, x0, jac='3-point', args=(a,))
        res = minimize_batched(f_vec, x0, jac='3-point', args=(a,),
                               workers=_thread_map)
        self.assert_same(res, ref)

    def test_chunking(self):
        # one chunk per CPU (at least two problems per chunk), sizes add up
        sizes = []

        def recording_map(func, iterable):
            tasks = list(iterable)
            sizes.extend(task[1].shape[1] for task in tasks)
            return _thread_map(func, tasks)

        B = 101
        x0 = np.random.default_rng(23984).uniform(-2, 2, size=(2, B))
        res = minimize_batched(rosen, x0, jac=rosen_der, workers=recording_map)
        n_cpus = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        assert len(sizes) == min(n_cpus, B // 2) or n_cpus == 1
        assert sum(sizes) in (B, 0)
        assert min(sizes, default=2) >= 2
        assert np.all(res.success)

    def test_small_batch_is_not_distributed(self):
        def failing_map(func, iterable):
            raise AssertionError("must not be called")

        x0 = np.zeros((2, 3))
        res = minimize_batched(rosen, x0, jac=rosen_der, workers=failing_map)
        assert_allclose(res.x, 1, atol=1e-4)

    def test_input_validation(self):
        x0 = np.zeros((2, 8))
        for workers in [0, -2, 1.5, 'all']:
            with pytest.raises(ValueError, match="`workers` must be"):
                minimize_batched(rosen, x0, jac=rosen_der, workers=workers)
        with pytest.raises(ValueError, match="cannot be used together"):
            minimize_batched(rosen, x0, jac=rosen_der, workers=2,
                             callback=lambda res: None)
        # no parallelism: `workers=1` and `None` are allowed with a callback
        minimize_batched(rosen, x0, jac=rosen_der, workers=1,
                         callback=lambda res: None)


def f_xp(x, a):
    # Array API compatible test function with minimum at x = 1 (any `a`)
    xp = array_namespace(x, a)
    return xp.sum((x - 1)**2 * (1 + x**2), axis=0) + a


def g_xp(x, a):
    return 2*(x - 1)*(1 + x**2) + 2*x*(x - 1)**2


@make_xp_test_case(minimize_batched)
class TestMinimizeBatchedArrayAPI:

    @pytest.mark.parametrize('dtype', ['float32', 'float64'])
    @pytest.mark.parametrize('jac', ['analytic', 'fun', '2-point'])
    def test_basic(self, xp, dtype, jac):
        dtype = getattr(xp, dtype)
        rng = np.random.default_rng(3984573)
        x0_np = rng.uniform(-1.5, 1.5, size=(3, 2, 2))
        a_np = np.asarray([[0., 1.], [1., 2.]])
        x0 = xp.asarray(x0_np, dtype=dtype)
        a = xp.asarray(a_np, dtype=dtype)

        def fun(x, a):
            assert array_namespace(x, a) is array_namespace(x0)
            assert xp_device(x) == xp_device(x0)
            assert x.dtype == dtype
            return f_xp(x, a)

        def fun_and_jac(x, a):
            return fun(x, a), g_xp(x, a)

        kwds = {'analytic': dict(jac=g_xp),
                'fun': dict(fun=fun_and_jac, jac=True),
                '2-point': dict(jac='2-point')}[jac]
        f = kwds.pop('fun', fun)
        res = minimize_batched(f, x0, args=(a,), **kwds)

        # reference: NumPy in the same precision
        np_dtype = np.float32 if dtype == xp.float32 else np.float64
        ref = minimize_batched(f_xp, x0_np.astype(np_dtype),
                               args=(a_np.astype(np_dtype),),
                               jac=g_xp if jac != '2-point' else '2-point')
        atol = 1e-3 if dtype == xp.float32 else 1e-8
        xp_assert_close(res.x, xp.asarray(ref.x, dtype=dtype), atol=atol)
        xp_assert_close(res.x, xp.ones_like(res.x), atol=100*atol)
        xp_assert_close(res.fun, xp.asarray(ref.fun, dtype=dtype),
                        atol=atol, rtol=atol)
        xp_assert_equal(res.success, xp.ones((2, 2), dtype=xp.bool))
        xp_assert_equal(res.status, xp.zeros((2, 2), dtype=xp.int32))
        assert res.nit.dtype == xp.int32
        assert res.message.shape == (2, 2)

    def test_workers(self, xp):
        # parallel chunks with a thread map return results in the namespace
        # and on the device of x0, equal to the serial results
        x0 = xp.asarray(np.random.default_rng(4).uniform(-1.5, 1.5, (3, 8)))
        a = xp.zeros(8, dtype=x0.dtype)
        ref = minimize_batched(f_xp, x0, jac=g_xp, args=(a,))
        res = minimize_batched(f_xp, x0, jac=g_xp, args=(a,),
                               workers=_thread_map)
        assert array_namespace(res.x) is array_namespace(x0)
        assert xp_device(res.x) == xp_device(x0)
        xp_assert_equal(res.x, ref.x)
        xp_assert_equal(res.nit, ref.nit)

    def test_torch_autograd(self, xp):
        # Outputs that are part of an autograd graph are supported
        if not is_torch(xp):
            pytest.skip("torch-specific test")
        import torch

        def fun(x, a):
            x = x.detach().requires_grad_(True)
            f = torch.sum((x - 1)**2 * (1 + x**2), dim=0) + a
            g, = torch.autograd.grad(f.sum(), x)
            return f, g

        x0 = xp.asarray(np.full((2, 3), 0.5), dtype=xp.float64)
        a = xp.zeros(3, dtype=xp.float64)
        res = minimize_batched(fun, x0, jac=True, args=(a,))
        xp_assert_close(res.x, xp.ones((2, 3), dtype=xp.float64), atol=1e-6)
