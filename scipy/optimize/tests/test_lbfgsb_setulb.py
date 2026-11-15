import numpy as np
import pytest
from scipy.optimize import _lbfgsb, minimize
from scipy._lib._util import HAS_ILP64


def objfun(x):
    """simplified objective func to test lbfgsb bound violation"""
    x0 = [0.8750000000000278,
          0.7500000000000153,
          0.9499999999999722,
          0.8214285714285992,
          0.6363636363636085]
    x1 = [1.0, 0.0, 1.0, 0.0, 0.0]
    x2 = [1.0,
          0.0,
          0.9889733043149325,
          0.0,
          0.026353554421041155]
    x3 = [1.0,
          0.0,
          0.9889917442915558,
          0.0,
          0.020341986743231205]

    f0 = 5163.647901211178
    f1 = 5149.8181642072905
    f2 = 5149.379332309634
    f3 = 5149.374490771297

    g0 = np.array([-0.5934820547965749,
                   1.6251549718258351,
                   -71.99168459202559,
                   5.346636965797545,
                   37.10732723092604])
    g1 = np.array([-0.43295349282641515,
                   1.008607936794592,
                   18.223666726602975,
                   31.927010036981997,
                   -19.667512518739386])
    g2 = np.array([-0.4699874455100256,
                   0.9466285353668347,
                   -0.016874360242016825,
                   48.44999161133457,
                   5.819631620590712])
    g3 = np.array([-0.46970678696829116,
                   0.9612719312174818,
                   0.006129809488833699,
                   48.43557729419473,
                   6.005481418498221])

    if np.allclose(x, x0):
        f = f0
        g = g0
    elif np.allclose(x, x1):
        f = f1
        g = g1
    elif np.allclose(x, x2):
        f = f2
        g = g2
    elif np.allclose(x, x3):
        f = f3
        g = g3
    else:
        raise ValueError(
            'Simplified objective function not defined '
            'at requested point')
    return (np.copy(f), np.copy(g))


def test_setulb_floatround():
    """test if setulb() violates bounds

    checks for violation due to floating point rounding error
    """

    n = 5
    m = 10
    factr = 1e7
    pgtol = 1e-5
    maxls = 20
    int_dtype = np.int64 if HAS_ILP64 else np.int32

    nbd = np.full(shape=(n,), fill_value=2, dtype=int_dtype)
    low_bnd = np.zeros(n, dtype=np.float64)
    upper_bnd = np.ones(n, dtype=np.float64)

    x0 = np.array(
        [0.8750000000000278,
         0.7500000000000153,
         0.9499999999999722,
         0.8214285714285992,
         0.6363636363636085])
    x = np.copy(x0)

    f = np.array(0.0, dtype=np.float64)
    g = np.zeros(n, dtype=np.float64)

    wa = np.zeros(2*m*n + 5*n + 11*m*m + 8*m, dtype=np.float64)
    iwa = np.zeros(3*n, dtype=int_dtype)
    task = np.zeros(2, dtype=int_dtype)
    ln_task = np.zeros(2, dtype=int_dtype)
    lsave = np.zeros(4, dtype=int_dtype)
    isave = np.zeros(44, dtype=int_dtype)
    dsave = np.zeros(29, dtype=np.float64)

    for n_iter in range(7):  # 7 steps required to reproduce error
        f, g = objfun(x)

        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr, pgtol, wa,
                       iwa, task, lsave, isave, dsave, maxls, ln_task)

        assert (x <= upper_bnd).all() and (x >= low_bnd).all(), (
            "_lbfgsb.setulb() stepped to a point outside of the bounds")


def test_gh_issue18730():
    # issue 18730 reported that l-bfgs-b did not work with objectives
    # returning single precision gradient arrays
    def fun_single_precision(x):
        x = x.astype(np.float32)
        return np.sum(x**2), (2*x)

    res = minimize(fun_single_precision, x0=np.array([1., 1.]), jac=True,
                   method="l-bfgs-b")
    np.testing.assert_allclose(res.fun, 0., atol=1e-15)


def test_abnormal_termination_fun_consistent_with_x():
    # On abnormal termination of the line search, setulb restores x and g to
    # the last accepted iterate. `res.fun` must correspond to `res.x`, too
    # (it used to be the function value at the last trial point).
    def fun(x):
        # gradient with the wrong sign makes every line search fail
        return x @ x, -2*x

    for maxls in [1, 3]:
        res = minimize(fun, x0=np.array([1., 2.]), jac=True,
                       method="l-bfgs-b", options=dict(maxls=maxls))
        assert res.message.startswith("ABNORMAL")
        assert not res.success
        f, g = fun(res.x)
        assert res.fun == f
        np.testing.assert_array_equal(res.jac, g)


def _batch_state(B, n, m):
    int_dtype = np.int64 if HAS_ILP64 else np.int32
    return dict(
        f=np.zeros(B), g=np.zeros((B, n)),
        wa=np.zeros((B, 2*m*n + 5*n + 11*m*m + 8*m)),
        iwa=np.zeros((B, 3*n), dtype=int_dtype),
        task=np.zeros((B, 2), dtype=int_dtype),
        lsave=np.zeros((B, 4), dtype=int_dtype),
        isave=np.zeros((B, 44), dtype=int_dtype),
        dsave=np.zeros((B, 29)),
        ln_task=np.zeros((B, 2), dtype=int_dtype),
        xk=np.zeros((B, n)), fk=np.zeros(B), gk=np.zeros((B, n)),
        nit=np.zeros(B, dtype=np.int64), nfev=np.zeros(B, dtype=np.int64),
    )


def _call_setulb_batch(m, x, l, u, nbd, st, idx, factr=1e7, pgtol=1e-5,
                       maxls=20, maxiter=15000, maxfun=15000):
    _lbfgsb.setulb_batch(m, x, l, u, nbd, st['f'], st['g'], factr, pgtol,
                         st['wa'], st['iwa'], st['task'], st['lsave'],
                         st['isave'], st['dsave'], maxls, st['ln_task'],
                         st['xk'], st['fk'], st['gk'], st['nit'], st['nfev'],
                         maxiter, maxfun, idx)


def _quartic(x, c):
    return np.sum((x - c)**2 + 0.1*x**4), 2*(x - c) + 0.4*x**3


def test_setulb_batch_matches_setulb():
    # Driving B problems with setulb_batch gives the same states as driving
    # each problem separately with setulb.
    n, m, B = 4, 5, 3
    factr, pgtol, maxls, maxiter = 1e7, 1e-5, 20, 6
    int_dtype = np.int64 if HAS_ILP64 else np.int32
    rng = np.random.default_rng(23498)
    x0 = rng.uniform(-2, 2, size=(B, n))
    cs = rng.uniform(-1, 1, size=(B, n))
    l = np.full((B, n), -1.5)
    u = np.full((B, n), 1.5)
    nbd = np.full((B, n), 2, dtype=int_dtype)
    nbd[:, 0] = 0
    nbd[:, 1] = 1

    # reference: separate problems, with the loop of _minimize_lbfgsb
    refs = []
    for i in range(B):
        st = _batch_state(1, n, m)
        st = {k: v[0] for k, v in st.items()}
        x = x0[i].copy()
        f, g, nit, nfev = 0.0, np.zeros(n), 0, 0
        while True:
            _lbfgsb.setulb(m, x, l[i], u[i], nbd[i], f, g, factr, pgtol,
                           st['wa'], st['iwa'], st['task'], st['lsave'],
                           st['isave'], st['dsave'], maxls, st['ln_task'])
            if st['task'][0] == 3:
                f, g = _quartic(x, cs[i])
                nfev += 1
            elif st['task'][0] == 1:
                nit += 1
                if nit >= maxiter:
                    st['task'][:] = (5, 504)
            else:
                break
        refs.append((x, st['task'].copy(), st['isave'].copy(), nit, nfev))

    x = x0.copy()
    st = _batch_state(B, n, m)
    active = np.arange(B)
    while active.size:
        _call_setulb_batch(m, x, l, u, nbd, st, active, factr=factr,
                           pgtol=pgtol, maxls=maxls, maxiter=maxiter)
        active = active[st['task'][active, 0] == 3]
        for i in active:
            st['f'][i], st['g'][i] = _quartic(x[i], cs[i])
            st['nfev'][i] += 1

    for i, (x_ref, task_ref, isave_ref, nit_ref, nfev_ref) in enumerate(refs):
        np.testing.assert_array_equal(x[i], x_ref)
        np.testing.assert_array_equal(st['xk'][i], x_ref)
        np.testing.assert_array_equal(st['task'][i], task_ref)
        np.testing.assert_array_equal(st['isave'][i], isave_ref)
        assert st['nit'][i] == nit_ref
        assert st['nfev'][i] == nfev_ref
        assert st['fk'][i] == _quartic(x_ref, cs[i])[0]
        assert np.all((x[i] >= l[i]) | (nbd[i] == 0))
        assert np.all((x[i] <= u[i]) | (nbd[i] < 2))


def test_setulb_batch_maxfun_and_empty_idx():
    n, m, B = 2, 3, 2
    int_dtype = np.int64 if HAS_ILP64 else np.int32
    x = np.ones((B, n))
    l = np.zeros((B, n))
    u = np.zeros((B, n))
    nbd = np.zeros((B, n), dtype=int_dtype)
    st = _batch_state(B, n, m)

    # an empty index array is a no-op
    before = {k: v.copy() for k, v in st.items()}
    _call_setulb_batch(m, x, l, u, nbd, st, np.empty(0, dtype=np.intp))
    for k in st:
        np.testing.assert_array_equal(st[k], before[k])

    # maxfun: the problem stops at the first new iterate
    idx = np.arange(B)
    _call_setulb_batch(m, x, l, u, nbd, st, idx)
    assert np.all(st['task'][:, 0] == 3)
    st['f'][:], st['g'][:] = 2.0, 2*x
    st['nfev'][:] = [1, 100]
    while np.any(st['task'][:, 0] == 3):
        idx = np.nonzero(st['task'][:, 0] == 3)[0]
        _call_setulb_batch(m, x, l, u, nbd, st, idx, maxfun=50)
        for i in idx:
            if st['task'][i, 0] == 3:
                st['f'][i], st['g'][i] = np.sum(x[i]**2), 2*x[i]
                st['nfev'][i] += 1
    np.testing.assert_array_equal(st['task'][1], [5, 502])
    assert st['nit'][1] == 1


def test_setulb_batch_input_validation():
    n, m, B = 3, 2, 2
    int_dtype = np.int64 if HAS_ILP64 else np.int32
    x = np.zeros((B, n))
    l = np.zeros((B, n))
    u = np.ones((B, n))
    nbd = np.full((B, n), 2, dtype=int_dtype)
    idx = np.arange(B, dtype=np.intp)

    def call(**changes):
        st = _batch_state(B, n, m)
        args = dict(m=m, x=x.copy(), l=l, u=u, nbd=nbd, idx=idx)
        for k, v in changes.items():
            (args if k in args else st)[k] = v
        _call_setulb_batch(args['m'], args['x'], args['l'], args['u'],
                           args['nbd'], st, args['idx'])

    call()  # valid
    with pytest.raises(_lbfgsb.error, match=r"\(wa\) must have shape"):
        call(wa=np.zeros((B, 10)))
    with pytest.raises(_lbfgsb.error, match=r"\(nbd\) must be"):
        call(nbd=nbd.astype(np.float64))
    with pytest.raises(_lbfgsb.error, match=r"\(g\) must be"):
        call(g=np.zeros((n, B)).T)  # not C-contiguous
    with pytest.raises(_lbfgsb.error, match=r"\(f\) must have shape"):
        call(f=np.zeros(B + 1))
    with pytest.raises(_lbfgsb.error, match=r"\(nit\) must be"):
        call(nit=np.zeros(B, dtype=np.int32))
    x_ro = x.copy()
    x_ro.flags.writeable = False
    with pytest.raises(_lbfgsb.error, match=r"\(x\) must be writeable"):
        call(x=x_ro)
    with pytest.raises(_lbfgsb.error, match="out-of-range"):
        call(idx=np.asarray([0, B], dtype=np.intp))
    with pytest.raises(_lbfgsb.error, match="out-of-range"):
        call(idx=np.asarray([-1], dtype=np.intp))
    with pytest.raises(ValueError, match="m must be"):
        call(m=0)
