#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "numpy/arrayobject.h"
#include <math.h>
#include "src/lbfgsb.h"

#define PYERR(errobj,message) {PyErr_SetString(errobj,message); return NULL;}

static PyObject* lbfgsb_error;

static char doc_setulb[] = (
    "setulb(m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, task, lsave, isave, dsave, maxls, ln_task)\n\n"
    "Limited-memory Broyden-Fletcher-Goldfarb-Shanno with bounds.\n\n"
    "Low-level interface to the L-BFGS-B optimization algorithm. "
    "Performs a single iteration step.\n\n"
    "Parameters\n"
    "----------\n"
    "m : int\n"
    "    Number of limited-memory variable metric corrections.\n"
    "x : ndarray, shape (n,)\n"
    "    Current solution vector, float64. Modified in-place.\n"
    "l : ndarray, shape (n,)\n"
    "    Lower bounds on variables, float64.\n"
    "u : ndarray, shape (n,)\n"
    "    Upper bounds on variables, float64.\n"
    "nbd : ndarray, shape (n,)\n"
    "    Bound type for each variable, int32. 0: unbounded, 1: only lower, "
    "2: both, 3: only upper.\n"
    "f : float\n"
    "    Current objective function value.\n"
    "g : ndarray, shape (n,)\n"
    "    Gradient of objective function at current point, float64.\n"
    "factr : float\n"
    "    Factor for convergence test on objective function.\n"
    "pgtol : float\n"
    "    Projected gradient tolerance for convergence.\n"
    "wa : ndarray\n"
    "    Workspace array for internal computations, float64.\n"
    "iwa : ndarray\n"
    "    Integer workspace array for internal computations, int32.\n"
    "task : ndarray\n"
    "    Task control array encoding algorithm state, int32. Modified in-place.\n"
    "lsave : ndarray\n"
    "    Logical state array, int32. Modified in-place.\n"
    "isave : ndarray\n"
    "    Integer state array, int32. Modified in-place.\n"
    "dsave : ndarray\n"
    "    Real state array, float64. Modified in-place.\n"
    "maxls : int\n"
    "    Maximum number of line search steps per iteration.\n"
    "ln_task : ndarray\n"
    "    Task string length array, int32.\n\n"
    "Returns\n"
    "-------\n"
    "None\n"
    "    Modifies x, task, lsave, isave, and dsave in-place.\n\n"
);

static PyObject*
lbfgsb_setulb(PyObject *self, PyObject *args)
{

    Py_ssize_t maxls_input, m_input;
    CBLAS_INT m, n, maxls;
    double f, factr, pgtol;
    CBLAS_INT *nbd, *iwa, *isave, *lsave, *taskptr, *ln_taskptr;
    double *x, *l, *u, *g, *wa, *dsave;

    PyArrayObject *ap_x=NULL, *ap_l=NULL, *ap_u=NULL, *ap_g=NULL, *ap_nbd=NULL;
    PyArrayObject *ap_wa=NULL, *ap_iwa=NULL, *ap_task=NULL, *ap_ln_task=NULL;
    PyArrayObject *ap_lsave=NULL, *ap_isave=NULL, *ap_dsave=NULL;

    // Check argument types
    // m,x,l,u,nbd,f,g,factr,pgtol,wa,iwa,task,lsave,isave,dsave,maxls,ln_task
    if (!(PyArg_ParseTuple(args,
                           "nO!O!O!O!dO!ddO!O!O!O!O!O!nO!",
                           &m_input,                                // n
                           &PyArray_Type, (PyObject **)&ap_x,       // O!
                           &PyArray_Type, (PyObject **)&ap_l,       // O!
                           &PyArray_Type, (PyObject **)&ap_u,       // O!
                           &PyArray_Type, (PyObject **)&ap_nbd,     // O!
                           &f,                                      // d
                           &PyArray_Type, (PyObject **)&ap_g,       // O!
                           &factr,                                  // d
                           &pgtol,                                  // d
                           &PyArray_Type, (PyObject **)&ap_wa,      // O!
                           &PyArray_Type, (PyObject **)&ap_iwa,     // O!
                           &PyArray_Type, (PyObject **)&ap_task,    // O!
                           &PyArray_Type, (PyObject **)&ap_lsave,   // O!
                           &PyArray_Type, (PyObject **)&ap_isave,   // O!
                           &PyArray_Type, (PyObject **)&ap_dsave,   // O!
                           &maxls_input,                            // n
                           &PyArray_Type, (PyObject **)&ap_ln_task  // O!
                           ))) { return NULL; }

    maxls = (CBLAS_INT)maxls_input;
    m = (CBLAS_INT)m_input;
    int tcode = sizeof(CBLAS_INT) == sizeof(npy_int32) ? NPY_INT32 : NPY_INT64;

    // Check if arrays are contiguous and with the right dtype.
    // All arrays are 1D hence both F and C contiguous flags are set to True.
    if (!(PyArray_IS_C_CONTIGUOUS(ap_x)) || (PyArray_TYPE(ap_x) != NPY_FLOAT64))
    {
        PYERR(lbfgsb_error, " Argument (x) must be a contiguous array of type float64.");
    }
    if (!(PyArray_IS_C_CONTIGUOUS(ap_l)) || (PyArray_TYPE(ap_l) != NPY_FLOAT64))
    {
        PYERR(lbfgsb_error, " Argument (l) must be a contiguous array of type float64.");
    }
    if (!(PyArray_IS_C_CONTIGUOUS(ap_u)) || (PyArray_TYPE(ap_u) != NPY_FLOAT64))
    {
        PYERR(lbfgsb_error, " Argument (u) must be a contiguous array of type float64.");
    }
    if (!(PyArray_IS_C_CONTIGUOUS(ap_wa)) || (PyArray_TYPE(ap_wa) != NPY_FLOAT64))
    {
        PYERR(lbfgsb_error, " Argument (wa) must be a contiguous array of type float64.");
    }
    if (!(PyArray_IS_C_CONTIGUOUS(ap_iwa)) || (PyArray_TYPE(ap_iwa) != tcode))
    {
        PYERR(lbfgsb_error, " Argument (iwa) must be a contiguous array of type int32.");
    }
    if (!(PyArray_IS_C_CONTIGUOUS(ap_dsave)) || (PyArray_TYPE(ap_dsave) != NPY_FLOAT64))
    {
        PYERR(lbfgsb_error, " Argument (dsave) must be a contiguous array of type float64.");
    }
    if (!(PyArray_IS_C_CONTIGUOUS(ap_nbd)) || (PyArray_TYPE(ap_nbd) != tcode))
    {
        PYERR(lbfgsb_error, " Argument (nbd) must be a contiguous array of type int32.");
    }
    if (!(PyArray_IS_C_CONTIGUOUS(ap_task)) || (PyArray_TYPE(ap_task) != tcode))
    {
        PYERR(lbfgsb_error, " Argument (task) must be a contiguous array of type int32.");
    }
    if (!(PyArray_IS_C_CONTIGUOUS(ap_lsave)) || (PyArray_TYPE(ap_lsave) != tcode))
    {
        PYERR(lbfgsb_error, " Argument (lsave) must be a contiguous array of type int32.");
    }
    if (!(PyArray_IS_C_CONTIGUOUS(ap_isave)) || (PyArray_TYPE(ap_isave) != tcode))
    {
        PYERR(lbfgsb_error, " Argument (isave) must be a contiguous array of type int32.");
    }
    if (!(PyArray_IS_C_CONTIGUOUS(ap_ln_task)) || (PyArray_TYPE(ap_ln_task) != tcode))
    {
        PYERR(lbfgsb_error, " Argument (ln_task) must be a contiguous array of type int32.");
    }
    n = PyArray_DIMS(ap_x)[0];

    x = (double *)PyArray_DATA(ap_x);
    l = (double *)PyArray_DATA(ap_l);
    u = (double *)PyArray_DATA(ap_u);
    nbd = (CBLAS_INT *)PyArray_DATA(ap_nbd);
    g = (double *)PyArray_DATA(ap_g);
    wa = (double *)PyArray_DATA(ap_wa);
    iwa = (CBLAS_INT *)PyArray_DATA(ap_iwa);
    lsave = (CBLAS_INT *)PyArray_DATA(ap_lsave);
    isave = (CBLAS_INT *)PyArray_DATA(ap_isave);
    dsave = (double *)PyArray_DATA(ap_dsave);
    taskptr = (CBLAS_INT *)PyArray_DATA(ap_task);
    ln_taskptr = (CBLAS_INT *)PyArray_DATA(ap_ln_task);

    // Make the function call
    setulb(n, m, x, l, u, nbd, &f, g, factr, pgtol, wa, iwa, taskptr, lsave,
           isave, dsave, maxls, ln_taskptr);

    Py_RETURN_NONE;
}


static char doc_setulb_batch[] = (
    "setulb_batch(m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, task, lsave, "
    "isave, dsave, maxls, ln_task, xk, fk, gk, nit, nfev, maxiter, maxfun, idx)\n\n"
    "Advance many independent L-BFGS-B problems.\n\n"
    "Low-level, batched interface to the L-BFGS-B algorithm. All state arrays\n"
    "are stacked along the first axis, with one row per problem (B problems of\n"
    "n variables each). Each problem ``i`` in ``idx`` is advanced until it\n"
    "requests a new evaluation of f and g (``task[i, 0] == 3``) or terminates.\n"
    "NEW_X events are handled internally: ``nit[i]`` is incremented, the\n"
    "accepted iterate is copied to ``xk[i]``, ``fk[i]``, ``gk[i]``, and the run\n"
    "is stopped when ``nit[i] >= maxiter`` (task message 504) or\n"
    "``nfev[i] > maxfun`` (task message 502). The GIL is released while the\n"
    "problems are advanced.\n\n"
    "Parameters\n"
    "----------\n"
    "m : int\n"
    "    Number of limited-memory variable metric corrections.\n"
    "x, l, u : ndarray, shape (B, n), float64\n"
    "    Current points (modified in-place), lower and upper bounds.\n"
    "nbd : ndarray, shape (B, n), int32 (int64 for ILP64 builds)\n"
    "    Bound type for each variable (0: unbounded, 1: only lower, 2: both,\n"
    "    3: only upper).\n"
    "f : ndarray, shape (B,), float64\n"
    "    Function values. Modified in-place.\n"
    "g : ndarray, shape (B, n), float64\n"
    "    Gradients. Modified in-place.\n"
    "factr, pgtol : float\n"
    "    Convergence tolerances.\n"
    "wa : ndarray, shape (B, 2*m*n + 5*n + 11*m*m + 8*m), float64\n"
    "iwa : ndarray, shape (B, 3*n), integer\n"
    "task, ln_task : ndarray, shape (B, 2), integer\n"
    "lsave : ndarray, shape (B, 4), integer\n"
    "isave : ndarray, shape (B, 44), integer\n"
    "dsave : ndarray, shape (B, 29), float64\n"
    "    Per-problem workspace and state, modified in-place.\n"
    "maxls : int\n"
    "    Maximum number of line search steps per iteration.\n"
    "xk, gk : ndarray, shape (B, n), float64; fk : ndarray, shape (B,), float64\n"
    "    Last accepted iterate of every problem. Modified in-place.\n"
    "nit : ndarray, shape (B,), int64\n"
    "    Number of iterations per problem. Modified in-place.\n"
    "nfev : ndarray, shape (B,), int64\n"
    "    Number of function evaluations per problem (read-only).\n"
    "maxiter, maxfun : int\n"
    "    Iteration and function evaluation limits.\n"
    "idx : ndarray, shape (K,), intp\n"
    "    Unique indices of the problems to advance.\n\n"
    "Returns\n"
    "-------\n"
    "None\n\n"
    "Notes\n"
    "-----\n"
    "Different calls may run concurrently from several threads as long as they\n"
    "do not share any state arrays.\n"
);


/*
 * Check that `arr` is an aligned, C-contiguous array of dtype `typenum` with
 * `ndim` dimensions of extent (d0[, d1]); and writeable if requested.
 */
static int
check_batch_arr(PyArrayObject *arr, const char *name, int typenum, int ndim,
                npy_intp d0, npy_intp d1, int writeable)
{
    if (PyArray_NDIM(arr) != ndim ||
        PyArray_DIMS(arr)[0] != d0 ||
        (ndim == 2 && PyArray_DIMS(arr)[1] != d1))
    {
        if (ndim == 1) {
            PyErr_Format(lbfgsb_error, "Argument (%s) must have shape (%zd,).",
                         name, (Py_ssize_t)d0);
        } else {
            PyErr_Format(lbfgsb_error, "Argument (%s) must have shape (%zd, %zd).",
                         name, (Py_ssize_t)d0, (Py_ssize_t)d1);
        }
        return -1;
    }
    if (PyArray_TYPE(arr) != typenum ||
        !PyArray_IS_C_CONTIGUOUS(arr) ||
        !PyArray_ISALIGNED(arr))
    {
        PyArray_Descr *descr = PyArray_DescrFromType(typenum);
        PyErr_Format(lbfgsb_error,
                     "Argument (%s) must be an aligned, C-contiguous array of "
                     "dtype %c%d.", name, descr->kind, (int)PyDataType_ELSIZE(descr));
        Py_DECREF(descr);
        return -1;
    }
    if (writeable && !PyArray_ISWRITEABLE(arr))
    {
        PyErr_Format(lbfgsb_error, "Argument (%s) must be writeable.", name);
        return -1;
    }
    return 0;
}


static PyObject*
lbfgsb_setulb_batch(PyObject *self, PyObject *args)
{
    Py_ssize_t m_input, maxls_input;
    long long maxiter, maxfun;
    double factr, pgtol;

    PyArrayObject *ap_x=NULL, *ap_l=NULL, *ap_u=NULL, *ap_nbd=NULL, *ap_f=NULL;
    PyArrayObject *ap_g=NULL, *ap_wa=NULL, *ap_iwa=NULL, *ap_task=NULL;
    PyArrayObject *ap_lsave=NULL, *ap_isave=NULL, *ap_dsave=NULL;
    PyArrayObject *ap_ln_task=NULL, *ap_xk=NULL, *ap_fk=NULL, *ap_gk=NULL;
    PyArrayObject *ap_nit=NULL, *ap_nfev=NULL, *ap_idx=NULL;

    if (!(PyArg_ParseTuple(args,
                           "nO!O!O!O!O!O!ddO!O!O!O!O!O!nO!O!O!O!O!O!LLO!",
                           &m_input,
                           &PyArray_Type, (PyObject **)&ap_x,
                           &PyArray_Type, (PyObject **)&ap_l,
                           &PyArray_Type, (PyObject **)&ap_u,
                           &PyArray_Type, (PyObject **)&ap_nbd,
                           &PyArray_Type, (PyObject **)&ap_f,
                           &PyArray_Type, (PyObject **)&ap_g,
                           &factr,
                           &pgtol,
                           &PyArray_Type, (PyObject **)&ap_wa,
                           &PyArray_Type, (PyObject **)&ap_iwa,
                           &PyArray_Type, (PyObject **)&ap_task,
                           &PyArray_Type, (PyObject **)&ap_lsave,
                           &PyArray_Type, (PyObject **)&ap_isave,
                           &PyArray_Type, (PyObject **)&ap_dsave,
                           &maxls_input,
                           &PyArray_Type, (PyObject **)&ap_ln_task,
                           &PyArray_Type, (PyObject **)&ap_xk,
                           &PyArray_Type, (PyObject **)&ap_fk,
                           &PyArray_Type, (PyObject **)&ap_gk,
                           &PyArray_Type, (PyObject **)&ap_nit,
                           &PyArray_Type, (PyObject **)&ap_nfev,
                           &maxiter,
                           &maxfun,
                           &PyArray_Type, (PyObject **)&ap_idx
                           ))) { return NULL; }

    const int tcode = sizeof(CBLAS_INT) == sizeof(npy_int32) ? NPY_INT32 : NPY_INT64;
    const double cblas_int_max = sizeof(CBLAS_INT) == sizeof(npy_int32) ?
                                 (double)NPY_MAX_INT32 : (double)NPY_MAX_INT64;

    if (PyArray_NDIM(ap_x) != 2) {
        PYERR(lbfgsb_error, "Argument (x) must be a 2-D array of shape (B, n).");
    }
    const npy_intp B = PyArray_DIMS(ap_x)[0];
    const npy_intp n_in = PyArray_DIMS(ap_x)[1];

    if (n_in < 1) { PYERR(lbfgsb_error, "The number of variables must be >= 1."); }
    if (m_input < 1) { PYERR(PyExc_ValueError, "m must be >= 1."); }
    if (maxls_input < 1) { PYERR(PyExc_ValueError, "maxls must be >= 1."); }

    // The workspace offsets are stored in CBLAS_INT inside setulb; make sure
    // the workspace size of a single problem is representable.
    const double lwa_d = 2.0*m_input*n_in + 5.0*n_in + 11.0*m_input*m_input + 8.0*m_input;
    if ((double)n_in*3.0 > cblas_int_max || lwa_d > cblas_int_max ||
        (double)m_input > cblas_int_max || (double)maxls_input > cblas_int_max ||
        lwa_d*(double)(B > 0 ? B : 1) > (double)NPY_MAX_INTP)
    {
        PYERR(PyExc_ValueError, "Problem size too large for the L-BFGS-B workspace.");
    }
    const npy_intp m = (npy_intp)m_input;
    const npy_intp lwa = 2*m*n_in + 5*n_in + 11*m*m + 8*m;

    if (check_batch_arr(ap_x, "x", NPY_FLOAT64, 2, B, n_in, 1) ||
        check_batch_arr(ap_l, "l", NPY_FLOAT64, 2, B, n_in, 0) ||
        check_batch_arr(ap_u, "u", NPY_FLOAT64, 2, B, n_in, 0) ||
        check_batch_arr(ap_nbd, "nbd", tcode, 2, B, n_in, 0) ||
        check_batch_arr(ap_f, "f", NPY_FLOAT64, 1, B, 0, 1) ||
        check_batch_arr(ap_g, "g", NPY_FLOAT64, 2, B, n_in, 1) ||
        check_batch_arr(ap_wa, "wa", NPY_FLOAT64, 2, B, lwa, 1) ||
        check_batch_arr(ap_iwa, "iwa", tcode, 2, B, 3*n_in, 1) ||
        check_batch_arr(ap_task, "task", tcode, 2, B, 2, 1) ||
        check_batch_arr(ap_lsave, "lsave", tcode, 2, B, 4, 1) ||
        check_batch_arr(ap_isave, "isave", tcode, 2, B, 44, 1) ||
        check_batch_arr(ap_dsave, "dsave", NPY_FLOAT64, 2, B, 29, 1) ||
        check_batch_arr(ap_ln_task, "ln_task", tcode, 2, B, 2, 1) ||
        check_batch_arr(ap_xk, "xk", NPY_FLOAT64, 2, B, n_in, 1) ||
        check_batch_arr(ap_fk, "fk", NPY_FLOAT64, 1, B, 0, 1) ||
        check_batch_arr(ap_gk, "gk", NPY_FLOAT64, 2, B, n_in, 1) ||
        check_batch_arr(ap_nit, "nit", NPY_INT64, 1, B, 0, 1) ||
        check_batch_arr(ap_nfev, "nfev", NPY_INT64, 1, B, 0, 0))
    {
        return NULL;
    }
    if (PyArray_NDIM(ap_idx) != 1) {
        PYERR(lbfgsb_error, "Argument (idx) must be a 1-D array.");
    }
    const npy_intp n_idx = PyArray_DIMS(ap_idx)[0];
    if (check_batch_arr(ap_idx, "idx", NPY_INTP, 1, n_idx, 0, 0)) { return NULL; }

    const npy_intp *idx = (const npy_intp *)PyArray_DATA(ap_idx);
    for (npy_intp k = 0; k < n_idx; k++)
    {
        if (idx[k] < 0 || idx[k] >= B) {
            PYERR(lbfgsb_error, "Argument (idx) contains an out-of-range index.");
        }
    }

    double *x = (double *)PyArray_DATA(ap_x);
    double *l = (double *)PyArray_DATA(ap_l);
    double *u = (double *)PyArray_DATA(ap_u);
    CBLAS_INT *nbd = (CBLAS_INT *)PyArray_DATA(ap_nbd);
    double *f = (double *)PyArray_DATA(ap_f);
    double *g = (double *)PyArray_DATA(ap_g);
    double *wa = (double *)PyArray_DATA(ap_wa);
    CBLAS_INT *iwa = (CBLAS_INT *)PyArray_DATA(ap_iwa);
    CBLAS_INT *task = (CBLAS_INT *)PyArray_DATA(ap_task);
    CBLAS_INT *lsave = (CBLAS_INT *)PyArray_DATA(ap_lsave);
    CBLAS_INT *isave = (CBLAS_INT *)PyArray_DATA(ap_isave);
    double *dsave = (double *)PyArray_DATA(ap_dsave);
    CBLAS_INT *ln_task = (CBLAS_INT *)PyArray_DATA(ap_ln_task);
    double *xk = (double *)PyArray_DATA(ap_xk);
    double *fk = (double *)PyArray_DATA(ap_fk);
    double *gk = (double *)PyArray_DATA(ap_gk);
    int64_t *nit = (int64_t *)PyArray_DATA(ap_nit);
    const int64_t *nfev = (const int64_t *)PyArray_DATA(ap_nfev);

    // No Python objects are touched below; the argument tuple keeps all
    // arrays alive for the duration of the call.
    Py_BEGIN_ALLOW_THREADS
    setulb_batch((CBLAS_INT)n_in, (CBLAS_INT)m, (ptrdiff_t)n_idx,
                 (const ptrdiff_t *)idx, x, l, u, nbd, f, g, factr, pgtol,
                 wa, (ptrdiff_t)lwa, iwa, task, lsave, isave, dsave,
                 (CBLAS_INT)maxls_input, ln_task, xk, fk, gk, nit, nfev,
                 (int64_t)maxiter, (int64_t)maxfun);
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}


static struct PyMethodDef lbfgsb_module_methods[] = {
  {"setulb", lbfgsb_setulb, METH_VARARGS, doc_setulb},
  {"setulb_batch", lbfgsb_setulb_batch, METH_VARARGS, doc_setulb_batch},
  {NULL,     NULL,          0,            NULL}
};


static int module_exec(PyObject *module) {

    if (_import_array() < 0) { return -1; }

    lbfgsb_error = PyErr_NewException("_lbfgsb.error", NULL, NULL);
    if (lbfgsb_error == NULL) { return -1; }

    if (PyModule_AddObject(module, "error", lbfgsb_error) < 0) {
        Py_DECREF(lbfgsb_error);
        return -1;
    }
#if Py_GIL_DISABLED
    PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
#endif
    return 0;
}


static struct PyModuleDef_Slot _lbfgsb_slots[] = {
    {Py_mod_exec, module_exec},
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#if PY_VERSION_HEX >= 0x030d00f0  // Python 3.13+
    // signal that this module supports running without an active GIL
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};


static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_lbfgsb",
    .m_size = 0,
    .m_methods = lbfgsb_module_methods,
    .m_slots = _lbfgsb_slots
};


PyMODINIT_FUNC
PyInit__lbfgsb(void)
{
    return PyModuleDef_Init(&moduledef);
}
