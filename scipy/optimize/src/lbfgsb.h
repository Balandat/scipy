#ifndef __LBFGSB_H
#define __LBFGSB_H

#include "blaslapack_declarations.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>

void setulb(
    CBLAS_INT n, CBLAS_INT m, double* x, double* l, double* u, CBLAS_INT* nbd, double* f, double* g, double factr,
    double pgtol, double* wa, CBLAS_INT* iwa, CBLAS_INT* task, CBLAS_INT* lsave, CBLAS_INT* isave, double* dsave, CBLAS_INT maxls, CBLAS_INT* ln_task
);

/*
 * Batched driver around setulb for many independent problems of equal size.
 *
 * All state arrays are stacked row-major with one row per problem, i.e. the
 * state of problem i starts at x + i*n, wa + i*lwa, iwa + i*3n, task + i*2,
 * lsave + i*4, isave + i*44, dsave + i*29, ln_task + i*2, etc.
 *
 * Every problem listed in idx[0..n_idx-1] is advanced (repeated setulb calls)
 * until it either requests a new function/gradient evaluation (task FG) or
 * terminates. NEW_X events are handled here: nit[i] is incremented, the
 * accepted iterate is copied to xk/fk/gk and the maxiter/maxfun stopping
 * rules of scipy.optimize._lbfgsb_py._minimize_lbfgsb are applied.
 * nfev is read-only and holds the number of function evaluations per problem.
 */
void setulb_batch(
    CBLAS_INT n, CBLAS_INT m, ptrdiff_t n_idx, const ptrdiff_t* idx,
    double* x, double* l, double* u, CBLAS_INT* nbd, double* f, double* g,
    double factr, double pgtol, double* wa, ptrdiff_t lwa, CBLAS_INT* iwa,
    CBLAS_INT* task, CBLAS_INT* lsave, CBLAS_INT* isave, double* dsave,
    CBLAS_INT maxls, CBLAS_INT* ln_task, double* xk, double* fk, double* gk,
    int64_t* nit, const int64_t* nfev, int64_t maxiter, int64_t maxfun
);


#endif /* ifndef */
