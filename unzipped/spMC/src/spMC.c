/* Declaration of prototypes and library used by spMC */
#include "spMC.h"

/* Inclusion of functions for direction classification */
#include "wfun.h"

/* Inclusion of functions for empirical transiograms */
#include "trans.h"

/* Inclusion of functions for multinomial categorical simulation */
#include "mcs.h"

void cEmbedLen(int *n, int *nc, double *coords, int *locId, int *data, int *cemoc, double *maxcens, double *tlen) {
  /* Computing strata lengths (i.e. mean lengths)
          *n - sample size
         *nc - number of columns
     *coords - matrix of coordinates
       *data - vector of data
      *locId - vector of locations id
      *cemoc - category of an embedded occurency
    *maxcens - censured lengths
       *tlen - lengths vector */

  int i, j, inde = 0, bgn = 0;
  double tmptl;

  for (i = 1; i < *n; i++) {
    if (locId[i - 1] == locId[i] && data[i - 1] == data[i]) {
      tmptl = pow(coords[i - 1] - coords[i], 2.0);
      for (j = 1; j < *nc; j++) {
        tmptl += pow(coords[*n * j + i - 1] - coords[*n * j + i], 2.0);
      }
      tlen[inde] += sqrt(tmptl);
    }
    else {
      if (locId[i - 1] == locId[i]) {
        tmptl = pow(coords[i - 1] - coords[i], 2.0);
        for (j = 1; j < *nc; j++) {
          tmptl += pow(coords[*n * j + i - 1] - coords[*n * j + i], 2.0);
        }
        maxcens[inde] = sqrt(tmptl);
      }
      else {
        maxcens[inde] = maxcens[inde - 1];
        for (j = inde - 1; j > bgn; j--) {
          maxcens[j] += maxcens[j - 1];
          maxcens[j] /= 2.0;
        }
        bgn = inde + 1;
      }
      cemoc[inde] = data[i - 1];
      ++inde;
    }
  }
  maxcens[inde] = maxcens[inde - 1];
  for (j = inde - 1; j > bgn; j--) {
    maxcens[j] += maxcens[j - 1];
    maxcens[j] /= 2.0;
  }
  cemoc[inde] = data[i - 1];
  *n = ++inde;
}

void cEmbedOc(int *n, int *nc, int *nk, double *coords, int *locId, int *data, int *cemoc, double *tlen) {
  /* Counting embedded occurences
          *n - sample size
         *nc - number of columns
         *nk - number of categories
     *coords - matrix of coordinates
       *data - vector of data
      *locId - vector of locations id
      *cemoc - embedded occurences
       *tlen - mean length vector */

  int i, j;
  double tmptl;

#if __VOPENMP
  #pragma omp parallel sections default(shared) private(i)
  {
    #pragma omp section
    {
#endif
      ++cemoc[data[0] - 1];
      for (i = 1; i < *n; i++) {
        if (locId[i - 1] != locId[i] || data[i - 1] != data[i]) ++cemoc[data[i] - 1];
      }
#if __VOPENMP
    }
    #pragma omp section
    {
#endif
      for (i = 1; i < *n; i++) {
        if (locId[i - 1] == locId[i] && data[i - 1] == data[i]) {
          tmptl = pow(coords[i - 1] - coords[i], 2.0);
          for (j = 1; j < *nc; j++) {
            tmptl += pow(coords[*n * j + i - 1] - coords[*n * j + i], 2.0);
          }
          tlen[data[i] - 1] += sqrt(fabs(tmptl));
        }
      }
#if __VOPENMP
    }
  }
  #pragma omp parallel for default(shared) private(i) schedule(static, 1)
#endif
  for (i = 0; i < *nk; i++) tlen[i] /= (double) cemoc[i];
}

void cEmbedTrans(int *n, int *nk, int *locId, int *data, int *tcount) {
  /* Counting embedded occurences
          *n - sample size
         *nk - number of categories
      *locId - vector of locations id
       *data - vector of data
     *tcount - transition occurences */

  int i = 0;

  for (; i < *n - 1; i++) {
    if (locId[i] == locId[i + 1] && data[i] != data[i + 1]) {
      ++tcount[data[i] - 1 + *nk * (data[i + 1] - 1)];
    }
  }

}

void embedTProbs(int *nk, double *tp) {
  /* Counting embedded transition probabilities
         *nk - number of categories
         *tp - transition occurences to transform in probabilities */

  int i, j;
  double rwsum;

#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j, rwsum) schedule(static, 1)
#endif
  for (i = 0; i < *nk; i++) {
    rwsum = 0.0;

    for (j = 0; j < *nk; j++) {
      rwsum += tp[i + *nk * j];
    }

    for (j = 0; j < *nk; j++) {
      tp[i + *nk * j] /= rwsum;
    }
  }

}

void ellinter(int *nc, int *nk, double *hh, double *coef, double *Rmat) {
  /* Ellipsoidally interpolation of transition rates matrix
         *nc - dimension of the sample space
         *nk - number of categories
         *hh - vector of scaled lags
       *coef - 'list' of estimated coefficients
       *Rmat - interpolated matrix */

  int i, j, k;

  for (j = 0; j < *nk; j++) {
    for (k = 0; k < *nk; k++) {
      Rmat[*nk * j + k] = 0.0;
      if (j != k) {
        for (i = 0; i < *nc; i++) {
          Rmat[*nk * j + k] += pow(coef[*nk * *nk * i + *nk * j + k] * hh[i], 2.0);
        }
        Rmat[*nk * j + k] = pow(fabs(Rmat[*nk * j + k]), 0.5);
      }
    }
  }

}

void getDst(int *nc, int *nr, double *site, double *coords, double *wgmLags) {
  /* Differences among simulation site coordinates and dataset coordinates
         *nc - dimension of coordinates
         *nr - dimension of the sample space
       *site - simulation coordinates
     *coords - sample coordinates
    *wgmLags - resulting lags */

  int i, j;

#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j) schedule(static, 1)
#endif
  for (i = 0; i < *nr; i++) {
    wgmLags[i] = coords[i] - site[0];
    wgmLags[*nc * *nr + i] = pow(wgmLags[i], 2.0);
    for (j = 1; j < *nc; j++) {
      wgmLags[*nr * j + i] = coords[*nr * j + i] - site[j];
      wgmLags[*nc * *nr + i] += pow(wgmLags[*nr * j + i], 2.0);
    }
    wgmLags[*nc * *nr + i] = sqrt(wgmLags[*nc * *nr + i]);
  }

}

void fastMatProd(int *nr, int *ni, double *mat1, int *nc, double *mat2, double *res) {
  /* HPC Matrix Product
         *nr - number of rows (1st matrix)
         *ni - number of colunms or rows (1st matrix and 2 matrix)
       *mat1 - 1st matrix
         *nc - number of colunms (2nd matrix)
       *mat2 - 2nd matrix
        *res - resulting matrix */

  int i, j, k;

#if __VOPENMP
  #if (_OPENMP > 201100)
    #pragma omp parallel for default(shared) private(i, j, k) collapse(2) schedule(static, 1)
  #else
    #if (_OPENMP > 200800)
      #pragma omp parallel for default(shared) private(i, j, k) schedule(static, 1)
    #endif
  #endif
#endif
  for (i = 0; i < *nr; i++) {
    for (j = 0; j < *nc; j++) {
      res[*nr * j + i] = mat1[i] * mat2[*ni * j];
      for (k = 1; k < *ni; k++) {
        res[*nr * j + i] += mat1[*nr * k + i] * mat2[*ni * j + k];
      }
    }
  }

}

void jointProbs(int *hmany, int *nk, int *ndata, double *Tmat, double *pProbs) {
  /* "Posterior" transition probabilities approximation
      *hmany - number of neighbours
         *nk - number of categories
  *ndata - vector of neighbour categories
       *Tmat - array of transition probabilities
     *pProbs - vector of probabilities */

  int i, j;
  double mysum = 0.0;

#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j) schedule(static, 1)
#endif
  for (j = 0; j < *nk; j++) {
    pProbs[j] = 1.0;
    for (i = 0; i < *hmany; i++) {
      if (!i) {
        pProbs[j] = pProbs[j] * Tmat[*nk * *nk * i + ndata[i] - 1 + *nk * j];
      }
      else {
        pProbs[j] = pProbs[j] * Tmat[*nk * *nk * i + j + *nk * (ndata[i] - 1)];
      }
    }
  }
  for (i = 0; i < *nk; i++) {
    mysum = mysum + pProbs[i];
  }
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i) schedule(static, 1)
#endif
  for (i = 0; i < *nk; i++) {
    pProbs[i] /= mysum;
  }  
}

void tsimCate(int *nk, int *n, double *prhat, int *initSim) {
  /* Simulation from transition probabilities estimation
         *nk - number of categories
          *n - sample size
      *prhat - matrix of posterior probabilities 
    *initSim - conditional simulation vector */

  int i, j;
  double rnd;
  
  /* Computing the cumulative probabilities */
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j) schedule(static, 1)
#endif
  for (i = 0; i < *n; i++) {
    for (j = 1; j < *nk; j++) {
      prhat[*n * j + i] += prhat[*n * (j - 1) + i];
    }
  }

  GetRNGstate();
  for (i = 0; i < *n; i++) {
    rnd = unif_rand();
    for (j = 0; j < *nk; j++) {
      if (rnd < prhat[*n * j + i]) {
        initSim[i] = j + 1;
        break;
      }
    }
  }
  PutRNGstate();
}

SEXP isOmp() {
  SEXP ans;
  PROTECT(ans = allocVector(LGLSXP, 1));
  #if __VOPENMP
    LOGICAL(ans)[0] = TRUE;
  #else
    LOGICAL(ans)[0] = FALSE;
  #endif
  UNPROTECT(1);
  return ans;
}

void getNumCores(int *n) {
  /* Get the max number of CPU cores
          *n - num of cores */
  #if __VOPENMP
    *n = omp_get_num_procs();
  #else
    *n = 1;
  #endif
}

void getNumSlaves(int *n) {
  /* Get the number of threads to use
          *n - num of threads */
  #if __VOPENMP
    #pragma omp parallel default(shared)
    {
      #pragma omp master
        *n = omp_get_num_threads();
    }
  #else
    *n = 0;
  #endif
}

void setNumSlaves(int *n) {
  /* Set the number of threads to use
          *n - num of threads */
  #if __VOPENMP
    if (omp_get_num_procs() < *n) {
      *n = omp_get_num_procs();
      omp_set_num_threads(*n);
    }
    else if(*n > 0) {
      omp_set_num_threads(*n);
    }
    else {
      omp_set_num_threads(1);
      *n = 1;
    }
  #else
    *n = 0;
  #endif
}

void fastSVDprod(double *vti, double *di, double *ui, int *nc) {
  /* HPC of SVD product
       *vti - matrix whose rows contain the right singular vectors
        *di - vector of singular values
        *ui - matrix whose columns contain the left singular vectors
        *nc - number of columns */

  int i, j, k;
  double *myprod;
  
  if ((myprod = (double *) malloc(*nc * *nc * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }

#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j, k) schedule(static, 1)
#endif
  for (i = 0; i < *nc; i++) {
    for (j = 0; j < *nc; j++) {
      vti[*nc * j + i] *= di[j];
    }
    for (j = 0; j < *nc; j++) {
      myprod[*nc * j + i] = 0.0;
      for (k = 0; k < *nc; k++) 
        myprod[*nc * j + i] += vti[*nc * k + i] * ui[*nc * j + k];
    }
  }
  memcpy(ui, myprod, *nc * *nc * sizeof(double));
  free(myprod);
}

SEXP annealingSIM(SEXP maxIt, SEXP old, SEXP x, SEXP grid, SEXP expr, SEXP rho) {
  /* Perform the simulated annealing to optimize the simulation
       maxIt - maximum number of iteration
         old - simulated field to optimize
           x - model to simulate
        grid - simulation grid
        expr - function to optimize
         rho - R enviroment */

  SEXP new, res, prob;
  int i, j, k, hhmm, n, m, nk, ris;
  int *iold, *inew, *xx, *yy, *ctrlvet;
  double *pvet;
  
  PROTECT(old = coerceVector(old, INTSXP));
  PROTECT(maxIt = coerceVector(maxIt, INTSXP));
  n = length(old);
  iold = INTEGER(old);
  PROTECT(new = allocVector(INTSXP, n));
  inew = INTEGER(new);
  PROTECT(res = allocVector(REALSXP, 2));
  defineVar(install("x"), x, rho);
  defineVar(install("grid"), grid, rho);
  prob = VECTOR_ELT(x, 0);
  for (i = 1; i < length(x); i++) {
    if(strcmp(CHAR(STRING_ELT(getAttrib(x, R_NamesSymbol), i)), "prop") == 0) {
      prob = VECTOR_ELT(x, i);
      break;
    }
  }
  nk = length(prob);
  if ((xx = (int *) malloc(n * sizeof(int))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  if ((ctrlvet = (int *) malloc(nk * sizeof(int))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  if ((pvet = (double *) malloc(nk * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }

  pvet[0] = REAL(prob)[0];
  for (i = 1; i < nk; i++) {
    pvet[i] = pvet[i - 1] + REAL(prob)[i];
  }
  GetRNGstate();

  for (i = 0; i < INTEGER(maxIt)[0]; i++) {
    ris = 0;
    hhmm = (int) ceil(n * unif_rand());
    /* set a similar sample into the "new" vector */
    while (ris == 0) {
      if ((yy = (int *) malloc(hhmm * sizeof(int))) == NULL) {
#if __VOPENMP
        #pragma omp critical
#endif
        error("%s", myMemErr);
      } // points to replace
#if __VOPENMP
      #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
      for (j = 0; j < n; j++) {
        xx[j] = j;
        inew[j] = iold[j];
      }
      m = n;
      for (j = 0; j < hhmm; j++) {
        k = (int) floor(m * unif_rand());
        yy[j] = xx[k];
        xx[k] = xx[--m];
      }
      for (k = 0; k < hhmm; k++) {
        for (j = 0; j < nk; j++) if (unif_rand() <= pvet[j]) {
          inew[yy[k]] = j + 1;
          break;
        }
      }
      free(yy);
      /* verify a sub-optimality condition */
      for(j = 0; j < nk; j++) ctrlvet[j] = 0;
      for(j = 0; j < n; j++) ctrlvet[inew[j] - 1] += 1;
      ris = 1;
      for(j = 0; j < nk; j++) if(ctrlvet[j] == 0) ris = 0;
    }

    defineVar(install("pp"), old, rho);
    REAL(res)[0] = REAL(eval(expr, rho))[0];
    defineVar(install("pp"), new, rho);
    REAL(res)[1] = REAL(eval(expr, rho))[0];

    if (REAL(res)[1] <= REAL(res)[0]) {
#if __VOPENMP
      #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
      for (j = 0; j < n; j++) iold[j] = inew[j];
    }
    else {
      if (unif_rand() < exp(- (REAL(res)[1] - REAL(res)[0]) / (i + 1.0))) {
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
        for (j = 0; j < n; j++) iold[j] = inew[j];
      }
    }
  }

  PutRNGstate();
  free(xx);
  free(pvet);
  free(ctrlvet);
  UNPROTECT(4);
  return(old);
}

SEXP geneticSIM(SEXP maxIt, SEXP old, SEXP x, SEXP grid, SEXP expr, SEXP rho) {
  /* Perform the genetic algorithm to optimize the simulation
       maxIt - maximum number of iteration
         old - simulated field to optimize
           x - model to simulate
        grid - simulation grid
        expr - function to optimize
         rho - R enviroment */

  SEXP new, res, prob, child0, child1;
  int i, j, k, hhmm, n, m, nk, ris, pri, pos = 0;
  int *iold, *inew, *xx, *yy, *ctrlvet, *tmpc;
  double tmp;
  double *pvet;
  
  PROTECT(old = coerceVector(old, INTSXP));
  PROTECT(maxIt = coerceVector(maxIt, INTSXP));
  n = length(old);
  iold = INTEGER(old);
  PROTECT(new = allocVector(INTSXP, n));
  inew = INTEGER(new);
  PROTECT(child0 = allocVector(INTSXP, n));
  PROTECT(child1 = allocVector(INTSXP, n));
  PROTECT(res = allocVector(REALSXP, 4));
  defineVar(install("x"), x, rho);
  defineVar(install("grid"), grid, rho);
  prob = VECTOR_ELT(x, 0);
  for (i = 1; i < length(x); i++) {
    if(strcmp(CHAR(STRING_ELT(getAttrib(x, R_NamesSymbol), i)), "prop") == 0) {
      prob = VECTOR_ELT(x, i);
      break;
    }
  }
  nk = length(prob);
  if ((xx = (int *) malloc(n * sizeof(int))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  if ((tmpc = (int *) malloc(n * sizeof(int))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  if ((ctrlvet = (int *) malloc(nk * sizeof(int))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  if ((pvet = (double *) malloc(nk * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  pvet[0] = REAL(prob)[0];
  for (i = 1; i < nk; i++) {
    pvet[i] = pvet[i - 1] + REAL(prob)[i];
  }
  GetRNGstate();

  for (i = 0; i < INTEGER(maxIt)[0]; i++) {
    ris = 0;
    hhmm = (int) ceil(n * unif_rand());
    /* set a similar sample into the "new" vector - Surviver Mutation */
    while (ris == 0) {
      if ((yy = (int *) malloc(hhmm * sizeof(int))) == NULL) {
#if __VOPENMP
        #pragma omp critical
#endif
        error("%s", myMemErr);
      } // points to replace
#if __VOPENMP
      #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
      for (j = 0; j < n; j++) {
        xx[j] = j;
        inew[j] = iold[j];
      }
      m = n;
      for (j = 0; j < hhmm; j++) {
        k = (int) floor(m * unif_rand());
        yy[j] = xx[k];
        xx[k] = xx[--m];
      }
      for (k = 0; k < hhmm; k++) {
        for (j = 0; j < nk; j++) if (unif_rand() <= pvet[j]) {
          inew[yy[k]] = j + 1;
          break;
        }
      }
      free(yy);
      /* verify a sub-optimality condition */
      for(j = 0; j < nk; j++) ctrlvet[j] = 0;
      for(j = 0; j < n; j++) ctrlvet[inew[j] - 1] += 1;
      ris = 1;
      for(j = 0; j < nk; j++) if(ctrlvet[j] == 0) ris = 0;
    }
    /* Crossover */
    for (j = 0; j < n; j++) {
      if (unif_rand() > 0.5) {
        INTEGER(child0)[j] = inew[j];
        INTEGER(child1)[j] = iold[j];
      }
      else {
        INTEGER(child0)[j] = iold[j];
        INTEGER(child1)[j] = inew[j];
      }
    }

    /* Check for the the child0 */
    pri = 0;
#if __VOPENMP
    #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
    for (j = 0; j < n; j++) tmpc[j] = INTEGER(child0)[j];
    for (j = 0; j < nk; j++) ctrlvet[j] = 0;
    for (j = 0; j < n; j++) ctrlvet[tmpc[j] - 1] += 1;
    ris = 1;
    for (j = 0; j < nk; j++) if(ctrlvet[j] == 0) ris = 0;
    while (ris == 0) {
      if ((yy = (int *) malloc(hhmm * sizeof(int))) == NULL) {
#if __VOPENMP
        #pragma omp critical
#endif
        error("%s", myMemErr);
      } // points to replace
#if __VOPENMP
      #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
      for (j = 0; j < n; j++) {
        xx[j] = j;
        if (pri) tmpc[j] = INTEGER(child0)[j];
      }
      if (pri == 0) pri = !pri;
      m = n;
      for (j = 0; j < hhmm; j++) {
        k = (int) floor(m * unif_rand());
        yy[j] = xx[k];
        xx[k] = xx[--m];
      }
      for (k = 0; k < hhmm; k++) {
        for (j = 0; j < nk; j++) if (unif_rand() <= pvet[j]) {
          tmpc[yy[k]] = j + 1;
          break;
        }
      }
      free(yy);
      /* verify a sub-optimality condition */
      for(j = 0; j < nk; j++) ctrlvet[j] = 0;
      for(j = 0; j < n; j++) ctrlvet[tmpc[j] - 1] += 1;
      ris = 1;
      for(j = 0; j < nk; j++) if(ctrlvet[j] == 0) ris = 0;
    }
#if __VOPENMP
    #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
    for (j = 0; j < n; j++) INTEGER(child0)[j] = tmpc[j];

    /* Check for the the child1 */    
    pri = 0;
#if __VOPENMP
    #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
    for (j = 0; j < n; j++) tmpc[j] = INTEGER(child1)[j];
    for (j = 0; j < nk; j++) ctrlvet[j] = 0;
    for (j = 0; j < n; j++) ctrlvet[tmpc[j] - 1] += 1;
    ris = 1;
    for (j = 0; j < nk; j++) if(ctrlvet[j] == 0) ris = 0;
    while (ris == 0) {
      if ((yy = (int *) malloc(hhmm * sizeof(int))) == NULL) {
#if __VOPENMP
        #pragma omp critical
#endif
        error("%s", myMemErr);
      } // points to replace
#if __VOPENMP
      #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
      for (j = 0; j < n; j++) {
        xx[j] = j;
        if (pri) tmpc[j] = INTEGER(child1)[j];
      }
      if (pri == 0) pri = !pri;
      m = n;
      for (j = 0; j < hhmm; j++) {
        k = (int) floor(m * unif_rand());
        yy[j] = xx[k];
        xx[k] = xx[--m];
      }
      for (k = 0; k < hhmm; k++) {
        for (j = 0; j < nk; j++) if (unif_rand() <= pvet[j]) {
          tmpc[yy[k]] = j + 1;
          break;
        }
      }
      free(yy);
      /* verify a sub-optimality condition */
      for(j = 0; j < nk; j++) ctrlvet[j] = 0;
      for(j = 0; j < n; j++) ctrlvet[tmpc[j] - 1] += 1;
      ris = 1;
      for(j = 0; j < nk; j++) if(ctrlvet[j] == 0) ris = 0;
    }
#if __VOPENMP
    #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
    for (j = 0; j < n; j++) INTEGER(child1)[j] = tmpc[j];

    defineVar(install("pp"), old, rho);
    REAL(res)[0] = REAL(eval(expr, rho))[0];
    defineVar(install("pp"), new, rho);
    REAL(res)[1] = REAL(eval(expr, rho))[0];
    defineVar(install("pp"), child0, rho);
    REAL(res)[2] = REAL(eval(expr, rho))[0];
    defineVar(install("pp"), child1, rho);
    REAL(res)[3] = REAL(eval(expr, rho))[0];
    
    tmp = R_PosInf;
    for (j = 0; j < 4; j++) {
      if (REAL(res)[j] < tmp) {
          pos = j;
          tmp = REAL(res)[j];
      }
    }
    switch(pos) {
      case 1:
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
        for (j = 0; j < n; j++) iold[j] = inew[j];
        break;
      case 2:
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
        for (j = 0; j < n; j++) iold[j] = INTEGER(child0)[j];
        break;
      case 3:
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
        for (j = 0; j < n; j++) iold[j] = INTEGER(child1)[j];
        break;
      default:
        break;
    }
  }

  PutRNGstate();
  free(xx);
  free(pvet);
  free(ctrlvet);
  free(tmpc);
  UNPROTECT(6);
  return(old);
}

void expmat(double *x, int *n, double *z) {
  /* Compute the exponential matrix by performing the 'Scaling & Squaring' algorithm
       *x - input matrix
       *n - number of rows (or columns)
       *z - output matrix */

  int h, i, j, k, l, info = 0;
  int *ipiv;
  double s;
  double *nx, *x2, *x4, *x6, *pm, *um, *vm, *tmp;
  double t[] = {0.015, 0.25, 0.95, 2.1};
  double cm[] = {120.0, 30240.0, 17297280.0, 17643225600.0, \
                  60.0, 15120.0,  8648640.0,  8821612800.0, \
                  12.0,  3360.0,  1995840.0,  2075673600.0, \
                   1.0,   420.0,   277200.0,   302702400.0, \
                   0.0,    30.0,    25200.0,    30270240.0, \
                   0.0,     1.0,     1512.0,     2162160.0, \
                   0.0,     0.0,       56.0,      110880.0, \
                   0.0,     0.0,        1.0,        3960.0, \
                   0.0,     0.0,        0.0,          90.0, \
                   0.0,     0.0,        0.0,           1.0};
  double cv[] = {64764752532480000.0, 32382376266240000.0, 7771770303897600.0, \
                 1187353796428800.0, 129060195264000.0, 10559470521600.0, \
                 670442572800.0, 33522128640.0, 1323241920.0, 40840800.0, \
                 960960.0, 16380.0, 182.0, 1.0};
  
  if (*n <= 1) {
    *z = exp(*x);
  }
  else {
    if ((nx = (double *) malloc(*n * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((x2 = (double *) malloc(*n * *n * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((ipiv = (int *) malloc(*n * sizeof(int))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((tmp = (double *) malloc(*n * *n * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((pm = (double *) malloc(*n * *n * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((um = (double *) malloc(*n * *n * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((vm = (double *) malloc(*n * *n * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }

    for (i = 0; i < *n; i++) {
      nx[i] = 0.0;
      for (j = 0; j < *n; j++) {
        nx[i] += fabs(x[*n * i + j]);
      }
    }
    R_rsort(nx, *n);
    if (nx[*n - 1] <= 2.1) {
      for (l = 0; l < 4; l++) {
        if (nx[*n - 1] <= t[l]) break;
      }
      for (i = 0; i < *n; i++) {
        for (j = 0; j < *n; j++) {
          x2[*n * i + j] = 0.0;
          for (k = 0; k < *n; k++) {
            x2[*n * i + j] += x[*n * i + k] * x[*n * k + j];
          }
        }
      }
      memcpy(pm, x2, sizeof(double) * *n * *n);
      for (i = 0; i < *n; i++) {
        um[*n * i + i] = cm[4 + l];
        vm[*n * i + i] = cm[l];
        for (j = 0; j < *n; j++) {
          if (i != j) {
            um[*n * i + j] = 0.0;
            vm[*n * i + j] = 0.0;
          }
        }
      }
      for (k = 1; k <= l; k++) {
        for (i = 0; i < *n; i++) {
          for (j = 0; j < *n; j++) {
            um[*n * j + i] += cm[l + 4 * ((2 * k) + 1)] * pm[*n * j + i];
            vm[*n * j + i] += cm[l + 8 * k] * pm[*n * j + i];
          }
        }
        if (k == l) break;
        for (i = 0; i < *n; i++) {
          for (j = 0; j < *n; j++) {
            tmp[*n * i + j] = 0.0;
            for (h = 0; h < *n; h++) {
              tmp[*n * i + j] += pm[*n * i + h] * x2[*n * h + j];
            }
          }
        }
        memcpy(pm, tmp, sizeof(double) * *n * *n);
      }
      for (i = 0; i < *n; i++) {
        for (j = 0; j < *n; j++) {
          x2[*n * i + j] = 0.0;
          for (h = 0; h < *n; h++) {
            x2[*n * i + j] += x[*n * i + h] * um[*n * h + j];
          }
        }
      }
      for (i = 0; i < (*n * *n); i++) {
        um[i] = vm[i] - x2[i];
        vm[i] += x2[i];
      }
      F77_CALL(dgesv)(n, n, um, n, ipiv, vm, n, &info); 
      if (info > 0) {
        for (i = 0; i < *n * *n; i++) {
          vm[i] = R_NaN;
        }
      }
      memcpy(z, vm, sizeof(double) * *n * *n);
      free(vm);
    }
    else {
      free(vm);
      if ((x4 = (double *) malloc(*n * *n * sizeof(double))) == NULL) {
#if __VOPENMP
        #pragma omp critical
#endif
        error("%s", myMemErr);
      }
      if ((x6 = (double *) malloc(*n * *n * sizeof(double))) == NULL) {
#if __VOPENMP
        #pragma omp critical
#endif
        error("%s", myMemErr);
      }
      s = log(nx[*n - 1] / 5.4) / log(2.0);
      if (s > 0.0) {
            s = ceil(s);
            for (i = 0; i < *n * *n; i++) x[i] /= R_pow(2.0, s);
      }
      for (i = 0; i < *n; i++) {
        for (j = 0; j < *n; j++) {
          x2[*n * i + j] = 0.0;
          for (k = 0; k < *n; k++) {
            x2[*n * i + j] += x[*n * i + k] * x[*n * k + j];
          }
        }
      }
      for (i = 0; i < *n; i++) {
        for (j = 0; j < *n; j++) {
          x4[*n * i + j] = 0.0;
          for (k = 0; k < *n; k++) {
            x4[*n * i + j] += x2[*n * i + k] * x2[*n * k + j];
          }
        }
      }
      for (i = 0; i < *n; i++) {
        for (j = 0; j < *n; j++) {
          x6[*n * i + j] = 0.0;
          for (k = 0; k < *n; k++) {
            x6[*n * i + j] += x2[*n * i + k] * x4[*n * k + j];
          }
        }
      }
      for (i = 0; i < *n * *n; i++) {
        tmp[i] = cv[13] * x6[i] + cv[11] * x4[i] + cv[9] * x2[i];
      }
      for (i = 0; i < *n; i++) {
        for (j = 0; j < *n; j++) {
          pm[*n * i + j] = 0.0;
          for (k = 0; k < *n; k++) {
            pm[*n * i + j] += x6[*n * i + k] * tmp[*n * k + j];
          }
        }
      }
      for (i = 0; i < *n * *n; i++) {
        tmp[i] = pm[i] + cv[7] * x6[i] + cv[5] * x4[i] + cv[3] * x2[i];
      }
      for(i = 0; i < *n; i++) tmp[*n * i + i] += cv[1];
      for (i = 0; i < *n; i++) {
        for (j = 0; j < *n; j++) {
          um[*n * i + j] = 0.0;
          for (k = 0; k < *n; k++) {
            um[*n * i + j] += x[*n * i + k] * tmp[*n * k + j];
          }
        }
      }
      for (i = 0; i < *n * *n; i++) {
        tmp[i] = cv[12] * x6[i] + cv[10] * x4[i] + cv[8] * x2[i];
      }
      for (i = 0; i < *n; i++) {
        for (j = 0; j < *n; j++) {
          pm[*n * i + j] = 0.0;
          for (k = 0; k < *n; k++) {
            pm[*n * i + j] += x6[*n * i + k] * tmp[*n * k + j];
          }
        }
      }
      for (i = 0; i < *n * *n; i++) {
        tmp[i] = pm[i] + cv[6] * x6[i] + cv[4] * x4[i] + cv[2] * x2[i];
      }
      for(i = 0; i < *n; i++) tmp[*n * i + i] += cv[0];
      for (i = 0; i < (*n * *n); i++) {
        pm[i] = tmp[i] - um[i];
        tmp[i] += um[i];
      }
      F77_CALL(dgesv)(n, n, pm, n, ipiv, tmp, n, &info);
      if (info > 0) {
        for (i = 0; i < *n * *n; i++) {
          tmp[i] = R_NaN;
        }
      }
      memcpy(z, tmp, sizeof(double) * *n * *n);
      if (s > 0) {
        for (h = 0; h < s; h++) {
          for (i = 0; i < *n; i++) {
            for (j = 0; j < *n; j++) {
              x[*n * i + j] = 0.0;
              for (k = 0; k < *n; k++) {
                x[*n * i + j] += z[*n * i + k] * z[*n * k + j];
              }
            }
          }
          memcpy(z, x, sizeof(double) * *n * *n);
        }
      }
      free(x4);
      free(x6);
    }
    free(nx);
    free(x2);
    free(ipiv);
    free(tmp);
    free(pm);
    free(um);
  }
}

void predTPFIT(double *coefficients, double *prop, double *lags, int *mydim, double *mypred) {
  /* Compute transition probability matrices for a 1D model
       *coefficients - matrix of coefficients
               *prop - vector of proportions
               *lags - vector of 1D lags
              *mydim - dimension of *mypred
             *mypred - transition probability matrices */

  int i, j;
  double *mycoef, *dgcoef, *tmpdcf;
  
  if ((mycoef = (double *) malloc(mydim[0] * mydim[1] * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  if ((dgcoef = (double *) malloc(mydim[0] * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  if ((tmpdcf = (double *) malloc(mydim[0] * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
    
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j) schedule(static, 1)
#endif
  for (i = 0; i < mydim[0]; i++) {
    for (j = 0; j < mydim[0]; j++) {
      mycoef[i * mydim[0] + j] = (prop[i] / prop[j]) * coefficients[j * mydim[0] + i];
    }
  }
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j) schedule(static, 1)
#endif
  for (i = 0; i < mydim[0]; i++) {
    dgcoef[i] = mycoef[i * mydim[0] + i];
    mycoef[i * mydim[0] + i] = 0.0;
    tmpdcf[i] = mycoef[i];
    for (j = 1; j < mydim[0]; j++) tmpdcf[i] += mycoef[j * mydim[0] + i];
    tmpdcf[i] = (-dgcoef[i]) / tmpdcf[i];
  }
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j) schedule(static, 1)
#endif
  for (i = 0; i < mydim[0]; i++) {
    for (j = 0; j < mydim[0]; j++) {
      mycoef[i * mydim[0] + j] *= tmpdcf[j];
    }
  }
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i) schedule(static, 1)
#endif
  for (i = 0; i < mydim[0]; i++) mycoef[i * mydim[0] + i] = dgcoef[i];

  free(tmpdcf);
  free(dgcoef);

#if __VOPENMP
  #pragma omp parallel shared(mydim, myMemErr)
  {
#endif
    if ((h = (double *) malloc(mydim[0] * mydim[1] * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
#if __VOPENMP
  }

  #pragma omp parallel for default(shared) private(i, j) schedule(static, 1)
#endif
  for (i = 0; i < mydim[2]; i++) {
    if (lags[i] < 0.0) {
      memcpy(h, mycoef, sizeof(double) * mydim[0] * mydim[1]);
    }
    else {
      memcpy(h, coefficients, sizeof(double) * mydim[0] * mydim[1]);
    }
    for (j = 0; j < mydim[0] * mydim[1]; j++) h[j] *= fabs(lags[i]);
    expmat(h, mydim, &mypred[i * mydim[0] * mydim[1]]);
    nrmPrMat(&mypred[i * mydim[0] * mydim[1]], mydim);
  }

#if __VOPENMP
  #pragma omp parallel 
  {
#endif
    free(h);
#if __VOPENMP
  }
#endif
  free(mycoef);
}

void nrmPrMat(double *x, int *n) {
  /* Normalization of a probability matrix
       *x - probability matrix
       *n - number of rows (or columns) */

  int i = 0, j;
  double *sm;

  if ((sm = (double *) malloc(*n * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }

  for (; i < *n; i++) {
    sm[i] = x[i];
    for (j = 1; j < *n; j++) {
       sm[i] += x[*n * j + i];
    }
  }

  for (i = 0; i < *n; i++) {
    for (j = 0; j < *n; j++) {
       x[*n * j + i] /= sm[i];
    }
  }

  free(sm);
}

void revCoef(double *coefficients, double *prop, int *nk, double *mycoef) {
  /* Compute coefficients for the opposite direction
       *coefficients - matrix of transition rates
               *prop - vector of proportions
                 *nk - number of categories
             *mycoef - resulting matrix coefficients */

  int i, j;
  double *dgcoef, *tmpdcf;

  if ((dgcoef = (double *) malloc(*nk * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  if ((tmpdcf = (double *) malloc(*nk * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }

    
  for (i = 0; i < *nk; i++) {
    for (j = 0; j < *nk; j++) {
      mycoef[i * *nk + j] = (prop[i] / prop[j]) * coefficients[*nk * j + i];
    }
  }
  for (i = 0; i < *nk; i++) {
    dgcoef[i] = mycoef[(*nk + 1) * i];
    mycoef[(*nk + 1) * i] = 0.0;
    tmpdcf[i] = mycoef[i];
    for (j = 1; j < *nk; j++) tmpdcf[i] += mycoef[*nk * j + i];
    tmpdcf[i] = (-dgcoef[i]) / tmpdcf[i];
  }
  for (i = 0; i < *nk; i++) {
    for (j = 0; j < *nk; j++) {
      mycoef[*nk * i + j] *= tmpdcf[j];
    }
  }
  for (i = 0; i < *nk; i++) mycoef[(*nk + 1) * i] = dgcoef[i];

  free(tmpdcf);
  free(dgcoef);
}

void predVET(double *coefficients, double *revcoef, int *nk, int *nc, double *lag, double *pred) {
  /* Compute transition probability matrices for a given multidimensional model
       *coefficients - matrices of coefficients
            *revcoef - matrices of coefficients for opposite direction
                 *nk - number of categories
                 *nc - sample space dimension
                *lag - lag vector
               *pred - transition probability matrix */

  int i, j;
  double dst, summe;
  double *MRMat, *RMat, *tmplag;

  dst = R_pow(lag[0], 2.0);
  for (i = 1; i < *nc; i++) dst += R_pow(lag[i], 2.0);
  dst = R_pow(dst, 0.5);
  if (dst == 0.0) {
    for (i = 0; i < *nk; i++) {
      for (j = 0; j < *nk; j++) {
        pred[*nk * j + i] = 0.0;
      }
      ++pred[(*nk + 1) * i];
    }
  }
  else {
    if ((MRMat = (double *) malloc(*nc * *nk * *nk * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((RMat = (double *) malloc(*nk * *nk * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((tmplag = (double *) malloc(*nc * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }

    for (i = 0; i < *nc; i++) {
      tmplag[i] = lag[i] / dst;
      if (ISNAN(tmplag[i])) tmplag[i] = 0.0;
      if (lag[i] < 0.0) {
        memcpy(&MRMat[*nk * *nk * i], &revcoef[*nk * *nk * i], *nk * *nk * sizeof(double));
      }
      else {
        memcpy(&MRMat[*nk * *nk * i], &coefficients[*nk * *nk * i], *nk * *nk * sizeof(double));
      }
    }
    ellinter(nc, nk, tmplag, MRMat, RMat);

    free(MRMat);
    free(tmplag);

    for (i = 0; i < *nk; i++) {
      summe = RMat[i];
      for (j = 1; j < *nk; j++) {
        summe += RMat[*nk * j + i];
      }
      RMat[(*nk + 1) * i] -= summe;
      RMat[i] *= dst;
      for (j = 1; j < *nk; j++) {
        RMat[*nk * j + i] *= dst;
      }
    }
    expmat(RMat, nk, pred);
    nrmPrMat(pred, nk);

    free(RMat);  
  }
}

void predMULTI(double *coefficients, double *prop, double *lags, int *nk, int *nc, int *nr, double *mypred) {
  /* Compute transition probability matrices for a given multidimensional model by HPC applied to each lag vector
       *coefficients - matrices of coefficients
               *prop - vector of proportions
               *lags - matrix whose columns are lags
                 *nk - number of categories
                 *nc - sample space dimension
                 *nr - number of lags
             *mypred - transition probability matrices */

  int i;
  double *mycoef;

  if ((mycoef = (double *) malloc(*nk * *nk * *nc * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }


#if __VOPENMP
  #pragma omp parallel for default(shared) private(i) schedule(static, 1)
#endif
  for (i = 0; i < *nc; i++) {
    revCoef(&coefficients[*nk * *nk * i], prop, nk, &mycoef[*nk * *nk * i]);
  }
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i) schedule(static, 1)
#endif
  for (i = 0; i < *nr; i++) {
    predVET(coefficients, mycoef, nk, nc, &lags[*nc * i], &mypred[*nk * *nk * i]);
  }

  free(mycoef);
}

void predPSEUDOVET(double *coefficients, double *revcoef, int *nk, int *nc, int *whichd, double *lag, double *pred) {
  /* Compute transition probability matrices for a given multidimensional model
       *coefficients - matrices of coefficients
            *revcoef - matrices of coefficients for opposite direction
                 *nk - number of categories
                 *nc - sample space dimension
                *lag - lag vector
               *pred - transition probability matrix */

  int i, j;
  double dst, summe;
  double *RMat;
  
  if (ISNAN(*coefficients)) {
    memcpy(pred, coefficients, *nk * *nk * sizeof(double));
  }
  else {
    dst = R_pow(lag[0], 2.0);
    for (i = 1; i < *nc; i++) dst += R_pow(lag[i], 2.0);
    dst = R_pow(dst, 0.5);
    if (dst == 0.0) {
      for (i = 0; i < *nk; i++) {
        for (j = 0; j < *nk; j++) {
          pred[*nk * j + i] = 0.0;
        }
        ++pred[(*nk + 1) * i];
      }
    }
    else {
      if ((RMat = (double *) malloc(*nk * *nk * sizeof(double))) == NULL) {
#if __VOPENMP
        #pragma omp critical
#endif
        error("%s", myMemErr);
      }

      if (lag[*whichd - 1] < 0.0) {
        memcpy(RMat, revcoef, *nk * *nk * sizeof(double));
      }
      else {
        memcpy(RMat, coefficients, *nk * *nk * sizeof(double));
      }

      for (i = 0; i < *nk; i++) {
        summe = RMat[i];
        for (j = 1; j < *nk; j++) {
          summe += RMat[*nk * j + i];
        }
        RMat[(*nk + 1) * i] -= summe;
        RMat[i] *= dst;
        for (j = 1; j < *nk; j++) {
          RMat[*nk * j + i] *= dst;
        }
      }
      expmat(RMat, nk, pred);
      nrmPrMat(pred, nk);

      free(RMat);
    }
  }
}

void predPSEUDO(double *coefs, double *prop, double *lags, int *nk, int *nc, int *nr, int *nmat, int *wsd, int *whichd, double *mypred) {
  /* Compute transition probability matrices for a given multidimensional model by HPC applied to each lag vector
              *coefs - matrices of coefficients
               *prop - vector of proportions
               *lags - matrix whose columns are lags
                 *nk - number of categories
                 *nc - sample space dimension
                 *nr - number of lags
               *nmat - number of coefficients matrices 
                *wsd - same direction non duplicated
             *whichd - main dimension to check opposite direction
             *mypred - transition probability matrices */

  int i;
  double *mycoef;

  if ((mycoef = (double *) malloc(*nk * *nk * *nmat * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }

#if __VOPENMP
  #pragma omp parallel for default(shared) private(i) schedule(static, 1)
#endif
  for (i = 0; i < *nmat; i++) {
    if (ISNAN(coefs[*nk * *nk * i])) {
      memcpy(mycoef, coefs, *nk * *nk * sizeof(double));
    }
    else {
      revCoef(&coefs[*nk * *nk * i], prop, nk, &mycoef[*nk * *nk * i]);
    }
  }
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i) schedule(static, 1)
#endif
  for (i = 0; i < *nr; i++) {
    predPSEUDOVET(&coefs[*nk * *nk * (wsd[i] - 1)], &mycoef[*nk * *nk * (wsd[i] - 1)], nk, nc, whichd, &lags[*nc * i], &mypred[*nk * *nk * i]);
  }

  free(mycoef);

}

void rotaH(int *nc, double *matdir, double *vet) {
  int i, j;
  double *lag;

  if ((lag = (double *) malloc(*nc * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }

  for (i = 0; i < *nc; i++) {
    lag[i] = vet[0] * matdir[*nc * i];
    for (j = 1; j < *nc; j++) {
      lag[i] += vet[j] * matdir[*nc * i + j];
    }
  }
  memcpy(vet, lag, *nc * sizeof(double));
  free(lag);
}

void rotaxes(int *nc, double *ang, double *res) {
  int i, j;
  double *rotmat;
  
  rotmat = (double *) malloc(*nc * *nc * sizeof(double));
  
  res[0] = cos(ang[0]);
  res[1] = sin(ang[0]);
  res[*nc] = - sin(ang[0]);
  res[*nc + 1] = cos(ang[0]);
  
  for (i = 1; i < (*nc - 1); i++) {
    if (ang[i] != 0.0) {
      memcpy(rotmat, res, *nc * *nc * sizeof(double));
      for (j = 0; j <= i; j++) {
        res[j] = rotmat[j] * cos(ang[i]);
        res[*nc * (i + 1) + j] = rotmat[j] * (- sin(ang[i]));
      }
      res[i + 1] = sin(ang[i]);
      res[*nc * (i + 1) + (i + 1)] = cos(ang[i]);
    }
  }

  free(rotmat);
}

void fastrss(int *n, double *mypred, double *Tmat, double *rss) {
  /* Fast computation of the residual sum of squares
            n - length of mypred and Tmat
       mypred - predicted probabilities
         Tmat - empirical probabilities
          rss - residual sum of squares */

  int i;
  double *vet;

  if ((vet = (double *) malloc(*n * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }

#if __VOPENMP
  #pragma omp parallel for default(shared) private(i) schedule(static, 1)
#endif
  for (i = 0; i < *n; i++) {
    vet[i] = mypred[i] - Tmat[i];
    if (ISNA(vet[i]) || ISNAN(vet[i])) {
      vet[i] = 0.0;
    }
    else {
      vet[i] *= vet[i];
    }
  }
  *rss = 0.0;
  for (i = 0; i < *n; i++) *rss += vet[i];
  free(vet);
}

SEXP bclm(SEXP q, SEXP eps, SEXP res, SEXP echo, SEXP expr, SEXP Rnv) {
  /* Perform the bound-constrained Lagrangian minimization
          q - constant controlling the growth of rho
        eps - convergence tolerance
        res - initial point
       echo - boolean value
       expr - function to optimize
        Rnv - R enviroment */

  SEXP resOLD, r, ans;
  int i = 1, j, n;
  double mxdst, tmp;
  double *dres, *dold, *dr, *dans;
  
  PROTECT(q = coerceVector(q, REALSXP));
  PROTECT(eps = coerceVector(eps, REALSXP));
  PROTECT(echo = coerceVector(echo, LGLSXP));
  PROTECT(res = coerceVector(res, REALSXP));
  n = length(res);
  PROTECT(resOLD = allocVector(REALSXP, n));
  PROTECT(r = allocVector(REALSXP, 1));
  dres = REAL(res);
  dold = REAL(resOLD);
  dr = REAL(r);
  dr[0] = 0.0;
  
  for (;;) {
#if __VOPENMP
    #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
    for (j = 0; j < n; j++) {
      dold[j] = dres[j];
    }
    if (LOGICAL(echo)[0]) Rprintf("Iteration %d\n", i);
    defineVar(install("rho"), r, Rnv);
    defineVar(install("res"), res, Rnv);
    PROTECT(ans = coerceVector(eval(expr, Rnv), REALSXP));
    dans = REAL(ans);
    mxdst = 0.0;
    for (j = 0; j < n; j++) {
      tmp = fabs(dold[j] - dans[j]);
      if (mxdst < tmp) mxdst = tmp;
    }
#if __VOPENMP
    #pragma omp parallel for default(shared) private(j) schedule(static, 1)
#endif
    for (j = 0; j < n; j++) {
      dres[j] = dans[j];
    }
    UNPROTECT(1);
    if (mxdst < REAL(eps)[0]) break;
    ++i;
    if (dr[0] <= 0.0) dr[0] = 0.1;
    dr[0] *= REAL(q)[0];
  }
  UNPROTECT(6);
  return(res);
}

void knear(int *nc, int *nr, double *coords, int *nrs, double *grid, int *knn, int *indices) {
  /* Finding the k-nearest neighbours
           *nc - number of columns
           *nr - number of observation rows
       *coords - matrix of coordinates
          *nrs - number of simulation rows
         *grid - simulation grid
          *knn - number of neighbours
      *indices - row indices of the nearest observations */

  int i, j, k;
  double dst;

#if __VOPENMP
  #pragma omp parallel shared(knn, myMemErr)
  {
#endif
    if ((p = (double *) malloc(*knn * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((wo = (int *) malloc(*knn * sizeof(int))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
#if __VOPENMP
  }

  #pragma omp parallel for default(shared) private(i, j, k, dst) schedule(static, 1)
#endif
  for (i = 0; i < *nrs; i++) {
    for (j = 0; j < *knn; j++) {
      dst = pow(coords[j] - grid[i], 2.0);
      for (k = 1; k < *nc; k++) {
        dst += pow(coords[*nr * k + j] - grid[*nrs * k + i], 2.0);
      }
      // dst = sqrt(dst);
      p[j] = dst;
      wo[j] = j;
    }
    rsort_with_index(p, wo, *knn);
    for (j = *knn; j < *nr; j++) {
      dst = pow(coords[j] - grid[i], 2.0);
      for (k = 1; k < *nc; k++) {
        dst += pow(coords[*nr * k + j] - grid[*nrs * k + i], 2.0);
      }
      // dst = sqrt(dst);
      if (dst < p[*knn - 1]) {
        p[*knn - 1] = dst;
        wo[*knn - 1] = j;
        rsort_with_index(p, wo, *knn);
      }
    }
    R_isort(wo, *knn);
    indices[*knn * i] = wo[0];
    for (k = 1; k < *knn; k++) {
      indices[*knn * i + k] = wo[k];
    }
  }

#if __VOPENMP
  #pragma omp parallel
  {
#endif
    free(p);
    free(wo);
#if __VOPENMP
  }
#endif
}

void getIKPrbs(int *ordinary, int *indices, int *groups, int *knn, int *nc, int *nr, int *nrs, int *data, double *coords, double *grid, int *nk, double *coef, double *prop, double *probs) {
  /* Computing simulation probabilities
     *ordinary - boolean to distinguish ordinary or simple kriging
      *indices - row indices of the nearest observations
       *groups - vector with indices of same neighbours groups
          *knn - number of neighbours considered
           *nc - number of columns
           *nr - number of observation rows
          *nrs - number of simulation rows
         *data - vector of observed categories
       *coords - matrix of coordinates
         *grid - simulation grid
           *nk - number of categories
        *coefs - matrices of coefficients
         *prop - vector of proportions
        *probs - transition probability matrices */

  int g, i = 0, j, j1, j2, k, bg = 0, info = 0;
  int knn2 = *knn * *knn;
  int nk2 = *nk * *nk;
  int nt = *nk * *knn;
  double stds;
  double *Ttilde, *invTtilde, *revcoef, *Wtilde, *Otilde;
  
  k = *nk * (knn2 + *ordinary * (*knn * 2 + 1));
  if ((Ttilde = (double *) malloc(k * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  } //transition matrix for neighbour points
  if ((invTtilde = (double *) malloc(k * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  } //inverse transition matrix for neighbour points
  if ((revcoef = (double *) malloc(nk2 * *nc * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  } //matrices of coefficients for reverible MC
  k = nt + *ordinary * *nk;
  if ((Wtilde = (double *) malloc(k * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
    error("%s", myMemErr);
#endif
  } //vectors of kriging weights
  if ((Otilde = (double *) malloc(k * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  } //vectors for simulation point transition probabilities

#if __VOPENMP
  #pragma omp parallel shared(nc, ordinary, knn, myMemErr)
  {
#endif
    if ((TtLag = (double *) malloc(*nc * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((tmpMat = (double *) malloc(nk2 * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((pv = (int *) malloc((*ordinary + *knn) * sizeof(int))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
#if __VOPENMP
  }

  #pragma omp parallel for default(shared) private(k) schedule(static, 1)
#endif
  for (k = 0; k < *nc; k++) {
    revCoef(&coef[nk2 * k], prop, nk, &revcoef[nk2 * k]);
  }
  //set constraint values in the vectors
  if (*ordinary) {
#if __VOPENMP
    #pragma omp parallel for default(shared) private(k) schedule(static, 1)
#endif
    for (k = 0; k < *nk; k++) {
      Otilde[(*knn + 1) * k + *knn] = 1.0;
    }
  }

  g = groups[i];
  while (i < *nrs) {
    //computation for one group
#if __VOPENMP
    #pragma omp parallel for default(shared) private(j1, j2, k, j) schedule(static, 1)
#endif
    for (j1 = 0; j1 < *knn; j1++) {
      for (j2 = 0; j2 < *knn; j2++) {
        for (k = 0; k < *nc; k++) {
          TtLag[k] = coords[*nr * k + indices[*knn * i + j2]] - coords[*nr * k + indices[*knn * i + j1]];
        }
        //compute transition matrix
        predVET(coef, revcoef, nk, nc, TtLag, tmpMat);
        //set probability of the sample configuration for each category
        for (j = 0; j < *nk; j++) {
          k = (knn2 + *ordinary * (*knn * 2 + 1)) * j + (*knn + *ordinary) * j2 + j1;
          Ttilde[k] = tmpMat[(*nk + 1) * j] - (1.0 - (double) *ordinary) * prop[j];
        }
      }
    }
    if (*ordinary) {
#if __VOPENMP
      #pragma omp parallel for default(shared) private(j, k) schedule(static, 1)
#endif
      for (j = 0; j < *nk; j++) {
        for (k = 0; k < *knn; k++) {
          Ttilde[(knn2 + *knn * 2 + 1) * j + (*knn + 1) * k + *knn] = 1.0;
          Ttilde[(knn2 + *knn * 2 + 1) * j + knn2 + *knn + k] = 1.0;
        }
        Ttilde[(knn2 + *knn * 2 + 1) * (j + 1) - 1] = 0.0;
      }
      *knn += 1;
      knn2 = *knn * *knn;
    }
    //set the indicator matries
#if __VOPENMP
    #pragma omp parallel for default(shared) private(k) schedule(static, 1)
#endif
    for (k = 0; k < (*nk * knn2); k++) {
      invTtilde[k] = 0.0;
    }
#if __VOPENMP
    #pragma omp parallel for default(shared) private(k, j) schedule(static, 1)
#endif
    for (k = 0; k < *knn; k++) {
      for (j = 0; j < *nk; j++) {
        invTtilde[knn2 * j + k * (1 + *knn)] = 1.0;
      }
    }
   //invert the probability matrix of the sample configuration
#if __VOPENMP
    #pragma omp parallel for default(shared) private(j, info) schedule(static, 1)
#endif
    for (j = 0; j < *nk; j++) {
      F77_CALL(dgesv)(knn, knn, &Ttilde[knn2 * j], knn, pv, &invTtilde[knn2 * j], knn, &info);
    }
    if (*ordinary) {
      *knn -= 1;
      knn2 = *knn * *knn;
    }

    for (; bg < *nrs; bg++) {
      if (g != groups[bg]) break;
    }

    //cycle for points within the group
    for (; i < bg; i++) {
#if __VOPENMP
      #pragma omp parallel for default(shared) private(j1, k) schedule(static, 1)
#endif
      for (j1 = 0; j1 < *knn; j1++) {
        for (k = 0; k < *nc; k++) {
          TtLag[k] = grid[*nrs * k + i] - coords[*nr * k + indices[*knn * i + j1]];
        }
        predVET(coef, revcoef, nk, nc, TtLag, tmpMat);
        for (k = 0; k < *nk; k++) {
          Otilde[(*knn + *ordinary) * k + j1] = tmpMat[(*nk + 1) * k] - (1.0 - (double) *ordinary) * prop[k];
        }
      }
      j = 1;
      if (*ordinary) {
        *knn += 1;
        knn2 = *knn * *knn;
      }
      for (k = 0; k < *nk; k++) {
        fastMatProd(knn, knn, &invTtilde[knn2 * k], &j, &Otilde[*knn * k], &Wtilde[*knn * k]);
      }
      if (*ordinary) {
        *knn -= 1;
        knn2 = *knn * *knn;
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j1, j2) schedule(static, 1)
#endif
        for (j1 = 0; j1 < *knn; j1++) {
          for (j2 = 0; j2 < *nk; j2++) {
            if (j2 + 1 != data[indices[*knn * i + j1]]) {
              Wtilde[(*knn + 1) * j2 + j1] = 0.0;
            }
          }
        }
      }
      else {
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j1, j2) schedule(static, 1)
#endif
        for (j1 = 0; j1 < *knn; j1++) {
          for (j2 = 0; j2 < *nk; j2++) {
            if (j2 + 1 != data[indices[*knn * i + j1]]) {
              Wtilde[*knn * j2 + j1] *= (-prop[j2]);
            }
            else {
              Wtilde[*knn * j2 + j1] *= (1.0 - prop[j2]);
            }
          }
        }
      }

#if __VOPENMP
      #pragma omp parallel for default(shared) private(j1, j2) schedule(static, 1)
#endif
      for (j2 = 0; j2 < *nk; j2++) {
        probs[*nrs * j2 + i] = prop[j2] * (1.0 - (double) *ordinary) + Wtilde[(*knn + *ordinary) * j2];
        for (j1 = 1; j1 < *knn; j1++) {
          probs[*nrs * j2 + i] += Wtilde[(*knn + *ordinary) * j2 + j1];
        }
        if (probs[*nrs * j2 + i] > 1.0) probs[*nrs * j2 + i] = 1.0;
        if (probs[*nrs * j2 + i] < 0.0) probs[*nrs * j2 + i] = 0.0;
      }
      stds = 0.0; 
      for (j2 = 0; j2 < *nk; j2++) {
        stds += probs[*nrs * j2 + i];
      }
      if (stds == 0.0) {
        for (j2 = 0; j2 < *nk; j2++) {
          probs[*nrs * j2 + i] = prop[j2] * (1.0 - (double) *ordinary) + Wtilde[(*knn + *ordinary) * j2];
          for (j1 = 1; j1 < *knn; j1++) {
            probs[*nrs * j2 + i] += Wtilde[(*knn + *ordinary) * j2 + j1];
          }
          if (probs[*nrs * j2 + i] > 1.0) probs[*nrs * j2 + i] = 1.0;
          if (probs[*nrs * j2 + i] < stds) stds = probs[*nrs * j2 + i];
        }
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j2) schedule(static, 1)
#endif
        for (j2 = 0; j2 < *nk; j2++) {
          probs[*nrs * j2 + i] -= stds;
        }
        stds = probs[i];
        for (j2 = 1; j2 < *nk; j2++) {
          stds += probs[*nrs * j2 + i];
        }
      }
      if (stds == 0.0) {
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j2) schedule(static, 1)
#endif
        for (j2 = 0; j2 < *nk; j2++) {
          probs[*nrs * j2 + i] = prop[j2];
        }
      }
      else {
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j2) schedule(static, 1)
#endif
        for (j2 = 0; j2 < *nk; j2++) {
          probs[*nrs * j2 + i] /= stds;
        }
      }
    }
    g++;
  }

  free(Ttilde);
  free(invTtilde);
  free(revcoef);
  free(Wtilde);
  free(Otilde);

#if __VOPENMP
  #pragma omp parallel
  {
#endif
    free(TtLag);
    free(tmpMat);
    free(pv);
#if __VOPENMP
  }
#endif
}

void getCKPrbs(int *ordinary, int *indices, int *groups, int *knn, int *nc, int *nr, int *nrs, int *data, double *coords, double *grid, int *nk, double *coef, double *prop, double *probs) {
  /* Computing simulation probabilities
     *ordinary - boolean to distinguish ordinary or simple kriging
      *indices - row indices of the nearest observations
       *groups - vector with indices of same neighbours groups
          *knn - number of neighbours considered
           *nc - number of columns
           *nr - number of observation rows
          *nrs - number of simulation rows
         *data - vector of observed categories
       *coords - matrix of coordinates
         *grid - simulation grid
           *nk - number of categories
        *coefs - matrices of coefficients
         *prop - vector of proportions
        *probs - transition probability matrices */

  int g, i = 0, j, j1, j2, k, l, bg = 0, info = 0;
  int knn2 = *knn * *knn;
  int nk2 = *nk * *nk;
  int nt = *nk * *knn;
  double stds;
  double *Ttilde, *invTtilde, *revcoef, *Wtilde, *Otilde;
  
  k = nk2 * (knn2 + *ordinary * (*knn * 2 + 1));
  if ((Ttilde = (double *) malloc(k * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  } //transition matrix for neighbour points
  if ((invTtilde = (double *) malloc(k * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  } //inverse transition matrix for neighbour points
  if ((revcoef = (double *) malloc(nk2 * *nc * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  } //matrices of coefficients for reverible MC
  k = *nk * nt + *ordinary * nk2;
  if ((Wtilde = (double *) malloc(k * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  } //vectors of kriging weights
  if ((Otilde = (double *) malloc(k * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  } //vectors for simulation point transition probabilities

#if __VOPENMP
  #pragma omp parallel shared(nc, ordinary, knn, myMemErr)
  {
#endif
    if ((TtLag = (double *) malloc(*nc * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((tmpMat = (double *) malloc(nk2 * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((pv = (int *) malloc((*ordinary + *knn) * sizeof(int))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
#if __VOPENMP
  }

  #pragma omp parallel for default(shared) private(k) schedule(static, 1)
#endif
  for (k = 0; k < *nc; k++) {
    revCoef(&coef[nk2 * k], prop, nk, &revcoef[nk2 * k]);
  }
  //set constraint values in the vectors
  if (*ordinary) {
#if __VOPENMP
    #pragma omp parallel for default(shared) private(k) schedule(static, 1)
#endif
    for (k = 0; k < nk2; k++) {
      Otilde[(*knn + 1) * k + *knn] = 1.0;
    }
  }

  g = groups[i];
  while (i < *nrs) {
    //computation for one group
#if __VOPENMP
    #pragma omp parallel for default(shared) private(j1, j2, k, j, l) schedule(static, 1)
#endif
    for (j1 = 0; j1 < *knn; j1++) {
      for (j2 = 0; j2 < *knn; j2++) {
        for (k = 0; k < *nc; k++) {
          TtLag[k] = coords[*nr * k + indices[*knn * i + j2]] - coords[*nr * k + indices[*knn * i + j1]];
        }
        //compute transition matrix
        predVET(coef, revcoef, nk, nc, TtLag, tmpMat);
        //set probability of the sample configuration for each category couple
        for (j = 0; j < *nk; j++) {
          for (l = 0; l < *nk; l++) {
            if (j == l) {
              k = (knn2 + *ordinary * (*knn * 2 + 1)) * (1 + *nk) * j + (*knn + *ordinary) * j2 + j1;
              Ttilde[k] = tmpMat[(*nk + 1) * j] - (1.0 - (double) *ordinary) * prop[j];
            }
            else {
              k = (knn2 + *ordinary * (*knn * 2 + 1)) * (l + *nk * j) + *knn * j2 + j1;
              Ttilde[k] = tmpMat[*nk * j + l] - (1.0 - (double) *ordinary) * prop[j];
            }
          }
        }
      }
    }
    if (*ordinary) {
#if __VOPENMP
      #pragma omp parallel for default(shared) private(j, k) schedule(static, 1)
#endif
      for (j = 0; j < *nk; j++) {
        for (k = 0; k < *knn; k++) {
          Ttilde[(knn2 + *knn * 2 + 1) * (1 + *nk) * j + (*knn + 1) * k + *knn] = 1.0;
          Ttilde[(knn2 + *knn * 2 + 1) * (1 + *nk) * j + knn2 + *knn + k] = 1.0;
        }
        Ttilde[(knn2 + *knn * 2 + 1) * (1 + *nk) * j + knn2 + *knn + k] = 0.0;
      }
      *knn += 1;
      knn2 = *knn * *knn;
    }
    //set the indicator matrices
#if __VOPENMP
    #pragma omp parallel for default(shared) private(k) schedule(static, 1)
#endif
    for (k = 0; k < (nk2 * knn2); k++) {
      invTtilde[k] = 0.0;
    }
    if (*ordinary) {
      *knn -= 1;
      knn2 = *knn * *knn;
    }
#if __VOPENMP
    #pragma omp parallel for default(shared) private(k, j, l) schedule(static, 1)
#endif
    for (k = 0; k < *knn; k++) {
      for (j = 0; j < *nk; j++) {
        for (l = 0; l < *nk; l++) {
          if (j == l) {
            invTtilde[(knn2 + *ordinary * (*knn * 2 + 1)) * (1 + *nk) * j + (*knn + *ordinary + 1) * k] = 1.0;
          }
          else {
            invTtilde[(knn2 + *ordinary * (*knn * 2 + 1)) * (l + *nk * j) + (*knn + 1) * k] = 1.0;
          }
        }
      }
    }
    //invert the probability matrix of the sample configuration
    j2 = knn2 + *ordinary * (*knn * 2 + 1);
#if __VOPENMP
    #pragma omp parallel for default(shared) private(j, l, j1, info) schedule(static, 1)
#endif
    for (j = 0; j < *nk; j++) {
      for (l = 0; l < *nk; l++) {
        if (j == l) {
          j1 = *knn + *ordinary;
          F77_CALL(dgesv)(&j1, &j1, &Ttilde[j2 * (1 + *nk) * j], &j1, pv, &invTtilde[j2 * (1 + *nk) * j], &j1, &info);
        }
        else {
          j1 = *knn;
          F77_CALL(dgesv)(&j1, &j1, &Ttilde[j2 * (l + *nk * j)], &j1, pv, &invTtilde[j2 * (l + *nk * j)], &j1, &info);
        }
      }
    }

    for (; bg < *nrs; bg++) {
      if (g != groups[bg]) break;
    }

    //cycle for points within the group
    for (; i < bg; i++) {
#if __VOPENMP
      #pragma omp parallel for default(shared) private(j1, k, l) schedule(static, 1)
#endif
      for (j1 = 0; j1 < *knn; j1++) {
        for (k = 0; k < *nc; k++) {
          TtLag[k] = grid[*nrs * k + i] - coords[*nr * k + indices[*knn * i + j1]];
        }
        predVET(coef, revcoef, nk, nc, TtLag, tmpMat);
        for (k = 0; k < *nk; k++) {
          for (l = 0; l < *nk; l++) {
            if (l == k) {
              Otilde[(*knn + *ordinary) * (*nk + 1) * k + j1] = tmpMat[(*nk + 1) * k] - (1.0 - (double) *ordinary) * prop[k];
            } 
            else {
              Otilde[(*knn + *ordinary) * (*nk * k + l) + j1] = tmpMat[*nk * k + l] - (1.0 - (double) *ordinary) * prop[k];
            }
          }
        }
      }
      j = 1;
      for (k = 0; k < *nk; k++) {
        for (l = 0; l < *nk; l++) {
          if (l == k) {
            *knn += *ordinary;
            knn2 = *knn * *knn;
            fastMatProd(knn, knn, &invTtilde[knn2 * (*nk + 1) * k], &j, &Otilde[*knn * (*nk + 1) * k], &Wtilde[*knn * (*nk + 1) * k]);
            *knn -= *ordinary;
            knn2 = *knn * *knn;
          }
          else {
            fastMatProd(knn, knn, &invTtilde[(knn2 + *ordinary * (*knn * 2 + 1)) * (*nk * k + l)], &j, &Otilde[(*knn + *ordinary) * (*nk * k + l)], &Wtilde[(*knn + *ordinary) * (*nk * k + l)]);
          }
        }
      }
      if (*ordinary) {
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j1, j2, l) schedule(static, 1)
#endif
        for (j1 = 0; j1 < *knn; j1++) {
          for (j2 = 0; j2 < *nk; j2++) {
            for (l = 0; l < *nk; l++) {
              if (l + 1 != data[indices[*knn * i + j1]]) {
                Wtilde[(*knn + 1) * (*nk * j2 + l) + j1] = 0.0;
              }
            }
          }
        }
      }
      else {
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j1, j2, l) schedule(static, 1)
#endif
        for (j1 = 0; j1 < *knn; j1++) {
          for (j2 = 0; j2 < *nk; j2++) {
            for (l = 0; l < *nk; l++) {
              if (l + 1 != data[indices[*knn * i + j1]]) {
                Wtilde[*knn * (*nk * j2 + l) + j1] *= (-prop[l]);
              }
              else {
                Wtilde[*knn * (*nk * j2 + l) + j1] *= (1.0 - prop[l]);
              }
            }
          }
        }
      }

#if __VOPENMP
      #pragma omp parallel for default(shared) private(j1, j2, l) schedule(static, 1)
#endif
      for (j2 = 0; j2 < *nk; j2++) {
        probs[*nrs * j2 + i] = prop[j2] * (1.0 - (double) *ordinary);
        for (l = 0; l < *nk; l++) {
          for (j1 = 0; j1 < *knn; j1++) {
            probs[*nrs * j2 + i] += Wtilde[(*knn + *ordinary) * (*nk * l + j2) + j1];
          }
        }
        if (probs[*nrs * j2 + i] > 1.0) probs[*nrs * j2 + i] = 1.0;
        if (probs[*nrs * j2 + i] < 0.0) probs[*nrs * j2 + i] = 0.0;
      }
      stds = 0.0; 
      for (j2 = 0; j2 < *nk; j2++) {
        stds += probs[*nrs * j2 + i];
      }
      if (stds == 0.0) {
        for (j2 = 0; j2 < *nk; j2++) {
          probs[*nrs * j2 + i] = prop[j2] * (1.0 - (double) *ordinary);
            for (l = 0; l < *nk; l++) {
            for (j1 = 0; j1 < *knn; j1++) {
              probs[*nrs * j2 + i] += Wtilde[(*knn + *ordinary) * (*nk * l + j2) + j1];
            }
          }
          if (probs[*nrs * j2 + i] > 1.0) probs[*nrs * j2 + i] = 1.0;
          if (probs[*nrs * j2 + i] < stds) stds = probs[*nrs * j2 + i];
        }
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j2) schedule(static, 1)
#endif
        for (j2 = 0; j2 < *nk; j2++) {
          probs[*nrs * j2 + i] -= stds;
        }
        stds = probs[i];
        for (j2 = 1; j2 < *nk; j2++) {
          stds += probs[*nrs * j2 + i];
        }
      }
      if (stds == 0.0) {
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j2) schedule(static, 1)
#endif
        for (j2 = 0; j2 < *nk; j2++) {
          probs[*nrs * j2 + i] = prop[j2];
        }
      }
      else {
#if __VOPENMP
        #pragma omp parallel for default(shared) private(j2) schedule(static, 1)
#endif
        for (j2 = 0; j2 < *nk; j2++) {
          probs[*nrs * j2 + i] /= stds;
        }
      }
    }
    g++;
  }

  free(Ttilde);
  free(invTtilde);
  free(revcoef);
  free(Wtilde);
  free(Otilde);

#if __VOPENMP
  #pragma omp parallel
  {
#endif
    free(TtLag);
    free(tmpMat);
    free(pv);
#if __VOPENMP
  }
#endif
}

void cEmbFrq(double *s, int *nk, int *mt, double *eps, double *f) {
  /* Maximum Entropy estiamtion of Embedded Frequencies
         *s - vector of proportions divied by the mean lenghts
        *nk - number of categories
       *eps - double value of epsilon (test for convergence)
         *f - vector of embedded frequencies */

  int i, j, iter;
  double mysum;
  double *fold, *Fmat, *vet;

  if ((fold = (double *) malloc(*nk * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  if ((Fmat = (double *) malloc(*nk * *nk * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  if ((vet = (double *) malloc(*nk * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }

#if __VOPENMP
  #pragma omp parallel for default(shared) private(i) schedule(static, 1)
#endif
  for (i = 0; i < *nk; i++) {
    fold[i] = s[i];
  }
  for (iter = 0; iter < *mt; iter++) {
#if __VOPENMP
    #pragma omp parallel for default(shared) private(i, j, mysum) schedule(static, 1)
#endif
    for (i = 0; i < *nk; i++) {
      mysum = 0.0;
      for (j = 0; j < i; j++) {
        Fmat[*nk * i + j] = fold[i] * fold[j];
        mysum += Fmat[*nk * i + j];
      }
      for (j = i + 1; j < *nk; j++) {
        Fmat[*nk * i + j] = fold[i] * fold[j];
        mysum += Fmat[*nk * i + j];
      }
      Fmat[(*nk + 1) * i] = mysum;
    }
    mysum = 0.0;
    for (i = 0; i < *nk; i++) {
      mysum = mysum + Fmat[(*nk + 1) * i];
    }
#if __VOPENMP
    #pragma omp parallel for default(shared) private(i) schedule(static, 1)
#endif
    for (i = 0; i < *nk; i++) {
      f[i] = s[i] * mysum / Fmat[(*nk + 1) * i];
      vet[i] = fabs(f[i] - fold[i]);
    }
    mysum = vet[0];
    j = 0;
    for (i = 1; i < *nk; i++) {
      if (vet[i] > mysum) {
        mysum = vet[i];
        j = i;
      }
    }
    if (mysum < *eps) break;
#if __VOPENMP
    #pragma omp parallel for default(shared) private(i) schedule(static, 1)
#endif
    for (i = 0; i < *nk; i++) {
      fold[i] = f[i];
    }
  }
  free(fold);
  free(Fmat);
  free(vet);
}

void pathAlg(int *nrs, int *nrorig, int *nc, double *coords, double *grid, int *path, double *radius, int *nk, int *data, double *coefs, double *prop, double *prhat, int *pred) {
/* "Posterior" transition probabilities approximation and prediction (Path Based Algorithms)
           *nrs - number of simulation coordinates
        *nrorig - number of data coordinates
            *nc - sample space dimention
        *coords - matrix of data coordinates
          *grid - matrix of simulation coordinates
          *path - vector of sequential indices
      *mainDire - main directions
        *radius - searching radius for a sphere
            *nk - number of categories
          *data - vector of observed categories
         *coefs - matrices of coefficients
          *prop - vector of proportions
         *prhat - matrix of probability vectors
          *pred - vector of predicted categories */

  int i, j , k, l, nr, zeros, pmx, nlen, np;
  int *wh, *cnt, *mydata, *ndata, *which, *pos, *vndata;
  double vmx;
  double *wgmLags, *site, *neighbour, *nbLength, *Tmat;

  if ((site = (double *) malloc(*nc * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  for (i = 0; i < *nrs; i++) {
    nr = *nrorig + i;
    zeros = 0;
    if ((wgmLags = (double *) malloc((*nc + 1) * nr * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((mydata = (int *) malloc(nr * sizeof(int))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((wh = (int *) malloc(nr * sizeof(int))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    /* COPYING COORDINATES AND DATA */
// #if __VOPENMP
//     #pragma omp parallel for default(shared) private(j, k) schedule(static, 1)
// #endif
    for (j = 0; j < *nrorig; j++) {
      mydata[j] = data[j];
      for (k = 0; k < *nc; k++) {
        wgmLags[nr * k + j] = coords[*nrorig * k + j];  
      }
    }
// #if __VOPENMP
//     #pragma omp parallel for default(shared) private(j, k) schedule(static, 1)
// #endif
    for (j = 0; j < i; j++) {
      mydata[*nrorig + j] = pred[path[j] - 1];
      for (k = 0; k < *nc; k++) {
        wgmLags[nr * k + *nrorig + j] = grid[*nrs * k + path[j] - 1];
      }
    }
// #if __VOPENMP
//     #pragma omp parallel for default(shared) private(j) schedule(static, 1)
// #endif
    for (j = 0; j < *nc; j++) {
      site[j] = grid[*nrs * j + path[i] - 1];
    }
    /* COMPUTING LAGS AND DISTANCES */
    getDst(nc, &nr, site, wgmLags, wgmLags);
    /* CHECKING FOR ZERO LAGS AND LAGS WITHIN THE SEARCHING SPHERE */
// #if __VOPENMP
//     #pragma omp parallel for default(shared) private(j) schedule(static, 1) reduction(+ : zeros)
// #endif
    for (j = 0; j < nr; j++) {
      wh[j] = j;
      if (wgmLags[*nc * nr + j] == 0.0) zeros = zeros + 1;
      if (wgmLags[*nc * nr + j] > *radius) wh[j] = -1;
    }
    if (zeros) {
      if ((cnt = (int *) malloc(*nk * sizeof(int))) == NULL) {
#if __VOPENMP
        #pragma omp critical
#endif
        error("%s", myMemErr);
      }
// #if __VOPENMP
//       #pragma omp parallel for default(shared) private(j) schedule(static, 1)
// #endif
      for (j = 0; j < *nk; j++) {
        cnt[j] = 0;
      }
      /* COMPUTING PROBABILITIES AND PREDICTION IF ZERO LAGS ARE PRESENT */
      for (j = 0; j < nr; j++) {
        if (wgmLags[*nc * nr + j] == 0.0) ++cnt[mydata[j] - 1];
      }
      prhat[path[i] - 1] = ((double) cnt[0]) / ((double) zeros);
      pmx = 0;
      vmx = prhat[path[i] - 1];
      for (j = 1; j < *nk; j++) {
        prhat[*nrs * j + path[i] - 1] = ((double) cnt[j]) / ((double) zeros);
        if (vmx < prhat[*nrs * j + path[i] - 1]) {
          pmx = j;
          vmx = prhat[*nrs * j + path[i] - 1];
        }
      }
      pred[path[i] - 1] = pmx + 1;
      free(cnt);
    }
    else {
      /* COMPUTING NUMBER OF POINTS INSIDE THE SEARCHING SPHERE */
      nlen = 0;
// #if __VOPENMP
//       #pragma omp parallel for default(shared) private(j) schedule(static, 1) reduction(+ : nlen)
// #endif
      for (j = 0; j < nr; j++) {
        if (wh[j] >= 0) nlen = nlen + 1;
      }
      if (nlen) { 
        /* COPYING NEIGHBOURS DATA IF THEY ARE FOUND */
        if ((neighbour = (double *) malloc(*nc * nlen * sizeof(double))) == NULL) {
#if __VOPENMP
          #pragma omp critical
#endif
          error("%s", myMemErr);
        }
        if ((ndata = (int *) malloc(nlen * sizeof(int))) == NULL) {
#if __VOPENMP
          #pragma omp critical
#endif
          error("%s", myMemErr);
        }
        if ((nbLength = (double *) malloc(nlen * sizeof(double))) == NULL) {
#if __VOPENMP
          #pragma omp critical
#endif
          error("%s", myMemErr);
        }
        if ((which = (int *) malloc(nlen * sizeof(int))) == NULL) {
#if __VOPENMP
          #pragma omp critical
#endif
          error("%s", myMemErr);
        }
        if ((pos = (int *) malloc((1 << *nc) * sizeof(int))) == NULL) {
#if __VOPENMP
          #pragma omp critical
#endif
          error("%s", myMemErr);
        }
        k = 0;
        for (j = 0; j < nr; j++) {
          if (wh[j] >= 0) {
// #if __VOPENMP
//             #pragma omp parallel sections default(shared)
//             {
//               #pragma omp section
//               {
// #endif
                for (l = 0; l < *nc; l++) neighbour[nlen * l + k] = wgmLags[nr * l + j];
// #if __VOPENMP
//               }
//               #pragma omp section
//               {
// #endif
                ndata[k] = mydata[j];
// #if __VOPENMP
//               }
//               #pragma omp section
//               {
// #endif
                nbLength[k] = wgmLags[*nc * nr + j];
// #if __VOPENMP
//               }
//             }
// #endif
            ++k;
          }
        }
        /* FINDING CLOSER NEIGHBOURS NEAR THE MAIN DIRECTIONS */
        nearDire(nc, &nlen, neighbour, which);
        getPos(nbLength, which, &np, &nlen, nc, pos);
        free(nbLength);
        if ((nbLength = (double *) malloc(*nc * np * sizeof(double))) == NULL) {
#if __VOPENMP
          #pragma omp critical
#endif
          error("%s", myMemErr);
        }
        if ((vndata = (int *) malloc(np * sizeof(int))) == NULL) {
#if __VOPENMP
          #pragma omp critical
#endif
          error("%s", myMemErr);
        }
        k = 0;
        for (j = 0; j < (1 << *nc); j++) {
          if (pos[j] >= 0) {
            vndata[k] = ndata[pos[j]];
// #if __VOPENMP
//             #pragma omp parallel for default(shared) private(l) schedule(static, 1)
// #endif
            for (l = 0; l < *nc; l++) {
              nbLength[*nc * k + l] = neighbour[nlen * l + pos[j]];
            }
            ++k;
          }
        }
        free(neighbour);
        free(ndata);
        free(which);
        free(pos);
        /* COMPUTING PROBABILITIES IF SOME NEIGHBOURS ARE FOUND */
        if ((Tmat = (double *) malloc(*nk * *nk * np * sizeof(double))) == NULL) {
#if __VOPENMP
          #pragma omp critical
#endif
          error("%s", myMemErr);
        }
        predMULTI(coefs, prop, nbLength, nk, nc, &np, Tmat);
        if ((neighbour = (double *) malloc(*nk * sizeof(double))) == NULL) {
#if __VOPENMP
          #pragma omp critical
#endif
          error("%s", myMemErr);
        }
        jointProbs(&np, nk, vndata, Tmat, neighbour);
// #if __VOPENMP
//         #pragma omp parallel for default(shared) private(j) schedule(static, 1)
// #endif
        for (j = 0; j < *nk; j++) prhat[*nrs * j + path[i] - 1] = neighbour[j];
        free(Tmat);
        free(vndata);
        free(nbLength);
        free(neighbour);
      }
      else {
        /* COMPUTING PROBABILITIES IF NO NEIGHBOURS ARE FOUND */
// #if __VOPENMP
//         #pragma omp parallel for default(shared) private(j) schedule(static, 1)
// #endif
        for (j = 0; j < *nk; j++) prhat[*nrs * j + path[i] - 1] = prop[j];
      }

      /* COMPUTING PREDICTION */
      pmx = 0;
      vmx = prhat[path[i] - 1];
      for (j = 1; j < *nk; j++) {
        if (vmx < prhat[*nrs * j + path[i] - 1]) {
          pmx = j;
          vmx = prhat[*nrs * j + path[i] - 1];
        }
      }
      pred[path[i] - 1] = pmx + 1;
    }
    free(wgmLags);
    free(mydata);
    free(wh);
  }
  free(site);
}

void getPos(double *nbLength, int *which, int *np, int *nlen, int *nc, int *pos) {
  /* Computing vector position of the nearest orthogonal position
       *nbLength - lengths of neighbour lags
          *which - definition of main directions
             *np - number of non negative positions
           *nlen - number of nearest points
             *nc - sample space dimention
            *pos - nearest orthogonal position*/

  int i, j, cnp;
  int *idx;

  if ((idx = (int *) malloc(*nlen * sizeof(int))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i) schedule(static, 1)
#endif
  for (i = 0; i < *nlen; i++) {
    idx[i] = i;
  }
  rsort_with_index(nbLength, idx, *nlen);
  cnp = (1 << *nc);
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j) schedule(static, 1)
#endif
  for (i = 0; i < cnp; i++) {
    for (j = 0; j < *nlen; j++) {
      if (i == which[idx[j]]) break;
    }
    pos[i] = (j < *nlen) ? idx[j] : -1;
  }
  j = cnp;
  cnp = 0;
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i) schedule(static, 1) reduction(+ : cnp)
#endif
  for (i = 0; i < j; i++) {
    if (pos[i] >= 0) cnp = cnp + 1;
  }
  *np = cnp;
  free(idx);
}

void nearDire(int *nc, int *nlen, double *neighbour, int *which) {
  /* Find the nearest points in the $2 * nc$ directions
         *nc - dimension of coordinates
       *nlen - number of nearest points
  *neighbour - known nearest loacations
      *which - main directions as result */

  int i, j;
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j) schedule(static, 1)
#endif
  for (i = 0; i < *nlen; i++) {      // neighbour cycle
    which[i] = 0;
    for (j = 0; j < *nc; j++) {      // mainDire cycle
      if (neighbour[*nlen * j + i] > 0.0) which[i] += (1 << j);
    }
  }
}

void objfun(int *nrs, int *nk, int *nc, int *mySim, double *grid, double *coef, double *prop, double *res) {
  /* Computing objective function through transition probabilities
          *nrs - number of simulation rows
           *nk - number of categories
           *nc - number of columns
        *mySim - vector of simulated categories
         *grid - simulation grid
        *coefs - matrices of coefficients
         *prop - vector of proportions
          *res - numerical result of the objective function */
  int i, j, k, nk2;
  double reso = 0.0;
  double *revcoef;

  nk2 = *nk * *nk;
  if ((revcoef = (double *) malloc(nk2 * *nc * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }

#if __VOPENMP
  #pragma omp parallel shared(nc, nk2, myMemErr)
  {
#endif
    if ((tmpMat = (double *) malloc(nk2 * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((TtLag = (double *) malloc(*nc * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
#if __VOPENMP
  }

  #pragma omp parallel for default(shared) private(k) schedule(static, 1)
#endif
  for (k = 0; k < *nc; k++) {
    revCoef(&coef[nk2 * k], prop, nk, &revcoef[nk2 * k]);
  }
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j, k) reduction(+ : reso) schedule(static, 1)
#endif
  for (i = 0; i < *nrs; i++) {
    for (j = 0; j < *nrs; j++) {
      if (i != j) {
        // Compute the transition probabilities through the model
        for (k = 0; k < *nc; k++) {
          TtLag[k] = grid[*nrs * k + j] - grid[*nrs * k + i];
        }
        predVET(coef, revcoef, nk, nc, TtLag, tmpMat);
        // Compute the penalty value
        tmpMat[mySim[j] * *nk - *nk + mySim[i] - 1] -= 1.0;
        for (k = 0; k < nk2; k++) {
          reso = reso + fabs(tmpMat[k]);
        }
      }
    }
  }
  *res = reso;
#if __VOPENMP
  #pragma omp parallel
  {
#endif
    free(tmpMat);
    free(TtLag);
#if __VOPENMP
  }
#endif
  free(revcoef);
}

void fastobjfun(int *knn, int *indices, int *nrs, int *nk, int *nc, int *nr, int *mySim, double *grid, double *coef, double *prop, int *data, double *coords, double *res) {
  /* Fast computation of the objective function through transition probabilities
          *knn - number of neighbours considered
      *indices - row indices of the nearest observations
          *nrs - number of simulation rows
           *nk - number of categories
           *nc - number of columns
           *nr - number of observation rows
        *mySim - vector of simulated categories
         *grid - simulation grid
        *coefs - matrices of coefficients
         *prop - vector of proportions
         *data - vector of observed categories
       *coords - matrix of coordinates
          *res - numerical result of the objective function */
  int i, j, k, nk2;
  double reso = 0.0;
  double *revcoef;

  nk2 = *nk * *nk;
  if ((revcoef = (double *) malloc(nk2 * *nc * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }

#if __VOPENMP
  #pragma omp parallel shared(nc, nk2, myMemErr)
  {
#endif
    if ((tmpMat = (double *) malloc(nk2 * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((TtLag = (double *) malloc(*nc * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
#if __VOPENMP
  }

  #pragma omp parallel for default(shared) private(k) schedule(static, 1)
#endif
  for (k = 0; k < *nc; k++) {
    revCoef(&coef[nk2 * k], prop, nk, &revcoef[nk2 * k]);
  }
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j, k) reduction(+ : reso) schedule(static, 1)
#endif
  for (i = 0; i < *nrs; i++) {
    for (j = 0; j < *knn; j++) {
      // Compute the transition probabilities through the model
      for (k = 0; k < *nc; k++) {
        TtLag[k] = coords[*nr * k + indices[*knn * i + j]] - grid[*nrs * k + i];
      }
      predVET(coef, revcoef, nk, nc, TtLag, tmpMat);
      // Compute the penalty value
      tmpMat[data[indices[*knn * i + j]] * *nk - *nk + mySim[i] - 1] -= 1.0;
      for (k = 0; k < nk2; k++) {
        reso = reso + fabs(tmpMat[k]);
      }
    }
  }
  *res = reso;
#if __VOPENMP
  #pragma omp parallel
  {
#endif
    free(tmpMat);
    free(TtLag);
#if __VOPENMP
  }
#endif
  free(revcoef);
}
