void transCount(int *n, int *data, int *nc, double *coords, double *dire, double *tolerance,
                int *mpoints, double *bins, int *nk, double *trans) {
  /* Counting occurrences with the same direction     
          *n - sample size
       *data - vector of data
         *nc - dimension of the sample space
     *coords - matrix of coordinates
       *dire - fixed direction vector
  *tolerance - angle tolecrance in radians
    *mpoints - number of estimation points
       *bins - vector of lags
         *nk - number of categories
      *trans - tensor of transaction ocuurrences */

  int i, j, x, xh, ToF;
  double *d = NULL;

#if __VOPENMP
  #pragma omp parallel shared(nc, myMemErr)
  {
#endif
    if ((h = (double *) malloc(*nc * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((p = (double *) malloc(*nc * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
#if __VOPENMP
  }
#endif

  if ((d = (double *) calloc(*nc, sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }

  nsph2(nc, dire, d); // polar transformation of dire

#if __VOPENMP
  #if (_OPENMP > 201100)
    #pragma omp parallel for default(shared) private(i, j, x, xh, ToF) schedule(static, 1) collapse(2)
  #else
    #if (_OPENMP > 200800)
      #pragma omp parallel for default(shared) private(i, j, x, xh, ToF) schedule(static, 1)
    #endif
  #endif
#endif
  for (x = 0; x < *n; x++) {        // These cilces are relative to
    for (xh = 0; xh < *n; xh++) {   // each pair of observations
      if (x == xh) continue; // skip when the two points coincide

      /* Computing of lag vector h and it's polar transformation */
      for (i = *nc - 1; i >= 0; i--) {
        h[i] = coords[*n * i + x] - coords[*n * i + xh];
        p[i] = 0.0; // polar coordinates
      }
      nsph2(nc, h, p); // polar transformation of h

      /* Counting occurence */
      ToF = 1;
      if (*nc == 1) {
        ToF &= (*p > 0);
      }
      else {
        if (*nc > 1) if (!ISNAN(d[1])) ToF &= (sin(fabs(p[1] - d[1]) * 0.5) <= sin(fabs(*tolerance * 0.5)));
        for (j = 2; j < *nc; j++) {
          if (!ISNAN(d[j])) {
            ToF &= (sin(fabs(p[j] - d[j])) <= sin(fabs(*tolerance)));
          }
        }
      }

      if (ToF != 0) {
        for (i = 0; i < *mpoints; i++) {
          if (p[0] <= bins[i]) {
#if __VOPENMP
            #pragma omp atomic
#endif
              ++trans[data[x] - 1 + *nk * (data[xh] - 1) + *nk * *nk * i];
            break;
          }
        }
      }

    }
  }

#if __VOPENMP
  #pragma omp parallel
  {
#endif
    free(h);
    free(p);
#if __VOPENMP
  }
#endif
  free(d);

}

void transProbs(int *mpoints, int *nk, double *rwsum, double *empTR) {
  /* Computing transition probabilities
      *mpoints - number of estimation points
           *nk - number of categories
        *rwsum - dimension of the sample space
        *empTR - tensor of transaction probabilities */

  int i, j, k;

#if __VOPENMP
  #if (_OPENMP > 201100)
    #pragma omp parallel for default(shared) private(i, j, k) schedule(static, 1) collapse(3)
  #else
    #if (_OPENMP > 200800)
      #pragma omp parallel for default(shared) private(i, j, k) schedule(static, 1)
    #endif
  #endif
#endif
  for (i = 0; i < *mpoints; i++) {
    for (j = 0; j < *nk; j++) {
      for (k = 0; k < *nk; k++) {
        empTR[*nk * k + j + *nk * *nk * i] /= rwsum[*nk * i + j];
      }
    }
  }

}

void transSE(int *mpoints, int *nk, double *rwsum, double *empTR, double *se) {
  /* Computing log odds transition probabilities standard errors
      *mpoints - number of estimation points
           *nk - number of categories
        *rwsum - dimension of the sample space
        *empTR - tensor of transition counting
	   *se - tensor of standard errors */

  int i, j, k, pos;

#if __VOPENMP
  #if (_OPENMP > 201100)
    #pragma omp parallel for default(shared) private(i, j, k) schedule(static, 1) collapse(3)
  #else
    #if (_OPENMP > 200800)
      #pragma omp parallel for default(shared) private(i, j, k) schedule(static, 1)
    #endif
  #endif
#endif
  for (i = 0; i < *mpoints; i++) {
    for (j = 0; j < *nk; j++) {
      for (k = 0; k < *nk; k++) {
        pos = *nk * k + j + *nk * *nk * i;
        se[pos] = (double) rwsum[*nk * i + j] / (double) (empTR[pos] * (rwsum[*nk * i + j] - empTR[pos]));
        se[pos] = sqrt(se[pos]);
      }
    }
  }

}

void transLogOdds(int *mdim, double *empTR, double *empTLO) {
  /* Computing the log odds of the transition probabilities
         *mdim - dimension of tensors
        *empTR - tensor of transition probabilities 
        *empTR - tensor of transition log odds */

  int i, j, k, pos;

#if __VOPENMP
  #if (_OPENMP > 201100)
    #pragma omp parallel for default(shared) private(i, j, k, pos) schedule(static, 1) collapse(3)
  #else
    #if (_OPENMP > 200800)
      #pragma omp parallel for default(shared) private(i, j, k, pos) schedule(static, 1)
    #endif
  #endif
#endif
  for (i = 0; i < mdim[2]; i++) {
    for (j = 0; j < *mdim; j++) {
      for (k = 0; k < *mdim; k++) {
        pos = *mdim * k + j + *mdim * *mdim * i;
        empTLO[pos] = log(empTR[pos] / (1.0 - empTR[pos])) ;
      }
    }
  }

}

void LogOddstrans(int *mdim, double *empTLO, double *empTR) {
  /* Computing transition probabilities from log odds
         *mdim - dimension of tensors
        *empTR - tensor of transition probabilities 
        *empTR - tensor of transition log odds */

  int i, j, k, pos;

#if __VOPENMP
  #if (_OPENMP > 201100)
    #pragma omp parallel for default(shared) private(i, j, k, pos) schedule(static, 1) collapse(3)
  #else
    #if (_OPENMP > 200800)
      #pragma omp parallel for default(shared) private(i, j, k, pos) schedule(static, 1)
    #endif
  #endif
#endif
  for (i = 0; i < mdim[2]; i++) {
    for (j = 0; j < *mdim; j++) {
      for (k = 0; k < *mdim; k++) {
        pos = *mdim * k + j + *mdim * *mdim * i;
        empTR[pos] = 1.0 / (1.0 + exp(- empTLO[pos]));
      }
    }
  }

}

void revtProbs(double *dmat, int *idim) {
  /* Computing reverse transition probabilities through Bayes theorem 
      *dmat - matrices of transition probabilities (usually an array)
      *idim - dimension of the array in dmat */

  int i, j, k, pos, cpos;
  double tsum, tmp;
#if __VOPENMP
  #pragma omp parallel for default(shared) private(pos, cpos, i, j, k, tsum, tmp) schedule(static, 1)
#endif
  for (i = 0; i < idim[2]; i++) {
    pos = *idim * *idim * i;
    for (j = 0; j < *idim; j++) {
      cpos = pos + *idim * j;
      for (k = 0, tsum = 0.0; k < *idim; k++) { // transposition
        tmp = dmat[cpos + k];
        tsum += tmp;
        if (k > j) {
          dmat[cpos + k] = dmat[pos + *idim * k + j];
          dmat[pos + *idim * k + j] = tmp;
        }
      }
      for (k = 0; k < *idim; k++) { // normalisation
        dmat[pos + *idim * k + j] /= tsum;
      }
    }
  }
  pos = *idim * *idim;
  cpos = idim[2] / 2;
#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j, k, tmp) schedule(static, 1)
#endif
  for (i = 0; i < cpos; i++) { // revert the probabilities in the array
    k = idim[2] - i - 1;
    for (j = 0; j < pos; j++) {
      tmp = dmat[pos * k + j];
      dmat[pos * k + j] = dmat[pos * i + j];
      dmat[pos * i + j] = tmp;
    }
  }

}
