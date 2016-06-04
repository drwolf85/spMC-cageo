void jointProbsMCS(double *coords, int *hmany, double *grid, int *nrs, int *nc, int *nk, int *ndata, double *coefs, double *matdir, int *rota, double *pProbs) {
  /* "Posterior" transition probabilities approximation (Multinomial Categorical Simulation)
       *coords - matrix of data coordinates
        *hmany - number of data coordinates
         *grid - matrix of simulation coordinates
          *nrs - number of simulation coordinates
           *nc - sample space dimention
           *nk - number of categories
        *ndata - vector of neighbour categories
        *coefs - matrices of coefficients
       *matdir - rotation matrix
         *rota - boolean (it's FALSE if matdir is identity)
       *pProbs - matrix of probability vectors */

  int i, j, k;
  double mysum, mymax;//, myexp;
  double *mycoef;

//   myexp = 1.0 / (double) *hmany;

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
    revCoef(&coefs[*nk * *nk * i], pProbs, nk, &mycoef[*nk * *nk * i]);
  }

#if __VOPENMP
  #pragma omp parallel shared(nc, nk, myMemErr)
  {
#endif
    if ((h = (double *) malloc(*nc * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
    if ((p = (double *) malloc(*nk * *nk * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
#if __VOPENMP
  }

  #pragma omp parallel for default(shared) private(i, j, k, mysum, mymax) schedule(static, 1)
#endif
  for (i = 0; i < *nrs; i++) { // loop for grid
    for (j = 0; j < *hmany; j++) { // loop for coords
      for (k = 0; k < *nc; k++) {
        h[k] = coords[*hmany * k + j] - grid[*nrs * k + i];
      }
      if (*rota) rotaH(nc, matdir, h);
      predVET(coefs, mycoef, nk, nc, h, p); // calculate transition probabilities
      if (!ISNAN(p[0])) {
        pProbs[*nk * i] *= p[*nk * (ndata[j] - 1)];
//         pProbs[*nk * i] *= pow(p[*nk * (ndata[j] - 1)], myexp);
        mymax = pProbs[*nk * i]; // control statistic for computational stability. It avoids problems due to the C.L.T.
        for (k = 1; k < *nk; k++) {
          pProbs[*nk * i + k] *= p[*nk * (ndata[j] - 1) + k];
//           pProbs[*nk * i + k] *= pow(p[*nk * (ndata[j] - 1) + k], myexp);
          if (mymax < pProbs[*nk * i + k]) mymax = pProbs[*nk * i + k];
        }
        if (mymax < 0.001) {
         for (k = 0; k < *nk; k++) {
            pProbs[*nk * i + k] *= 1000.0;
          }
        }
      }
    }
    mysum = pProbs[*nk * i];
    for (k = 1; k < *nk; k++) mysum += pProbs[*nk * i + k];
    for (k = 0; k < *nk; k++) pProbs[*nk * i + k] /= mysum;
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
  free(mycoef);
}

void KjointProbsMCS(double *coords, int *hmany, double *grid, int *nrs, int *nc, int *nk, int *ndata, int *knn, double *coefs, int *indices, double *pProbs) {
  /* "Posterior" transition probabilities approximation with k-nearest neighbours
        *coords - matrix of data coordinates            (Multinomial Categorical Simulation)
         *hmany - number of data coordinates
          *grid - matrix of simulation coordinates
           *nrs - number of simulation coordinates
            *nc - sample space dimention
            *nk - number of categories
         *ndata - vector of observed categories
           *knn - number of k-nearest neighbours
       *indices - matrix of indices
         *coefs - matrices of coefficients
        *pProbs - matrix of probability vectors */

  int i, j, k;
  double mysum, mymax;//, myexp;
  double *mycoef;

//   myexp = 1.0 / (double) *knn;

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
    revCoef(&coefs[*nk * *nk * i], pProbs, nk, &mycoef[*nk * *nk * i]);
  }

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
    if ((p = (double *) malloc(*nk * *nk * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
#if __VOPENMP
  }

  #pragma omp parallel for default(shared) private(i, j, k, mysum, mymax) schedule(static, 1)
#endif
  for (i = 0; i < *nrs; i++) { // loop for grid
    for (j = 0; j < *knn; j++) { // loop for coords
      for (k = 0; k < *nc; k++) {
        h[k] = coords[*hmany * k + indices[*knn * i + j]] - grid[*nrs * k + i];
      }
      predVET(coefs, mycoef, nk, nc, h, p);
      if (!ISNAN(p[0])) {
        pProbs[*nk * i] *= p[*nk * (ndata[j] - 1)];
//         pProbs[*nk * i] *= pow(p[*nk * (ndata[j] - 1)], myexp);
        mymax = pProbs[*nk * i];
        for (k = 1; k < *nk; k++) {
          pProbs[*nk * i + k] *= p[*nk * (ndata[j] - 1) + k];
//           pProbs[*nk * i + k] *= pow(p[*nk * (ndata[j] - 1) + k], myexp);
          if (mymax < pProbs[*nk * i + k]) mymax = pProbs[*nk * i + k];
        }
        if (mymax < 0.001) {
          for (k = 0; k < *nk; k++) {
            pProbs[*nk * i + k] *= 1000.0;
          }
        }
      }
    }
    mysum = pProbs[*nk * i];
    for (k = 1; k < *nk; k++) mysum += pProbs[*nk * i + k];
    for (k = 0; k < *nk; k++) pProbs[*nk * i + k] /= mysum;
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
  free(mycoef);
}
