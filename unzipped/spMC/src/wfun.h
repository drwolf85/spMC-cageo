void wl(int *n, int *nc, double *coords, double *dire, double *tolerance, int *id) {
  /* Computing segments indicator  
          *n - sample size
         *nc - dimension of the sample space
     *coords - matrix of coordinates
       *dire - fixed direction vector
  *tolerance - angle tolecrance in radians
         *id - location id */

  int x, xh, i, ToF, mpos;
  double mymin = 0.0, *d = NULL, *mdst = NULL;

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

  if ((d = (double *) malloc(*nc * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }
  for (i = 0; i < *nc; i++) d[i] = 0.0;
  nsph(nc, dire, d); // polar transformation of dire
  for (x = 0; x < *n - 1; x++) { 
    if (id[x] == 0) id[x] = x + 1;
    if ((mdst = (double *) malloc((*n - x - 1) * sizeof(double))) == NULL) {
#if __VOPENMP
      #pragma omp critical
#endif
      error("%s", myMemErr);
    }
#if __VOPENMP
    #pragma omp parallel for default(shared) private(xh, i, ToF) schedule(static, 1)
#endif
    for (xh = x+1; xh < *n; xh++) {
      for (i = *nc - 1; i >= 0; i--) {
        h[i] = coords[*n * i + x] - coords[*n * i + xh];
        p[i] = 0.0; // polar coordinates
      }
      nsph(nc, h, p); // polar transformation of h
      ToF = 1;
      for (i = 1; i < *nc; i++) {
        if (!ISNAN(d[i]) && !ISNAN(p[i])) {
          ToF &= (sin(fabs(p[i] - d[i]) * 0.5) <= sin(0.5 * fabs(*tolerance)));
//           if (fabs(d[i]) != PI / 2.0) { 
//             ToF &= (p[i] >= d[i] - fabs(*tolerance));
//             ToF &= (p[i] <= d[i] + fabs(*tolerance));
//           }
//           else {
//             ToF &= (fabs(p[i]) >= PI / 2.0 - fabs(*tolerance));
//           }
        }
      }
      if (ToF) {
        mdst[xh - x - 1] = p[0];
      }
      else {
        mdst[xh - x - 1] = -p[0];
      }
    }
    mpos = -1;
    for (xh = x+1; xh < *n; xh++) {
      if (mdst[xh - x - 1] >= 0.0) {
        mymin = mdst[xh - x - 1];
        mpos = xh;
        break;
      }
    }
    for (xh++; xh < *n; xh++) {
      if ((mdst[xh - x - 1] < mymin) && (mdst[xh - x - 1] >= 0.0)) {
        mymin = mdst[xh - x - 1];
        mpos = xh;
      }
    }
    if (mpos > -1) {
      id[mpos] = id[x];
    }
    free(mdst);
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

void nsph(int *di, double *x, double *res) {
  /* Computing polar transformation
         *di - number of coordinates to transform
          *x - vector of coordinates
        *res - transformed vector */

  int i, j;
  if (*di > 0 && *di < 2) *res = fabs(*x);
  if (*di >= 2) { // Angles calculations
    res[*di - 1] = atan(x[*di - 1] / x[*di - 2]);
    for (i = *di - 2; i >= 0; i--) {
      for (j = *di - 1; j >= i; j--) {
        res[i] += R_pow_di(x[j], 2);
      }
      res[i] = sqrt(res[i]);
      if (i != 0) res[i] = atan(res[i] / x[j]);
    }
  }
}

void wd(double *lags, int *nc, int *nr, int *res) {
  /* Find points with the same direction 
       *lags - matrix whose columns are lags
         *nc - sample space dimension
         *nr - number of lags
        *res - location id */

  int i, j, k, ToF;
  double *plags;

  if ((plags = (double *) malloc(*nr * *nc * sizeof(double))) == NULL) {
#if __VOPENMP
    #pragma omp critical
#endif
    error("%s", myMemErr);
  }

#if __VOPENMP
  #pragma omp parallel for default(shared) private(i, j) schedule(static, 1)
#endif
  for (i = 0; i < *nr; i++) {
    for (j = 0; j < *nc; j++) plags[*nc * i + j] = 0.0;
    nsph(nc, &lags[*nc * i], &plags[*nc * i]);
  }
  for (i = 0; i < *nr - 1; i++) {
    if (res[i] == 0) {
      res[i] = i + 1;
#if __VOPENMP
      #pragma omp parallel for default(shared) private(j, k, ToF) schedule(static, 1)
#endif
      for (j = i + 1; j < *nr; j++) {
        if (res[j] == 0) {
          ToF = 1;
          for (k = 1; k < *nc; k++) {
            if (!ISNAN(plags[*nc * i + k]) && !ISNAN(plags[*nc * j + k])) {
              ToF &= (plags[*nc * i + k] == plags[*nc * j + k]);
            }
            else {
              if(!ISNAN(plags[*nc * i + k]) || !ISNAN(plags[*nc * j + k])) ToF = 0;
            }
          }
          if (ToF) {
            res[j] = i + 1;
          }
        }
      }
    }
  }
  if (res[*nr - 1] == 0) res[*nr - 1] = *nr;
  free(plags);
}

void nsph2(int *di, double *x, double *res) {
  /* Computing true polar transformation
         *di - number of coordinates to transform
          *x - vector of coordinates
        *res - transformed vector */

  int i;

  if (*di > 0 && *di < 2) *res = *x;
  if (*di >= 2) { // Angles calculations
   *res = *x * *x;
   *res +=  x[1] * x[1];
   res[1] = atan2(*x, x[1]); // the first angle % with opposite sign?
    for (i = 2; i < *di; i++) {
      *res += x[i] * x[i];
      res[i] = acos(x[i] / sqrt(*res));
    }
    // Radius computation
    *res = sqrt(*res);
  }
}
