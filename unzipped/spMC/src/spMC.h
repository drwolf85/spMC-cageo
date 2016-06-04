#if defined _OPENMP
  #if (_OPENMP > 200800)
    #include <omp.h>
    #define __VOPENMP 1
  #else
    #define __VOPENMP 0
  #endif
#else
  #define __VOPENMP 0
#endif
#include <stdio.h>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/Lapack.h>
#include <R_ext/Boolean.h>

int *wo = NULL, *pv = NULL;
double *h = NULL, *p = NULL, *TtLag = NULL, *tmpMat = NULL;
char myMemErr[] = "There is not enough empty memory";

#if __VOPENMP
  #pragma omp threadprivate(wo, pv, h, p, TtLag, tmpMat)
#endif

/* spMC.c */
void cEmbedLen(int *, int *, double *, int *, int *, int *, double *, double *);
void cEmbedOc(int *, int *, int *, double *, int *, int *, int *, double *);
void cEmbedTrans(int *, int *, int *, int *, int *);
void embedTProbs(int *, double *);
void ellinter(int *, int *, double *, double *, double *);
void getDst(int *, int *, double *, double *, double *);
void fastMatProd(int *, int *, double *, int *, double *, double *);
void jointProbs(int *, int *, int *, double *, double *);
void tsimCate(int *, int *, double *, int *);
SEXP isOmp();
void getNumCores(int *);
void getNumSlaves(int *);
void setNumSlaves(int *);
void fastSVDprod(double *, double *, double *, int *);
SEXP annealingSIM(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
SEXP geneticSIM(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
void expmat(double *, int *, double *);
void predTPFIT(double *, double *, double *, int *, double *);
void nrmPrMat(double *, int *);
void revCoef(double *, double *, int *, double *);
void predVET(double *, double *, int *, int *, double *, double *);
void predMULTI(double *, double *, double *, int *, int *, int*, double *);
void predPSEUDOVET(double *, double *, int *, int *, int *, double *, double *);
void predPSEUDO(double *, double *, double *, int *, int *, int *, int *, int *, int *, double *);
void rotaH(int *, double *, double *);
void rotaxes(int *, double *, double *);
void fastrss(int *, double *, double *, double *);
SEXP bclm(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
void knear(int *, int *, double *, int *, double *, int *, int *);
void getIKPrbs(int *, int *, int *, int *, int *, int *, int *, int *, double *, double *, int *, double *, double *, double *);
void getCKPrbs(int *, int *, int *, int *, int *, int *, int *, int *, double *, double *, int *, double *, double *, double *);
void cEmbFrq(double *, int *, int *, double *, double *);
void pathAlg(int *, int *, int *, double *, double *, int *, double *, int *, int *, double *, double *, double *, int *);
void getPos(double *, int *, int *, int *, int *, int *);
void nearDire(int *, int *, double *, int *);
void objfun(int *, int *, int *, int *, double *, double *, double *, double *);
void fastobjfun(int *, int *, int *, int *, int *, int *, int *, double *, double *, double *, int *, double *, double *);

/* trans.h */
void transCount(int *, int *, int *, double *, double *, double *, int *, double *, int *, double *);
void transProbs(int *, int *, double *, double *);
void transProbs(int *, int *, double *, double *);
void transSE(int *, int *, double *, double *, double *);
void transLogOdds(int *, double *, double *);
void LogOddstrans(int *, double *, double *);
void revtProbs(double *, int *);

/* wfun.h */
void wl(int *, int *, double *, double *, double *, int *);
void wd(double *, int *, int *, int *);
void nsph(int *, double *, double *);
void nsph2(int *, double *, double *);

/* mcs.h */
void jointProbsMCS(double *, int *, double *, int *, int *, int *, int *, double *, double *, int*, double *);
void KjointProbsMCS(double *, int *, double *, int *, int *, int *, int *, int *, double *, int *, double *);
