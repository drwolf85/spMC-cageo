\name{spMC-package}
\Rdversion{1.1}
\alias{spMC-package}
\alias{spMC}
\docType{package}

\title{Continuous Lag Spatial Markov Chains}

\description{
The main goal of this package is to provide a set of functions for
 \enumerate{
  \item the stratum lengths analysis along a chosen direction,
  \item fast estimation of continuous lag spatial Markov chains model parameters and probability computing (also for large data sets), 
  \item transition probability maps and transiograms drawing, 
  \item simulation methods for categorical random fields.
 }
}

\details{
\tabular{ll}{
Package: \tab spMC\cr
Type: \tab Package\cr
Version: \tab 0.3.6.3153\cr
Date: \tab 2015-09-24\cr
License: \tab GPL (>= 2)\cr
LazyLoad: \tab yes\cr
}

Several functions are available for the stratum lengths analysis, in particular they compute the stratum lengths for each stratum category, they compute the empirical distributions and many other tools for a graphical analysis.

Usually, the basic inputs for the most of the functions are a vector of categorical data and their location coordinates. They are used to estimate empirical transition probabilities (\link{transiogram}), to estimate model parameters (\link{tpfit} for one-dimensional Markov chains or \link{multi_tpfit} for multidimensional Markov chains). Once parameters are estimated, it's possible to compute theoretical transition probabilities by the use of the function \link{predict.tpfit} for one-dimensional Markov chains and \link{predict.multi_tpfit} for multidimensional ones.

The function \link{plot.transiogram} allows to plot one-dimensional transiograms, while \link{image.multi_tpfit} permit to draw transition probability maps. A powerful tool to explore graphically the anisotropy of such process is given by the functions \link{pemt} and \link{image.pemt}, which let the user to draw "quasi-empirical" transition probability maps.

Simulation methods are based on Indicator Kriging (\link{sim_ik}), Indicator Cokriging (\link{sim_ck}), Fixed or Random Path algorithms (\link{sim_path}) and Multinomial Categorical Simulation technique (\link{sim_mcs}).
}

\author{
Luca Sartore

Maintainer: Luca Sartore \email{drwolf85@gmail.com}
}

\references{
Allard, D., D'Or, D., Froidevaux, R. (2011) An efficient maximum entropy approach for categorical variable prediction. \emph{European Journal of Soil Science}, \bold{62}(3), 381-393.

Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Dynkin, E. B. (1961) \emph{Theory of Markov Processes}. Englewood Cliffs, N.J.: Prentice-Hall, Inc.

Higham, N. J. (2008) \emph{Functions of Matrices: Theory and Computation}. Society for Industrial and Applied Mathematics.

Li, W. (2007) A Fixed-Path Markov Chain Algorithm for Conditional Simulation of Discrete Spatial Variables. \emph{Mathematical Geology}, \bold{39}(2), 159-176.

Li, W. (2007) Markov Chain Random Fields for Estimation of Categorical Variables. \emph{Mathematical Geology}, \bold{39}(June), 321-335.

Li, W. (2007) Transiograms for Characterizing Spatial Variability of Soil Classes. \emph{Soil Science Society of America Journal}, \bold{71}(3), 881-893.

Pickard, D. K. (1980) Unilateral Markov Fields. \emph{Advances in Applied Probability}, \bold{12}(3), 655-671.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.

%Tolosana-Delgado, R., Pawlowsky-Glahn, V., Egozcue, J.J. (2008) Indicator Kriging without Order Relation Violations. \emph{Mathematical Geosciences}, \bold{40}(3), 327-347.

Weise, T. (2009) \emph{Global Optimization Algorithms - Theory and Application}. \url{http://www.it-weise.de/}.
}

\keyword{package}
\keyword{spatial}
