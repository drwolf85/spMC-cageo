\name{sim}
\Rdversion{1.1}
\alias{sim}

\title{Random Field Simulation}

\description{
The function simulates a random field. The simulation methods available are based on Indicator Kriging techniques (IK and CK), Fixed and Random Path (PATH) and Multinomial Categorical Simulation (MCS).
}

\usage{sim(x, data, coords, grid, method = "ik", ...)}

\arguments{
%%          x a multi.tpfit object
%%       data vector of data
%%     coords coordinates matrix
%%       grid simulation points
%%     method method to perform prediction and simulation c("ik", "ck", "path", "mcs")
%%        ... further option to pass to the function sim_*
%%
%%   Further arguments for Indicator Kriging (ik) and coKriging (ck)
%%        knn number of k-nearest neighbours
%%   ordinary boolean (if TRUE ordinary Kriging is applied rather than simple Kriging)
%%         GA boolean (if TRUE genetic algorithm is applied rather than simulated annealing)
%%     optype character with the objective function to minimize after the simulation
%%     max.it maximum number of iteration for the optimization method
%%
%%   Further arguments for Fixed and Random Path methods (path)
%%   radius radius to find neighbour points
%%    fixed boolean for random or fixed path algorithm
%%
%%   Further arguments for Multinomial Categorical Simulation (mcs)
%%      knn number of k-nearest neighbours (if NULL all data are neighbours)
%%   radius radius to find neighbour points
  \item{x}{an object of the class \code{multi_tpfit}, typically with the output of the function \code{\link{multi_tpfit}}.}
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{grid}{an \eqn{m \times d}{m x d} matrix where each row denotes the \eqn{d}{d}-D coordinates in the simulation grid.}
  \item{method}{a character object specifying the method to simulate the random field. Possible choises are \code{"ik"} (by default) for the indicator Kriging, \code{"ck"} for the indicator coKriging, \code{"path"} for the fixed and random path and \code{"mcs"} for the multinomial categorical simulation method.}
  \item{\dots}{other arguments to pass to the functions \code{\link{sim_ik}}, \code{\link{sim_ck}}, \code{\link{sim_path}} or \code{\link{sim_mcs}}.}
}

\details{
The methods implemented compute the approximation of posterior probabilities
\deqn{\Pr\left(Z(\mathbf{s}_0) = z_k \left\vert \bigcap_{i = 1}^n Z(\mathbf{s}_i) = z(\mathbf{s}_i)\right.\right).}{Pr(Z(s_0) = z_k | Z(s_i) = z(s_i)),}
\eqn{\hspace{0cm}}{where i = 1, ..., n.}

Once the probabilities are calculated for all the points in the simulation grid, the predictions (based on most probable category) and simulations are returned.
}

\value{A data frame containing the simulation grid, the simulated random field, predicted values and the approximated probabilities.}

\references{Allard, D., D'Or, D., Froidevaux, R. (2011) An efficient maximum entropy approach for categorical variable prediction. \emph{European Journal of Soil Science}, \bold{62}(3), 381-393.

Carle, S. F., Fogg, G. E. (1996) Transition Probability-Based Indicator Geostatistics. \emph{Mathematical Geosciences}, \bold{28}(4), 453-476.

Carle, S. F. (1999) T-PROGS: Transition Probability Geostatistical Software. University of California, Davis.

Li, W. (2007) A Fixed-Path Markov Chain Algorithm for Conditional Simulation of Discrete Spatial Variables. \emph{Mathematical Geology}, \bold{39}(2), 159-176.

Li, W. (2007) Markov Chain Random Fields for Estimation of Categorical Variables. \emph{Mathematical Geology}, \bold{39}(June), 321-335.

Pickard, D. K. (1980) Unilateral Markov Fields. \emph{Advances in Applied Probability}, \bold{12}(3), 655-671.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.

%Tolosana-Delgado, R., Pawlowsky-Glahn, V., Egozcue, J.J. (2008) Indicator Kriging without Order Relation Violations. \emph{Mathematical Geosciences}, \bold{40}(3), 327-347.

Weise, T. (2009) \emph{Global Optimization Algorithms - Theory and Application}. \url{http://www.it-weise.de/}.}

\seealso{\code{\link{sim_ik}}, \code{\link{sim_ck}}, \code{\link{sim_path}}, \code{\link{sim_mcs}}}

\examples{\donttest{
data(ACM)

# Model parameters estimation for the
# multinomial categorical simulation
x <- multi_tpfit(ACM$MAT5, ACM[, 1:3])

# Generate the simulation grid
mygrid <- list()
mygrid$X <- seq(min(ACM$X), max(ACM$X), length = 20)
mygrid$Y <- seq(min(ACM$Y), max(ACM$Y), length = 20)
mygrid$Z <- -40 * 0:9 - 1
mygrid <- as.matrix(expand.grid(mygrid$X, mygrid$Y, mygrid$Z))

# Simulate the random field through
# Simple Indicator Kriging algorithm and
mySim <- sim(x, ACM$MAT5, ACM[, 1:3], mygrid)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
