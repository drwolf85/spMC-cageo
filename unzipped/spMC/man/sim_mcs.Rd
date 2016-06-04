\name{sim_mcs}
\Rdversion{1.1}
\alias{sim_mcs}

\title{Multinomial Categorical Simulation}

\description{
The function simulates a random field through the Multinomial Categorical Simulation technique (MCS).
}

\usage{sim_mcs(x, data, coords, grid, knn = NULL)}

\arguments{
%%        x a multi_tpfit object
%%     data vector of data
%%   coords coordinates matrix
%%     grid simulation points
%%      knn number of k-nearest neighbours (if NULL all data are neighbours)
  \item{x}{an object of the class \code{multi_tpfit}, typically with the output of the function \code{\link{multi_tpfit}}.}
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{grid}{an \eqn{m \times d}{m x d} matrix where each row denotes the \eqn{d}{d}-D coordinates in the simulation grid.}
  \item{knn}{an integer value which specifies the number of k-nearest neighbours for each simulation point. If \code{NULL} (by default), all observations are considered.}
}

\details{
This method computes an approximation of posterior probabilities 
\deqn{\Pr\left(Z(\mathbf{s}_0) = z_k \left\vert \bigcap_{i = 1}^n Z(\mathbf{s}_i) = z(\mathbf{s}_i)\right.\right).}{Pr(Z(s_0) = z_k | Z(s_i) = z(s_i)),}
\eqn{\hspace{0cm}}{where i = 1, ..., n.} The algorithm is based on the Bayesian maximum entropy approach and it honours both the model structure and observed data.
}

\value{A data frame containing the simulation grid, the simulated random field, predicted values and the approximated probabilities.}

\references{Allard, D., D'Or, D., Froidevaux, R. (2011) An efficient maximum entropy approach for categorical variable prediction. \emph{European Journal of Soil Science}, \bold{62}(3), 381-393.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{sim_ck}}, \code{\link{sim_ik}}, \code{\link{sim_path}}}

\examples{\donttest{
data(ACM)

# Model parameters estimation for the
# multinomial categorical simulation
x <- multi_tpfit(ACM$MAT5, ACM[, 1:3])

# Generate the simulation grid
mygrid <- list()
mygrid$X <- seq(min(ACM$X), max(ACM$X), length = 3)
mygrid$Y <- seq(min(ACM$Y), max(ACM$Y), length = 3)
mygrid$Z <- -40 * 0:9 - 1
mygrid <- as.matrix(expand.grid(mygrid$X, mygrid$Y, mygrid$Z))

# Simulate the random field
myMCSim <- sim_mcs(x, ACM$MAT5, ACM[, 1:3], mygrid)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
