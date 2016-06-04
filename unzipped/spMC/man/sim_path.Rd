\name{sim_path}
\Rdversion{1.1}
\alias{sim_path}

\title{Conditional Simulation Based on Path Algorithms}

\description{
The function simulates a random field through the Fixed Path algorithm or Random Path technique.
}

\usage{sim_path(x, data, coords, grid, radius, fixed = FALSE)}

\arguments{
%%        x a multi_tpfit object
%%     data vector of data
%%   coords coordinates matrix
%%     grid simulation points
%%   radius radius to find neighbour points
%%    fixed boolean for random or fixed path algorithm
  \item{x}{an object of the class \code{multi_tpfit}, typically with the output of the function \code{\link{multi_tpfit}}.}
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{grid}{an \eqn{m \times d}{m x d} matrix where each row denotes the \eqn{d}{d}-D coordinates in the simulation grid.}
  \item{radius}{a numerical value that specifies a proper radius to search the nearest observed points within a \eqn{d}{d}-D sphere.}
  \item{fixed}{a logical value; if \code{TRUE}, the fixed path algorithm is performed. The random path algorithm is performed by default.}
}

\details{
These methods compute an approximation of posterior probabilities
\deqn{\Pr\left(Z(\mathbf{s}_0) = z_k \left\vert \bigcap_{i = 1}^n Z(\mathbf{s}_i) = z(\mathbf{s}_i)\right.\right).}{Pr(Z(s_0) = z_k | Z(s_i) = z(s_i)),}
\eqn{\mbox{\hspace{0cm}}}{where i = 1, ..., n.} Path algorithms are based on Pickard random fields, so that the states of such chain at any unsampled location depends on the state of its nearest known neighbours in axial directions.
}

\value{A data frame containing the simulation grid, the simulated random field, predicted values and the approximated probabilities.}

\references{Li, W. (2007) A Fixed-Path Markov Chain Algorithm for Conditional Simulation of Discrete Spatial Variables. \emph{Mathematical Geology}, \bold{39}(2), 159-176.

Li, W. (2007) Markov Chain Random Fields for Estimation of Categorical Variables. \emph{Mathematical Geology}, \bold{39}(June), 321-335.

Pickard, D. K. (1980) Unilateral Markov Fields. \emph{Advances in Applied Probability}, \bold{12}(3), 655-671.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}
%%\note{
%%%%  ~~further notes~~
%%}

\seealso{\code{\link{sim_ck}}, \code{\link{sim_ik}}, \code{\link{sim_mcs}}}

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
# the fixed path algorithm
myFixPathSim <- sim_path(x, ACM$MAT5, ACM[, 1:3], mygrid,
                         radius = 50, fixed = TRUE)

# Simulate the random field through
# the random path algorithm
myRndPathSim <- sim_path(x, ACM$MAT5, ACM[, 1:3], mygrid, radius = 50)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
