\name{multi_tpfit_ils}
\Rdversion{1.1}
\alias{multi_tpfit_ils}

\title{Iterated Least Squares Method for Multidimensional Model Parameters Estimation}

\description{
The function estimates the model parameters of a \eqn{d}{d}-D continuous lag spatial Markov chain by the use of the iterated least squares and the bound-constrained Lagrangian methods. Transition rates matrices along axial directions and proportions of categories are computed.
}

\usage{multi_tpfit_ils(data, coords, max.dist = Inf, mpoints = 20,
                tolerance = pi/8, rotation = NULL, q = 10,
                echo = FALSE, ..., mtpfit)}

\arguments{
%%         data vector of data
%%       coords coordinates matrix
%%     max.dist maximum distance for counting (expressed for each direction)
%%      mpoints number of lags (expressed for each direction)
%%    tolerance angle for tolerance (in radians, expressed for each direction)
%%     rotation vector of rotation angles (in radians)
%%            q constant greater than one controlling the growth of rho
%%         echo logical value to print the optimization output
%%          ... further arguments to pass to nlminb function
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{max.dist}{a numerical value which defines the maximum lag value. It is \code{Inf} by default.}
  \item{mpoints}{a numerical value which defines the number of lag intervals.}
  \item{tolerance}{a numerical value for the tolerance angle (in radians). It is \code{pi/8} by default.}
  \item{rotation}{a numerical vector of length \eqn{d - 1}{d - 1} with rotation angles (in radians), in order to perform the main axes rotation. No rotation is performed by default.}
  \item{q}{a numerical value greater than one for a constant which controls the growth of the penalization term in the loss function. It is equal to \code{10} by default.}
  \item{echo}{a logical value; if \code{TRUE}, the function prints some information about the optimization. It is \code{FALSE} by default.}
  \item{\dots}{other arguments to pass to the function \code{\link{nlminb}}.}
  \item{mtpfit}{an object \code{multi_tpfit} to optimize. If missing, the algorithm starts with null transition rates matrices.}
}

\details{
A \eqn{d}{d}-D continuous-lag spatial Markov chain is probabilistic model which is developed by interpolation of the transition rate matrices computed for the main directions. It defines transition probabilities \eqn{\Pr(Z(s + h) = z_k | Z(s) = z_j)}{Pr(Z(s + h) = z_k | Z(s) = z_j)} through
\deqn{\mbox{expm} (\Vert h \Vert R),}{expm(||h|| R),}
where \eqn{h}{h} is the lag vector and the entries of \eqn{R}{R} are ellipsoidally interpolated.

The ellipsoidal interpolation is given by
\deqn{\vert r_{jk} \vert = \sqrt{\sum_{i = 1}^d \left( \frac{h_i}{\Vert h \Vert} r_{jk, \mathbf{e}_i} \right)^2},}{|r_(jk, h)| = sqrt(sum_{i = 1}^d h_i / ||h|| r_(jk, e_i)),}
where \eqn{\mathbf{e}_i}{e_i} is a standard basis for a \eqn{d}{d}-D space.

If \eqn{h_i < 0}{h_i < 0} the respective entries \eqn{r_{jk, \mathbf{e}_i}}{r_(jk, e_i)} are replaced by \eqn{r_{jk, -\mathbf{e}_i}}{r_(jk, -e_i)}, which is computed as
\deqn{r_{jk, -\mathbf{e}_i} = \frac{p_k}{p_j} \, r_{kj, \mathbf{e}_i},}{r_(jk, -e_i) = p_k / p_j * r_(kj, e_i),}
where \eqn{p_k}{p_k} and \eqn{p_j}{p_j} respectively denote the proportions for the \eqn{k}{k}-th and \eqn{j}{j}-th categories. In so doing, the model may describe the anisotropy of the process.

In particular, to estimate entries of transition rate matrices computed for the main axial directions, we need to minimize the discrepancies between the empirical transiograms (see \code{\link{transiogram}}) and the predicted transition probabilities.

By the use of the iterated least squares, the diagonal entries of \eqn{R}{R} are constrained to be negative, while the off-diagonal transition rates are constrained to be positive. Further constraints are considered in order to obtain a proper transition rates matrix.
}

\value{
An object of the class \code{multi_tpfit} is returned. The function \code{\link{print.multi_tpfit}} is used to print the fitted model. The object is a list with the following components: 
  \item{coordsnames}{a character vector containing the name of each axis.}
  \item{coefficients}{a list containing the transition rates matrices computed for each axial direction.}
  \item{prop}{a vector containing the proportions of each observed category.}
  \item{tolerance}{a numerical value which denotes the tolerance angle (in radians).}
}

\references{Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\section{Warning}{If the process is not stationary, the optimization algorithm does not converge.}

\seealso{\code{\link{predict.multi_tpfit}}, \code{\link{print.multi_tpfit}}, \code{\link{image.multi_tpfit}}, \code{\link{tpfit_ils}}, \code{\link{transiogram}}}

\examples{\donttest{
data(ACM)

# Estimate the parameters of a 
# multidimensional MC model
multi_tpfit_ils(ACM$MAT3, ACM[, 1:3], 100)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{models}
