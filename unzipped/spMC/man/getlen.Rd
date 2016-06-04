\name{getlen}
\Rdversion{1.1}
\alias{getlen}

\title{Estimation of Stratum Lengths for Embedded Markov Chain}

\description{The function estimates the stratum lengths for a \eqn{d}{d}-D spatial embedded Markov chain for a specified direction \eqn{\phi}{phi}.}

\usage{getlen(data, coords, loc.id, direction, zero.allowed = FALSE)}

\arguments{
%%         data vector of data
%%       coords matrix of coordinates
%%       loc.id location Id (which_lines output)
%%    direction vector (or versor) of choosen direction
%% zero.allowed logical, it allows zero stratum lengths
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{loc.id}{a vector of \eqn{n}{n} values which indicates the directional line of each location. It is usually the output of the function \code{\link{which_lines}}.}
  \item{direction}{a \eqn{d}{d}-D numerical vector (or versor) which represents the chosen direction.}
  \item{zero.allowed}{a logical value which allows to return zero stratum lengths. It is \code{FALSE} by default.}
}

\details{
Stratum lengths are the lengths occupied by the same \eqn{k}{k}-th category along lines in the direction \eqn{\phi}{phi}.
}

\value{A list containing the following components:
  \item{length}{a numerical vector with the stratum lengths along the given direction.}
  \item{categories}{a vector with the stratum categories.}
  \item{maxcens}{a vector with the maxima estimated censored lengths for each stratum.}
  \item{directions}{a \eqn{d}{d}-D numerical vector which represents the chosen direction.}
  \item{zeros}{a logical values which denotes the possible presence of zero lengths.}
}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{mlen}}, \code{\link{which_lines}}}

\examples{\donttest{
data(ACM)
direction <- c(0,0,1)

# Compute the appertaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction)

# Estimate stratum lengths
gl <- getlen(ACM$MAT5, ACM[, 1:3], loc.id, direction)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
