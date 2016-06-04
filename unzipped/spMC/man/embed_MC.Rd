\name{embed_MC}
\Rdversion{1.1}
\alias{embed_MC}

\title{Transition Probabilities Estimation for Embedded Markov Chain}

\description{The function estimates the embedded transition probabilities matrix for a \eqn{1}{1}-D spatial embedded Markov chain.}

\usage{embed_MC(data, coords, loc.id, direction)}

\arguments{
%%        data vector of data
%%      coords matrix of coordinates
%%      loc.id location Id (which_lines output)
%%   direction vector (or versor) of chosen direction
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{loc.id}{a vector of \eqn{n}{n} values which indicats the directional line of each location. It is usually the output of the function \code{\link{which_lines}}.}
  \item{direction}{a \eqn{d}{d}-D numerical vector (or versor) which represents the chosen direction.}
}

\details{
An embedded Markov chain is probabilistic model which defines the transition probabilities between embedded occurrences. 

The resulting matrix is given by normalizing a transition count matrix, which doesn't depend on the length of embedded occurrences. Self-transitions of embedded occurrences are not observable, so diagonal entries are set to be \code{NA}.

It's also possible to calculate the transition probabilities matrix for several directions in a \eqn{d}{d}-D space through arguments \code{direction} and \code{loc.id}. If the user has no previous knowledge about \code{loc.id}, the function \code{\link{which_lines}} provides a method to compute the right values.
}

\value{A \eqn{K \times K}{K x K} matrix, where \eqn{K}{K} denotes the number of observed categories.}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Dynkin, E. B. (1961) \emph{Theory of Markov Processes. Englewood Cliffs}, N.J.: Prentice-Hall, Inc.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{which_lines}}, \code{\link{predict.tpfit}}, \code{\link{predict.multi_tpfit}}}

\examples{\donttest{
data(ACM)
direction <- c(0, 0, 1)

# Compute the appertaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction, pi/8)

# Estimate the embedded transition probabilities
# matrix for the categorical variable MAT5
embed_MC(ACM$MAT5, ACM[, 1:3], loc.id, direction)

# Estimate the embedded transition probabilities
# matrix for the categorical variable MAT3
embed_MC(ACM$MAT3, ACM[, 1:3], loc.id, direction)

# Estimate the embedded transition probabilities
# matrix for the categorical variable PERM
embed_MC(ACM$PERM, ACM[, 1:3], loc.id, direction)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{models}
