\name{which_lines}
\Rdversion{1.1}
\alias{which_lines}

\title{Points Classification through Directional Lines}

\description{The function classifies points which appertain to a same directional line.}

\usage{which_lines(coords, direction, tolerance = pi / 8)}

\arguments{
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{direction}{a \eqn{d}{d}-D numerical vector (or versor) which represents the chosen direction.}
  \item{tolerance}{a numerical value for the tolerance angle (in radians). It's \code{pi/8} by default.}
}

\details{
The algorithm used by this function searches the nearest points to a directional line. The function classifies such pairs of points that have the minimum distance and the same direction of the vector \eqn{\phi}{phi}.

This operation is done to order points, so that it's possible to compute mean lengths (\link{mlen}) and embedded transition probabilities (\link{embed_MC}).
}

\value{A numerical vector containing the line number for each point.}

\references{Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{embed_MC}}, \code{\link{mlen}}, \code{\link{getlen}}}

\examples{\donttest{
data(ACM)

direction <- c(0,0,1)

loc.id <- which_lines(ACM[, 1:3], direction)}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{classif}
