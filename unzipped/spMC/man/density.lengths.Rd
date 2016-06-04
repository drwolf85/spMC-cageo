\name{density.lengths}
\Rdversion{1.1}
\alias{density.lengths}

\title{Empirical Densities Estimation of Stratum Lengths}

\description{The function estimates the empirical conditional density of the stratum lengths given the category.}

\usage{\method{density}{lengths}(x, ..., log = FALSE, zeros.rm = TRUE)}

\arguments{
  \item{x}{an object of the class \code{lengths}, typically with the output of the function \code{\link{getlen}}.}
  \item{\dots}{other arguments to pass to the function \code{\link{density.default}}.}
  \item{log}{a logical value. If \code{TRUE}, the output density will be calculated for the logarithm of the lengths. It is \code{TRUE} by default.}
  \item{zeros.rm}{a logical value. If \code{FALSE}, the density will be estimated by including zero values. It is \code{TRUE} by default.}
}

\details{The function estimates the empirical density of the stratum lengths for each category by the use of the kernel methodology.}

\value{An object of class \code{density.lengths} is returned. It contains objects of class \code{density}, the given direction of the stratum lengths and a logical value which points out if the density is computed for the logarithm of stratum lengths.}

\references{Simonoff, J. S. (1996) \emph{Smoothing Methods in Statistics}. Springer-Verlag.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{getlen}}, \code{\link{density.default}}, \code{\link{plot.density.lengths}}, \code{\link{print.density.lengths}}}

\examples{\donttest{
data(ACM)
direction <- c(0,0,1)
     
# Compute the appertaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction)
     
# Estimate stratum lengths
gl <- getlen(ACM$MAT3, ACM[, 1:3], loc.id, direction)

# Compute the empirical densities of stratum lengths
dgl <- density(gl)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
