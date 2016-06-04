\name{print.density.lengths}
\Rdversion{1.1}
\alias{print.density.lengths}

\title{Printing Empirical Densities Estimates of Stratum Lengths}

\description{he function a summary of the empirical density stratum lengths calculated by \code{\link{density.lengths}}.}

\usage{\method{print}{density.lengths}(x, digits = NULL, ...)}

\arguments{
  \item{x}{an object of the class \code{density.lengths}, typically with the output of the function \code{\link{density.lengths}}.}
  \item{digits}{minimal number of digits, see \code{\link{print.default}}.}
  \item{\dots}{further arguments to pass to the function \code{\link{summary.data.frame}}.}
}

\value{A summary of the empirical distributions is printed on the screen or other output devices. No values are returned.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{density.lengths}}, \code{\link{plot.density.lengths}}}

\examples{\donttest{
data(ACM)
direction <- c(0,0,1)
     
# Compute the appartaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction)
     
# Estimate stratum lengths
gl <- getlen(ACM$MAT3, ACM[, 1:3], loc.id, direction)

# Compute the empirical densities of stratum lengths
dgl <- density(gl)

# Print the empirical densities of stratum lengths
print(dgl)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
