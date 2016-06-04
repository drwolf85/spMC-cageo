\name{hist.lengths}
\Rdversion{1.1}
\alias{hist.lengths}

\title{Histograms of Stratum Lengths for Each Observed Category}

\description{The function compute the histograms of the stratum lengths for each category. If \code{plot = TRUE}, the resulting object of class \code{hist.lengths} is plotted before it is returned.}

\usage{\method{hist}{lengths}(x, ..., log = FALSE, zeros.rm = TRUE)}

\arguments{
  \item{x}{an object of the class \code{lengths}, typically with the output of the function \code{\link{getlen}}.}
  \item{\dots}{further arguments to pass to the function \code{\link{hist}}.}
  \item{log}{a logical value. If \code{TRUE}, histograms will be calculated for the logarithm of the lengths. It is \code{FALSE} by default.}
  \item{zeros.rm}{a logical value. If \code{FALSE}, histograms will be computed by including zero values. It is \code{TRUE} by default.}
}

\value{If \code{plot = TRUE}, an image is produced on the current graphics device. The function returns an object of class \code{hist.lengths}. It contains class \code{histogram} objects, the given direction of the stratum lengths and a logical value which points out if histograms are computed for the logarithm of stratum lengths.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{getlen}}, \code{\link{hist}}, \code{\link{density.lengths}}, \code{\link{plot.density.lengths}}}

\examples{\donttest{
data(ACM)
direction <- c(0,0,1)
     
# Compute the appertaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction)
     
# Estimate stratum lengths
gl <- getlen(ACM$MAT3, ACM[, 1:3], loc.id, direction)

# Plot the histograms
hist(gl)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{hplot}
