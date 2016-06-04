\name{plot.hist.lengths}
\Rdversion{1.1}
\alias{plot.hist.lengths}

\title{Plot Histograms of Stratum Lengths}

\description{The function plots objects of class \code{hist.lengths}.}

\usage{\method{plot}{hist.lengths}(x, ...)}

\arguments{
  \item{x}{an object of the class \code{hist.lengths}, typically with the output of the function \code{\link{hist.lengths}}.}
  \item{\dots}{further plotting parameters.}
}

\value{An image is produced on the current graphics device. No values are returned.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{hist}}, \code{\link{hist.lengths}}, \code{\link{plot}}, \code{\link{print.density.lengths}}}

\examples{\donttest{
data(ACM)
direction <- c(0,0,1)
     
# Compute the appertaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction)
     
# Estimate stratum lengths
gl <- getlen(ACM$MAT3, ACM[, 1:3], loc.id, direction)

# Compute the histograms
hgl <- hist(gl, plot = FALSE)

# Plot the histograms
plot(hgl, col = "#efffef")
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{hplot}
