\name{boxplot.lengths}
\Rdversion{1.1}
\alias{boxplot.lengths}

\title{Stratum Lengths Boxplot}

\description{Produce box-and-whisker plots of the stratum lengths.}

\usage{\method{boxplot}{lengths}(x, ..., log = FALSE, zeros.rm = TRUE)}

\arguments{
  \item{x}{an object of the class \code{lengths}, typically with the output of the function \code{\link{getlen}}.}
  \item{\dots}{other arguments to pass to the function \code{\link{boxplot}}.}
  \item{log}{a logical value. If \code{TRUE}, the logarithm of the stratum lengths will be plotted. It is \code{FALSE} by default.}
  \item{zeros.rm}{a logical value. If \code{FALSE}, the box-and-whisker will be drawn by including zero values. It is \code{TRUE} by default.}
}

\details{The box-and-whisker plots give some information about the distribution of the stratum lengths for the observed categories along a given direction.}

\value{An image is produced on the current graphics device. The function returns a list with the following components:
  \item{stats}{a matrix containing the values used to plot the box-and-whisker plots.}
  \item{n}{a vector with the number of observations for each category.}
  \item{conf}{a matrix containing further values to draw the lower and upper extremes of the notch.}
  \item{out}{a vectors with the values of the outlier points.}
  \item{group}{a vector whose elements indicate to which category the outlier belongs.}
  \item{names}{a character vector with the names of each category.}
}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{plot.lengths}}, \code{\link{boxplot}}, \code{\link{getlen}}}

\examples{\donttest{
data(ACM)
direction <- c(0,0,1)

# Compute the appertaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction)

# Estimate stratum lengths
gl <- getlen(ACM$MAT3, ACM[, 1:3], loc.id, direction)

# Make the boxplot of the object gl
\dontshow{par(mfrow = c(1, 1))}
boxplot(gl)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{hplot}

