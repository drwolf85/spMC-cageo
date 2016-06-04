\name{summary.lengths}
\Rdversion{1.1}
\alias{summary.lengths}

\title{Summarizing Stratum Lengths}

\description{The function summarizes the stratum lengths for each observed category.}

\usage{\method{summary}{lengths}(object, ..., zeros.rm = TRUE)}

\arguments{
  \item{object}{an object of the class \code{lengths}, typically with the output of the function \code{\link{getlen}}.}
  \item{\dots}{further arguments passed to or from other methods.}
  \item{zeros.rm}{a logical values. If \code{FALSE}, summarizing statistics will be computed by including zero values. It is \code{TRUE} by default.}
}

\value{
An object of class \code{summary.lengths} containing the minimum, the first quartile, the median, the mean, the third quartile and the maximum of the stratum lengths for each observed category.
}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{getlen}}}

\examples{\donttest{
data(ACM)
direction <- c(0,0,1)
     
# Compute the appertaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction)
     
# Estimate stratum lengths
gl <- getlen(ACM$MAT3, ACM[, 1:3], loc.id, direction)

# Summarize the stratum lengths
sgl <- summary(gl)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
