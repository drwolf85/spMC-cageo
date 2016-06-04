\name{is.lengths}
\Rdversion{1.1}
\alias{is.lengths}

\title{Object test for lengths class}

\description{Function to test if an object is of the class \code{lengths}.}

\usage{is.lengths(object)}

\arguments{
  \item{object}{object to be tested.}
}

\details{The function returns \code{TRUE} if and only if its argument is a \code{lengths} object.}

\value{A logical value.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{getlen}}}

\examples{\donttest{
data(ACM)
direction <- c(0,0,1)
     
# Compute the appertaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction)
     
# Estimate stratum lengths
gl <- getlen(ACM$MAT3, ACM[, 1:3], loc.id, direction)

# Test the object gl
is.lengths(gl)

}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{attribute}
