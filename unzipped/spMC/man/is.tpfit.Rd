\name{is.tpfit}
\Rdversion{1.1}
\alias{is.tpfit}

\title{Object test for tpfit class}

\description{Function to test if an object is of the class \code{tpfit}.}

\usage{is.tpfit(object)}

\arguments{
\item{object}{object to be tested.}
}

\details{The function returns \code{TRUE} if and only if its argument is a \code{tpfit} object.}

\value{A logical value.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{tpfit}}}

\examples{\donttest{
data(ACM)

# Estimate the parameters of a 
# one-dimensional MC model
MoPa <- tpfit(ACM$MAT5, ACM[, 1:3], c(0, 0, 1))

# Test the object MoPa
is.tpfit(MoPa)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{attribute}
