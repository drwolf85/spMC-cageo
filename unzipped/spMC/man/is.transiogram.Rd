\name{is.transiogram}
\Rdversion{1.1}
\alias{is.transiogram}

\title{Object test for transiogram class}

\description{Function to test if an object is of the class \code{transiogram}.}

\usage{is.transiogram(object)}

\arguments{
\item{object}{object to be tested.}
}

\details{The function returns \code{TRUE} if and only if its argument is a \code{transiogram} object.}

\value{A logical value.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{transiogram}}, \code{\link{predict.tpfit}}}

\examples{\donttest{
data(ACM)

# Estimate the parameters of a 
# one-dimensional MC model
RTm <- tpfit(ACM$MAT5, ACM[, 1:3], c(0, 0, 1))

# Compute theoretical transition probabilities 
# from the one-dimensional MC model
TTPr <- predict(RTm, lags = 0:2/2)

# Compute empirical transition probabilities 
ETPr <- transiogram(ACM$MAT5, ACM[, 1:3], c(0, 0, 1), 200, 20)

# Test the objects TTPr and ETPr
is.transiogram(TTPr)
is.transiogram(ETPr)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{attribute}
