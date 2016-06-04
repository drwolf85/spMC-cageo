\name{is.multi_transiogram}
\Rdversion{1.1}
\alias{is.multi_transiogram}

\title{Object test for multi_transiogram class}

\description{Function to test if an object is of the class \code{multi_transiogram}.}

\usage{is.multi_transiogram(object)}

\arguments{
\item{object}{object to be tested.}
}

\details{The function returns \code{TRUE} if and only if its argument is a \code{multi_transiogram} object.}

\value{A logical value.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{predict.multi_tpfit}}}

\examples{\donttest{
data(ACM)

# Estimate the parameters of a 
# multidimensional MC model
RTm <- multi_tpfit(ACM$MAT3, ACM[, 1:3])

# Generate the matrix of 
# multidimensional lags
lags <- expand.grid(X=-1:1, Y=-1:1, Z=-1:1)
lags <- as.matrix(lags)

# Compute transition probabilities 
# from the multidimensional MC model
TrPr <- predict(RTm, lags)

# Test the object TrPr
is.multi_transiogram(TrPr)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{attribute}
