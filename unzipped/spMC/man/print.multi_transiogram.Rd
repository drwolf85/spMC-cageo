\name{print.multi_transiogram}
\Rdversion{1.1}
\alias{print.multi_transiogram}

\title{Printing Theoretical Multidimensional Transiograms}

\description{
The function prints theoretical transition probabilities given by \code{\link{predict.multi_tpfit}}.
}
\usage{\method{print}{multi_transiogram}(x, ...)}

\arguments{
  \item{x}{an object of the class \code{multi_transiogram}, typically with the output of the function \code{\link{predict.multi_tpfit}}.}
  \item{\dots}{further arguments passed to or from other methods.}
}

\value{Transition probabilities are printed on the screen or other output devices. No values are returned.}

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

# Print results
print(TrPr)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
