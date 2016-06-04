\name{print.transiogram}
\Rdversion{1.1}
\alias{print.transiogram}

\title{Printing Theoretical or Empirical One-dimensional Transiograms}

\description{The function prints transition probabilities given by \code{\link{predict.multi_tpfit}} or \code{\link{transiogram}}.}

\usage{\method{print}{transiogram}(x, ...)}

\arguments{
  \item{x}{an object of the class \code{transiogram}, typically with the output of the function \code{\link{predict.tpfit}} or \code{\link{transiogram}}.}
  \item{\dots}{further arguments passed to or from other methods.}
}

\value{Transition probabilities are printed on the screen or other output devices. No values are returned.}

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

# Print results
print(TTPr)
print(ETPr)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
