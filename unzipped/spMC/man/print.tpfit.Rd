\name{print.tpfit}
\Rdversion{1.1}
\alias{print.tpfit}

\title{Printing Model Parameters for One-dimensional Continuous Lag Spatial MC}

\description{The function prints parameter estimation results given by \code{\link{tpfit}}.}

\usage{\method{print}{tpfit}(x, ...)}

\arguments{
  \item{x}{an object of the class \code{tpfit}, typically with the output of the function \code{\link{tpfit}}.}
  \item{\dots}{further arguments passed to or from other methods.}
}

\value{Estimation results are printed on the screen or other output devices. No values are returned.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{tpfit}}}

\examples{\donttest{
data(ACM)

# Estimate the parameters of a 
# one-dimensional MC model
MoPa <- tpfit(ACM$MAT5, ACM[, 1:3], c(0, 0, 1))

# Print results
print(MoPa)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
