\name{print.multi_tpfit}
\Rdversion{1.1}
\alias{print.multi_tpfit}

\title{Printing Model Parameters for Multidimensional Continuous Lag Spatial MC}

\description{The function prints parameter estimation results given by \code{\link{multi_tpfit}}.}

\usage{\method{print}{multi_tpfit}(x, ...)}

\arguments{
  \item{x}{an object of the class \code{multi_tpfit}, typically with the output of the function \code{\link{multi_tpfit}}.}
  \item{\dots}{further arguments passed to or from other methods.}
}

\value{Estimation results are printed on the screen or other output devices. No values are returned.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{multi_tpfit}}}

\examples{\donttest{
data(ACM)

# Estimate the parameters of a 
# multidimensional MC models
MoPa <- multi_tpfit(ACM$MAT5, ACM[, 1:3])

# Print results
print(MoPa)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
