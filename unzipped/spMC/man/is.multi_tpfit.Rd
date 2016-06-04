\name{is.multi_tpfit}
\Rdversion{1.1}
\alias{is.multi_tpfit}

\title{Object test for multi_tpfit class}

\description{Function to test if an object is of the class \code{multi_tpfit}.}

\usage{is.multi_tpfit(object)}

\arguments{
\item{object}{object to be tested.}
}

\details{The function returns \code{TRUE} if and only if its argument is a \code{multi_tpfit} object.}

\value{A logical value.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{multi_tpfit}}}

\examples{\donttest{
data(ACM)

# Estimate the parameters of a 
# multidimensional MC models
MoPa <- multi_tpfit(ACM$MAT5, ACM[, 1:3])

# Test the object MoPa
is.multi_tpfit(MoPa)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{attribute}
