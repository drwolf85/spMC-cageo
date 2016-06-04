\name{is.pemt}
\Rdversion{1.1}
\alias{is.pemt}

\title{Images with Multi-direct`ional Transiograms}

\description{
The function plots \eqn{2}{2}-D sections of a multi-directional transiogram computed without any ellipsoidal interpolation.}

\usage{is.pemt(object)}

\arguments{
  \item{object}{object to be tested.}
}

\details{The function returns \code{TRUE} if and only if its argument is a \code{pemt} object.}

\value{A logical value.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{pemt}}, \code{\link{image.pemt}}}

\examples{\donttest{
data(ACM)

# Compute a 2-D section of a
# multi-directional transiogram
psEmpTr <- pemt(ACM$MAT3, ACM[, 1:3], 2,
                max.dist = c(20, 10, 5), 
                which.dire=c(1, 3), 
                mle = TRUE)

# Test the object psEmpTr
is.pemt(psEmpTr)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{attribute}
