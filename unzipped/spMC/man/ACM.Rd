\name{ACM}
\Rdversion{1.1}
\alias{ACM}
\docType{data}

\title{ACM Data}

\description{
The data set refers to a sampled area which is located in the province of Venice. Its sample units report the geographical position of the perforation, the depth, the ground permeability and other two categorical variables which denote the soil composition.
}

\usage{data(ACM)}

\format{
  A data frame with 2321 observations on the following 6 variables.
  \describe{
    \item{\code{X}}{a numeric vector (longitude)}
    \item{\code{Y}}{a numeric vector (latitude)}
    \item{\code{Z}}{a numeric vector (depth)}
    \item{\code{MAT5}}{a factor with levels \code{Clay}, \code{Gravel}, \code{Mix of Sand and Clay}, \code{Mix of Sand and Gravel} and \code{Sand}}
    \item{\code{MAT3}}{a factor with levels \code{Clay}, \code{Gravel} and \code{Sand}}
    \item{\code{PERM}}{a logical vector (symmetric dichotomous variable)}
  }
}

\source{Fabbri, P. (2010) Professor at the Geosciences Department of the University of Padua.\cr\email{paolo.fabbri@unipd.it}}

\references{Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\examples{\donttest{
data(ACM)
str(ACM)
summary(ACM)
}}

\keyword{datasets}
