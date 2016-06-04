\name{print.lengths}
\Rdversion{1.1}
\alias{print.lengths}

\title{Printing Stratum Lengths for Each Observed Category}

\description{The function prints stratum lengths given by \code{\link{getlen}}.}

\usage{\method{print}{lengths}(x, ...)}

\arguments{
  \item{x}{an object of the class \code{lengths}, typically with the output of the function \code{\link{getlen}}.}
  \item{\dots}{further arguments passed to or from other methods.}
}

\value{Stratum lengths grouped by category are printed on the screen or other output devices. No values are returned.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{getlen}}}

\examples{\donttest{
data(ACM)
direction <- c(0,0,1)
     
# Compute the appertaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction)
     
# Estimate stratum lengths
gl <- getlen(ACM$MAT3, ACM[, 1:3], loc.id, direction)

# Print stratum lengths
print(gl)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
