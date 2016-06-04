\name{print.summary.lengths}
\Rdversion{1.1}
\alias{print.summary.lengths}

\title{Printing Stratum Lengths Summary for Each Observed Category}

\description{The function prints the summary of stratum lengths given by \code{\link{summary.lengths}}.}

\usage{\method{print}{summary.lengths}(x, ...)}

\arguments{
  \item{x}{an object of the class \code{summary.lengths}, typically with the output of the function \code{\link{summary.lengths}}.}
  \item{\dots}{further arguments passed to or from other methods.}
}

\value{The summary of stratum lengths grouped by category is printed on the screen or other output devices. No values are returned.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{getlen}}, \code{\link{summary.lengths}}}

\examples{\donttest{
data(ACM)
direction <- c(0,0,1)
     
# Compute the appartaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction)
     
# Estimate stratum lengths
gl <- getlen(ACM$MAT3, ACM[, 1:3], loc.id, direction)

# Summarize the stratum lengths
sgl <- summary(gl)

# Print the summary of stratum lengths
print(sgl)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}