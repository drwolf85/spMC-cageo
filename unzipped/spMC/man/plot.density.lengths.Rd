\name{plot.density.lengths}
\Rdversion{1.1}
\alias{plot.density.lengths}

\title{Plot Empirical Densities Estimates of Stratum Lengths}

\description{The function plot the empirical densities of stratum lengths computed along a given direction.}

\usage{\method{plot}{density.lengths}(x, main = NULL, xlab = NULL, ylab = "Density", type = "l",
     zero.line = TRUE, ...)}

\arguments{
  \item{x}{an object of the class \code{density.lengths}, typically with the output of the function \code{\link{density.lengths}}.}
  \item{main}{an overall title for the plot.}
  \item{xlab}{a title for the \eqn{x}{x}-axis.}
  \item{ylab}{a title for the \eqn{y}{y}-axis.}
  \item{type}{plotting parameter for the type of graphic (see \code{\link{plot}}).}
  \item{zero.line}{logical value. If \code{TRUE} (by default), the function adds a base line at \eqn{y = 0}{y = 0}.}
  \item{\dots}{other plotting parameters.}
}

\value{An image is produced on the current graphics device. No values are returned.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{density.default}}, \code{\link{density.lengths}}, \code{\link{plot}}, \code{\link{print.density.lengths}}}

\examples{\donttest{
data(ACM)
direction <- c(0,0,1)
     
# Compute the appertaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction)
     
# Estimate stratum lengths
gl <- getlen(ACM$MAT3, ACM[, 1:3], loc.id, direction)

# Compute the empirical densities of stratum log-lengths
dgl <- density(gl, log = TRUE)

# Plot the empirical densities of stratum log-lengths
plot(dgl)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{hplot}