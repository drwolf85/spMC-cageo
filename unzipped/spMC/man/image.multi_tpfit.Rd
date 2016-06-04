\name{image.multi_tpfit}
\Rdversion{1.1}
\alias{image.multi_tpfit}

\title{Images with Multidimensional Transiograms}

\description{The function plots \eqn{2}{2}-D sections of a predicted multidimensional transiograms computed through ellipsoidal interpolation.}

\usage{\method{image}{multi_tpfit}(x, mpoints, which.dire, max.dist, main,
      mar, ask = TRUE, ..., nlevels = 10, contour = TRUE)}

\arguments{
%%            x multi_tpfit object
%%      mpoints number of points per axes
%%   which.dire two chosen 1D directions
%%     max.dist vector of maximum distances
%%         main title string
%%          mar vector to pass to par()
%%          ask boolean to pass to par()
%%          ... other args to pass to image()
%%      nlevels number of levels to pass to contour()
%%      contour boolean values that permits to draw contour lines
  \item{x}{an object of the class \code{multi_tpfit}, typically with the output of the function \code{\link{multi_tpfit}}.}
  \item{mpoints}{the number of points per axes. It controls the accuracy of images to plot.}
  \item{which.dire}{a vector with two chosen axial directions. If omitted, all \eqn{2}{2}-D sections are plotted.}
  \item{max.dist}{a scalar or a vector of maximum length for the chosen axial directions.}
  \item{main}{the main title (on top) whose font and size are fixed.}
  \item{mar}{a scalar or a numerical vector of the form \code{c(bottom, left, top, right)} which gives the number of margin lines to be specified on the four sides of image to plot. See \code{\link{par}}(\code{mar=}.).}
  \item{ask}{a logical value; if \code{TRUE}, the user is asked for input, before each plot. See \code{\link{par}}(\code{ask=}.).}
  \item{\dots}{other arguments to pass to the function \code{\link{image}}.}
  \item{nlevels}{the number of levels to pass to the function \code{\link{contour}}.}
  \item{contour}{logical. If \code{TRUE}, the function \code{\link{contour}} is used to draw contour lines over the image. Defaults to \code{TRUE}.}
}

\details{
A multidimensional transiogram is a diagram which shows the transition probabilities for a single pair of categories. It is computed for any lag vector \eqn{h}{h} through 
\deqn{\mbox{expm} (\Vert h \Vert R),}{expm(||h|| R),}
where entries of \eqn{R}{R} are ellipsoidally interpolated (see \code{\link{multi_tpfit}}).

The exponential matrix is evaluated by the scaling and squaring algorithm.
}

\value{An image is produced on the current graphics device. No values are returned.}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Higham, N. J. (2008) \emph{Functions of Matrices: Theory and Computation}. Society for Industrial and Applied Mathematics.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}


\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{multi_tpfit}}, \code{\link{pemt}}, \code{\link{image.pemt}}, \code{\link{image}}, \code{\link{plot.transiogram}}}

\examples{\donttest{
data(ACM)

# Estimate model parameter
x <- multi_tpfit(ACM$MAT5, ACM[, 1:3])

# Set short names for categories 3 and 4
names(x$prop)[3:4] <- c("Clay and Sand", "Gravel and Sand")

# Plot 2-D theoretical sections of
# a multidimensional transiogram
image(x, 40, max.dist=c(200,200,20), which.dire=2:3,
    mar = .7, col=rev(heat.colors(500)),
    breaks=0:500/500, nlevels = 5)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{hplot}
