\name{contour.pemt}
\Rdversion{1.1}
\alias{contour.pemt}

\title{Display Contours with Multi-directional Transiograms}

\description{
The function draws the \eqn{2}{2}-D sections contour plots of a multi-directional transiogram computed without any ellipsoidal interpolation and superpose the contour lines of the theoretical transition probabilities.}

\usage{
\method{contour}{pemt}(x, nlevels = 10, col = c("black", "blue"), main,
     mar, ask = TRUE, ...)
}

\arguments{
%%            x object return by the function pemt()
%%      nlevels number of levels to pass to contour()
%%          col vector of colors for two transiograms
%%         main title string
%%          mar vector to pass to par()  
%%          ask boolean to pass to par()
%%          ... other args to pass to contour()
  \item{x}{an object of class \code{pemt}.}
  \item{nlevels}{the number of levels to pass to the function \code{\link{contour}}.}
  \item{col}{a vector of two colors to pass to the function \code{\link{contour}}. The former color refers to the multi-directional transiogram, while the latter is used to draw contour lines of the theoretical transition probabilities.}
  \item{main}{the main title (on top) whose font and size are fixed.}
  \item{mar}{a scalar or a numerical vector of the form \code{c(bottom, left, top, right)} which gives the number of margin lines to be specified on the four sides of image to plot. See \code{\link{par}}(\code{mar=}.).}
  \item{ask}{a logical value; if \code{TRUE}, the user is asked for input, before each plot. See \code{\link{par}}(ask=.).}
  \item{\dots}{other arguments to pass to the function \code{\link{contour}}.}
}

\details{
A multidimensional transiogram is a diagram which shows the transition probabilities for a single pair of categories. The probability is computed for any lag vector \eqn{h}{h} through 
\deqn{\mbox{expm} (\Vert h \Vert R_h),}{expm(||h|| R),}
where entries of \eqn{R_h}{R} are not ellipsoidally interpolated, but they are estimated for the direction specified by the vector \eqn{h}{h}.

The exponential matrix is evaluated by the scaling and squaring algorithm.
}

\value{An image is produced on the current graphics device. No values are returned.}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Higham, N. J. (2008) \emph{Functions of Matrices: Theory and Computation}. Society for Industrial and Applied Mathematics.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{image.pemt}}, \code{\link{contour}}, \code{\link{plot.transiogram}}}

\examples{\donttest{
data(ACM)

# Compute a 2-D section of a 
# multi-directional transiogram
psEmpTr <- pemt(ACM$MAT3, ACM[, 1:3], 2,
                max.dist = c(200, 200, 20), 
                which.dire=c(1, 3), 
                mle = "avg")

# Contour plots of 2-D sections of 
# multi-directional transiograms
contour(psEmpTr, mar = .7)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{hplot}
