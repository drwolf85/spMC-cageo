\name{mlen}
\Rdversion{1.1}
\alias{mlen}

\title{Mean Length Estimation for Embedded Markov Chain}

\description{The function estimates the mean length for a \eqn{d}{d}-D spatial embedded Markov chain for a specified direction \eqn{\phi}{phi}.}

\usage{mlen(data, coords, loc.id, direction, mle = "avg")}

\arguments{
%%        data vector of data
%%      coords matrix of coordinates
%%      loc.id location Id (which_lines output)
%%   direction vector (or versor) of chosen direction
%%         mle string of characters, 
%%             if "mle" the MLEs will be returned (log-normal distro assumed)
%%             it can be logical for backward compatibility reasons

  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{loc.id}{a vector of \eqn{n}{n} values which indicates the directional line of each location. It is usually the output of the function \code{\link{which_lines}}.}
  \item{direction}{a \eqn{d}{d}-D numerical vector (or versor) which represents the chosen direction.}
  \item{mle}{a character value. If \code{"trm"}, the trimmed arithmetic average will be used to calculate the mean lengths. If \code{"mdn"}, the trimmed median will be considered. If \code{"mlk"}, the maximum likelihood mean lengths will be computed. If \code{"avg"}, the arithmetic mean will be performed. For backward compatibility reasons, it can accept logical values, so that \code{TRUE} is equivalent to \code{"mlk"} and \code{FALSE} to \code{"avg"}.}
}

\details{
The mean length is the total length occupied by the \eqn{k}{k}-th category divided by the number of its embedded occurrences along lines in the direction \eqn{\phi}{phi}. More robust methods are implemented, such as the trimmed mean and the trimmed median.

If the stratum lengths are censored, the maximum likelihood approach is more appropriate than the arithmetic mean. In this case, the stratum lengths are assumed to be independent realizations from a log-normal random variable. The quantity to maximize is
\deqn{L(\mu_1, \ldots, \mu_K, \sigma_1, \ldots, \sigma_K) = \prod_{i = 1}^m \prod_{k = 1}^K \left[ \int_{l_i}^{l_i+u_i} \frac{1}{x \sigma_k \sqrt{2}} \exp \left\lbrace - \frac{(\log x - \mu_k)^2}{2 \sigma_k^2} \right\rbrace \right]^{z_{k, i}} \mbox{d}x,}{L(mu, sigma) = prod_(i=1)^m prod_(k=1)^K Pr(l_i < X < l_i + u_i)^(z_k),}
where \eqn{\boldsymbol{\mu} = (\mu_1, \ldots, \mu_K)^\top}{mu} and \eqn{\boldsymbol{\sigma} = (\sigma_1, \ldots, \sigma_K)^\top}{sigma} are vectors of parameters, \eqn{l_i}{l_i} is the observed stratum length, \eqn{u_i}{u_i} denotes the upper bound of the censor and \eqn{z_{k, i}}{z_k} denotes a dummy variable which assumes value 1 if and only if the \eqn{i}{i}-th stratum is referred to the \eqn{k}{k}-th category.
}

\value{A numeric vector containing the mean length for each observed category.}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{which_lines}}}

\examples{\donttest{
data(ACM)
direction <- c(0,0,1)

# Compute the appartaining directional line for each location
loc.id <- which_lines(ACM[, 1:3], direction)

# Estimate the mean lengths for each observed category
ml <- mlen(ACM$MAT5, ACM[, 1:3], loc.id, direction, mle = "avg")

# Equivalently
gl <- getlen(ACM$MAT5, ACM[, 1:3], loc.id, direction, zero.allowed = TRUE)
ml1 <- tapply(gl$length, gl$categories, mean)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
