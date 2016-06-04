\name{setCores}
\Rdversion{1.1}
\alias{setCores}

\title{Set the number of CPU cores for HPC}

\description{The function set the number of CPU cores for parallel computation by the use of OpenMP library (\url{http://openmp.org/}). If the package was not complied with the library OpenMP (>= 3.0), this function is disabled.}

\usage{setCores(n)}

\arguments{
  \item{n}{an integer value denoting the number of CPU cores to use; if it exceeds the total number of cores, all of them will be used. If missing, the number of CPU cores in use will be displayed.}
}

\details{
When the package is loaded, only one CPU core is used.
}

\value{The total number of CPU cores in use will be returned and a message will be displayed. If the package was not complied with the library OpenMP (>= 3.0), the value one will be returned.}

\references{SunTM ONE Studio 8 (2003) \emph{OpenMP API User's Guide}. Sun Microsystems Inc., Santa Clara, U.S.A.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\examples{\donttest{
#Display the number of CPU cores in use
setCores()

#Set 2 CPU cores for parallel computation
setCores(2)

#Set 1 CPU core for serial computation
setCores(1)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{programming}
