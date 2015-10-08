spMC: an R-package for 3D lithological reconstructions based on spatial Markov chains
=====================================================================================

[Luca Sartore](mailto://luca.sartore@unive.it)(1), [Paolo Fabbri](paolo.fabbri@unipd.it)(2), [Carlo Gaetan](mailto://gaetan@unive.it)(1)

  1. Dipartimento di Scienze Ambientali, Informatica e Statistica,
     Università "Ca' Foscari" di Venezia,
     Campus Scientifico, Via Torino 155,
     I-30172 Mestre-Venezia, Italy
  2. Dipartimento di Geoscienze,
     Università di Padova,
     via Gradenigo 6,
     35131 Padova, Italy

Corresponding author: [Luca Sartore, luca.sartore@unive.it](mailto://luca.sartore@unive.it)

Abstract
--------

The paper presents the  spatial Markov Chains (spMC) R-package and a case study of subsoil prediction/simulation in a plain site of the NE Italy. 
spMC is a quite complete collection of advanced methods for data inspection, besides spMC implements Markov Chain models to estimate experimental transition probabilities of categorical lithological data. 
Furthermore, in spMC package the most known estimation/simulation methods as indicator Kriging and CoKriging were implemented, but also most advanced methods such as path methods and Bayesian procedure exploiting the maximum entropy. 
Because the spMC package was thought for intensive geostatistical computations, part of the code is implemented with parallel computing via the OpenMP constructs, allowing to deal with more than five lithologies, but trying to keep a computational efficiency. 
A final analysis of this computational efficiency of spMC compares the prediction/simulation results using different numbers of CPU cores, considering the example data set of the case study available in the package.

Keywords: Categorical data, Transition probabilities, Transiogram modeling, Indicator Cokriging, Bayesian entropy, 3D lithological conditional simulation/prediction

Citation
--------

Sartore, L., Fabbri, P. and Gaetan, C. (submitted). spMC: an R-package for 3D lithological reconstructions based on spatial Markov chains

Instruction
-----------

Extract the file `spMC.zip` and read other details in the extracted file `README.html`.

License
-------

The package is distributed on CRAN at [this web page](https://cran.r-project.org/web/packages/spMC/index.html) under the terms of the [GENERAL PUBLIC LICENSE (>= 2)](https://cran.r-project.org/web/licenses/GPL-2). The package spMC version 0.3.6.3153 was edited in order to satisfy the requirement for the manuscript submission to *Computer and Geosciences* (Elsevier). This specific version will never appear on the CRAN servers, but it can be available on the Elsevier servers with the same terms and conditions specified by the [GENERAL PUBLIC LICENSE (>= 2)](https://cran.r-project.org/web/licenses/GPL-2).
