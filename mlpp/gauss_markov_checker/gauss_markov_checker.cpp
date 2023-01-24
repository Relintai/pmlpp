//
//  GaussMarkovChecker.cpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "gauss_markov_checker.h"
#include "../stat/stat.h"
#include <iostream>


void MLPPGaussMarkovChecker::checkGMConditions(std::vector<double> eps) {
	bool condition1 = arithmeticMean(eps);
	bool condition2 = homoscedasticity(eps);
	bool condition3 = exogeneity(eps);

	if (condition1 && condition2 && condition3) {
		std::cout << "Gauss-Markov conditions were not violated. You may use OLS to obtain a BLUE estimator" << std::endl;
	} else {
		std::cout << "A test of the expected value of 0 of the error terms returned " << std::boolalpha << condition1 << ", a test of homoscedasticity has returned " << std::boolalpha << condition2 << ", and a test of exogenity has returned " << std::boolalpha << "." << std::endl;
	}
}

bool MLPPGaussMarkovChecker::arithmeticMean(std::vector<double> eps) {
	Stat stat;
	if (stat.mean(eps) == 0) {
		return 1;
	} else {
		return 0;
	}
}

bool MLPPGaussMarkovChecker::homoscedasticity(std::vector<double> eps) {
	Stat stat;
	double currentVar = (eps[0] - stat.mean(eps)) * (eps[0] - stat.mean(eps)) / eps.size();
	for (int i = 0; i < eps.size(); i++) {
		if (currentVar != (eps[i] - stat.mean(eps)) * (eps[i] - stat.mean(eps)) / eps.size()) {
			return 0;
		}
	}
	return 1;
}

bool MLPPGaussMarkovChecker::exogeneity(std::vector<double> eps) {
	Stat stat;
	for (int i = 0; i < eps.size(); i++) {
		for (int j = 0; j < eps.size(); j++) {
			if (i != j) {
				if ((eps[i] - stat.mean(eps)) * (eps[j] - stat.mean(eps)) / eps.size() != 0) {
					return 0;
				}
			}
		}
	}
	return 1;
}

