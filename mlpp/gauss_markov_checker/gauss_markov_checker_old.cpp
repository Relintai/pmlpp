//
//  GaussMarkovChecker.cpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "gauss_markov_checker_old.h"
#include "../stat/stat_old.h"
#include <iostream>

void MLPPGaussMarkovCheckerOld::checkGMConditions(std::vector<real_t> eps) {
	bool condition1 = arithmeticMean(eps);
	bool condition2 = homoscedasticity(eps);
	bool condition3 = exogeneity(eps);

	if (condition1 && condition2 && condition3) {
		std::cout << "Gauss-Markov conditions were not violated. You may use OLS to obtain a BLUE estimator" << std::endl;
	} else {
		std::cout << "A test of the expected value of 0 of the error terms returned " << std::boolalpha << condition1 << ", a test of homoscedasticity has returned " << std::boolalpha << condition2 << ", and a test of exogenity has returned " << std::boolalpha << "." << std::endl;
	}
}

bool MLPPGaussMarkovCheckerOld::arithmeticMean(std::vector<real_t> eps) {
	MLPPStatOld stat;
	if (stat.mean(eps) == 0) {
		return true;
	} else {
		return false;
	}
}

bool MLPPGaussMarkovCheckerOld::homoscedasticity(std::vector<real_t> eps) {
	MLPPStatOld stat;
	real_t currentVar = (eps[0] - stat.mean(eps)) * (eps[0] - stat.mean(eps)) / eps.size();
	for (uint32_t i = 0; i < eps.size(); i++) {
		if (currentVar != (eps[i] - stat.mean(eps)) * (eps[i] - stat.mean(eps)) / eps.size()) {
			return false;
		}
	}

	return true;
}

bool MLPPGaussMarkovCheckerOld::exogeneity(std::vector<real_t> eps) {
	MLPPStatOld stat;
	for (uint32_t i = 0; i < eps.size(); i++) {
		for (uint32_t j = 0; j < eps.size(); j++) {
			if (i != j) {
				if ((eps[i] - stat.mean(eps)) * (eps[j] - stat.mean(eps)) / eps.size() != 0) {
					return false;
				}
			}
		}
	}

	return true;
}

void MLPPGaussMarkovCheckerOld::_bind_methods() {
}
