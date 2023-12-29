

#include "gauss_markov_checker.h"
#include "../stat/stat.h"
#include <iostream>

/*
void MLPPGaussMarkovChecker::checkGMConditions(std::vector<real_t> eps) {
	bool condition1 = arithmeticMean(eps);
	bool condition2 = homoscedasticity(eps);
	bool condition3 = exogeneity(eps);

	if (condition1 && condition2 && condition3) {
		std::cout << "Gauss-Markov conditions were not violated. You may use OLS to obtain a BLUE estimator" << std::endl;
	} else {
		std::cout << "A test of the expected value of 0 of the error terms returned " << std::boolalpha << condition1 << ", a test of homoscedasticity has returned " << std::boolalpha << condition2 << ", and a test of exogenity has returned " << std::boolalpha << "." << std::endl;
	}
}

bool MLPPGaussMarkovChecker::arithmeticMean(std::vector<real_t> eps) {
	MLPPStat stat;
	if (stat.mean(eps) == 0) {
		return true;
	} else {
		return false;
	}
}

bool MLPPGaussMarkovChecker::homoscedasticity(std::vector<real_t> eps) {
	MLPPStat stat;
	real_t currentVar = (eps[0] - stat.mean(eps)) * (eps[0] - stat.mean(eps)) / eps.size();
	for (uint32_t i = 0; i < eps.size(); i++) {
		if (currentVar != (eps[i] - stat.mean(eps)) * (eps[i] - stat.mean(eps)) / eps.size()) {
			return false;
		}
	}

	return true;
}

bool MLPPGaussMarkovChecker::exogeneity(std::vector<real_t> eps) {
	MLPPStat stat;
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
*/

void MLPPGaussMarkovChecker::_bind_methods() {
}
