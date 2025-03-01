/*************************************************************************/
/*  gauss_markov_checker.cpp                                             */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2023-present Péter Magyar.                              */
/* Copyright (c) 2022-2023 Marc Melikyan                                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "gauss_markov_checker.h"
#include "../core/stat.h"
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
