
#ifndef MLPP_GAUSS_MARKOV_CHECKER_H
#define MLPP_GAUSS_MARKOV_CHECKER_H

//
//  GaussMarkovChecker.hpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "core/math/math_defs.h"

#include <string>
#include <vector>


class MLPPGaussMarkovChecker {
public:
	void checkGMConditions(std::vector<real_t> eps);

	// Independent, 3 Gauss-Markov Conditions
	bool arithmeticMean(std::vector<real_t> eps); // 1) Arithmetic Mean of 0.
	bool homoscedasticity(std::vector<real_t> eps); // 2) Homoscedasticity
	bool exogeneity(std::vector<real_t> eps); // 3) Cov of any 2 non-equal eps values = 0.
private:
};


#endif /* GaussMarkovChecker_hpp */
