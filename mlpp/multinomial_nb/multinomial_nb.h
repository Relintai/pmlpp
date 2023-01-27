
#ifndef MLPP_MULTINOMIAL_NB_H
#define MLPP_MULTINOMIAL_NB_H

//
//  MultinomialNB.hpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "core/math/math_defs.h"

#include <map>
#include <vector>


class MLPPMultinomialNB {
public:
	MLPPMultinomialNB(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, int class_num);
	std::vector<real_t> modelSetTest(std::vector<std::vector<real_t>> X);
	real_t modelTest(std::vector<real_t> x);
	real_t score();

private:
	void computeTheta();
	void Evaluate();

	// Model Params
	std::vector<real_t> priors;

	std::vector<std::map<real_t, int>> theta;
	std::vector<real_t> vocab;
	int class_num;

	// Datasets
	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;
	std::vector<real_t> y_hat;
};

#endif /* MultinomialNB_hpp */
