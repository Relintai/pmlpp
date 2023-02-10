
#ifndef MLPP_BERNOULLI_NB_OLD_H
#define MLPP_BERNOULLI_NB_OLD_H

//
//  BernoulliNB.hpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "core/math/math_defs.h"

#include <map>
#include <vector>

class MLPPBernoulliNBOld {
public:
	MLPPBernoulliNBOld(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet);
	std::vector<real_t> modelSetTest(std::vector<std::vector<real_t>> X);
	real_t modelTest(std::vector<real_t> x);
	real_t score();

private:
	void computeVocab();
	void computeTheta();
	void Evaluate();

	// Model Params
	real_t prior_1 = 0;
	real_t prior_0 = 0;

	std::vector<std::map<real_t, int>> theta;
	std::vector<real_t> vocab;
	int class_num;

	// Datasets
	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;
	std::vector<real_t> y_hat;
};

#endif /* BernoulliNB_hpp */