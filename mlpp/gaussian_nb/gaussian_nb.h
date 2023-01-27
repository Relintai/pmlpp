
#ifndef MLPP_GAUSSIAN_NB_H
#define MLPP_GAUSSIAN_NB_H

//
//  GaussianNB.hpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "core/math/math_defs.h"

#include <vector>


class MLPPGaussianNB {
public:
	MLPPGaussianNB(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, int class_num);
	std::vector<real_t> modelSetTest(std::vector<std::vector<real_t>> X);
	real_t modelTest(std::vector<real_t> x);
	real_t score();

private:
	void Evaluate();

	int class_num;

	std::vector<real_t> priors;
	std::vector<real_t> mu;
	std::vector<real_t> sigma;

	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;

	std::vector<real_t> y_hat;
};

#endif /* GaussianNB_hpp */
