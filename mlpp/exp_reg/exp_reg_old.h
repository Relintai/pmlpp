
#ifndef MLPP_EXP_REG_OLD_H
#define MLPP_EXP_REG_OLD_H

//
//  ExpReg.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include <string>
#include <vector>

class MLPPExpRegOld {
public:
	MLPPExpRegOld(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);
	std::vector<real_t> modelSetTest(std::vector<std::vector<real_t>> X);
	real_t modelTest(std::vector<real_t> x);
	void gradientDescent(real_t learning_rate, int max_epoch, bool UI = 1);
	void SGD(real_t learning_rate, int max_epoch, bool UI = 1);
	void MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI = 1);
	real_t score();
	void save(std::string fileName);

private:
	real_t Cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	std::vector<real_t> Evaluate(std::vector<std::vector<real_t>> X);
	real_t Evaluate(std::vector<real_t> x);
	void forwardPass();

	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;
	std::vector<real_t> y_hat;
	std::vector<real_t> weights;
	std::vector<real_t> initial;
	real_t bias;

	int n;
	int k;

	// Regularization Params
	std::string reg;
	real_t lambda;
	real_t alpha; /* This is the controlling param for Elastic Net*/
};

#endif /* ExpReg_hpp */
