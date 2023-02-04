
#ifndef MLPP_LIN_REG_H
#define MLPP_LIN_REG_H

//
//  LinReg.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include <string>
#include <vector>

class MLPPLinReg {
public:
	MLPPLinReg(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);
	std::vector<real_t> modelSetTest(std::vector<std::vector<real_t>> X);
	real_t modelTest(std::vector<real_t> x);
	void NewtonRaphson(real_t learning_rate, int max_epoch, bool UI);
	void gradientDescent(real_t learning_rate, int max_epoch, bool UI = false);
	void SGD(real_t learning_rate, int max_epoch, bool UI = false);

	void Momentum(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool UI = false);
	void NAG(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool UI = false);
	void Adagrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t e, bool UI = false);
	void Adadelta(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t e, bool UI = false);
	void Adam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI = false);
	void Adamax(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI = false);
	void Nadam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI = false);

	void MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI = false);
	void normalEquation();
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
	real_t bias;

	int n;
	int k;

	// Regularization Params
	std::string reg;
	int lambda;
	int alpha; /* This is the controlling param for Elastic Net*/
};

#endif /* LinReg_hpp */
