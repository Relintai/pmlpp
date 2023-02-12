
#ifndef MLPP_C_LOG_LOG_REG_H
#define MLPP_C_LOG_LOG_REG_H

//
//  CLogLogReg.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include <string>
#include <vector>

class MLPPCLogLogReg {
public:
	std::vector<real_t> model_set_test(std::vector<std::vector<real_t>> X);
	real_t model_test(std::vector<real_t> x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void mle(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	MLPPCLogLogReg(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, std::string p_reg = "None", real_t p_lambda = 0.5, real_t p_alpha = 0.5);

	MLPPCLogLogReg();
	~MLPPCLogLogReg();

private:
	void weight_initialization(int k);
	void bias_initialization();

	real_t cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	real_t evaluatev(std::vector<real_t> x);
	real_t propagatev(std::vector<real_t> x);

	std::vector<real_t> evaluatem(std::vector<std::vector<real_t>> X);
	std::vector<real_t> propagatem(std::vector<std::vector<real_t>> X);

	void forward_pass();

	std::vector<std::vector<real_t>> _input_set;
	std::vector<real_t> _output_set;
	std::vector<real_t> _y_hat;
	std::vector<real_t> _z;
	std::vector<real_t> _weights;
	real_t bias;

	int _n;
	int _k;

	// Regularization Params
	std::string _reg;
	real_t _lambda;
	real_t _alpha; /* This is the controlling param for Elastic Net*/
};

#endif /* CLogLogReg_hpp */
