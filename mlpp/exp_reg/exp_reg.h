
#ifndef MLPP_EXP_REG_H
#define MLPP_EXP_REG_H

//
//  ExpReg.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include <string>
#include <vector>

class MLPPExpReg {
public:
	std::vector<real_t> model_set_test(std::vector<std::vector<real_t>> X);
	real_t model_test(std::vector<real_t> x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	void save(std::string file_name);

	MLPPExpReg(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, std::string p_reg = "None", real_t p_lambda = 0.5, real_t p_alpha = 0.5);

private:
	real_t cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	real_t evaluatev(std::vector<real_t> x);
	std::vector<real_t> evaluatem(std::vector<std::vector<real_t>> X);

	void forward_pass();

	std::vector<std::vector<real_t>> _input_set;
	std::vector<real_t> _output_set;
	std::vector<real_t> _y_hat;
	std::vector<real_t> _weights;
	std::vector<real_t> _initial;
	real_t _bias;

	int _n;
	int _k;

	// Regularization Params
	std::string _reg;
	real_t _lambda;
	real_t _alpha; /* This is the controlling param for Elastic Net*/
};

#endif /* ExpReg_hpp */
