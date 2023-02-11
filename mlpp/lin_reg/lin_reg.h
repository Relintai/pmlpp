
#ifndef MLPP_LIN_REG_H
#define MLPP_LIN_REG_H

//
//  LinReg.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../regularization/reg.h"

#include <string>
#include <vector>

class MLPPLinReg : public Reference {
	GDCLASS(MLPPLinReg, Reference);

public:
	/*
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output_set();
	void set_output_set(const Ref<MLPPVector> &val);

	MLPPReg::RegularizationType get_reg();
	void set_reg(const MLPPReg::RegularizationType val);

	real_t get_lambda();
	void set_lambda(const real_t val);

	real_t get_alpha();
	void set_alpha(const real_t val);
	*/

	std::vector<real_t> model_set_test(std::vector<std::vector<real_t>> X);
	real_t model_test(std::vector<real_t> x);

	void newton_raphson(real_t learning_rate, int max_epoch, bool ui = false);
	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void momentum(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool ui = false);
	void nag(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool ui = false);
	void adagrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t e, bool ui = false);
	void adadelta(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t e, bool ui = false);
	void adam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui = false);
	void adamax(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui = false);
	void nadam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	void normal_equation();

	real_t score();

	void save(std::string fileName);

	bool is_initialized();
	void initialize();

	MLPPLinReg(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, std::string p_reg = "None", real_t p_lambda = 0.5, real_t p_alpha = 0.5);

	MLPPLinReg();
	~MLPPLinReg();

protected:
	real_t cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	real_t evaluatev(std::vector<real_t> x);
	std::vector<real_t> evaluatem(std::vector<std::vector<real_t>> X);

	void forward_pass();

	static void _bind_methods();

	std::vector<std::vector<real_t>> _input_set;
	std::vector<real_t> _output_set;
	std::vector<real_t> _y_hat;
	std::vector<real_t> _weights;
	real_t _bias;

	int _n;
	int _k;

	// Regularization Params
	std::string _reg;
	int _lambda;
	int _alpha; /* This is the controlling param for Elastic Net*/

	bool _initialized;
};

#endif /* LinReg_hpp */
