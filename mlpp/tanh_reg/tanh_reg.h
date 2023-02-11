
#ifndef MLPP_TANH_REG_H
#define MLPP_TANH_REG_H

//
//  TanhReg.hpp
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

class MLPPTanhReg : public Reference {
	GDCLASS(MLPPTanhReg, Reference);

public:
	/*
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> get_output_set();
	void set_output_set(const Ref<MLPPMatrix> &val);

	MLPPReg::RegularizationType get_reg();
	void set_reg(const MLPPReg::RegularizationType val);

	real_t get_lambda();
	void set_lambda(const real_t val);

	real_t get_alpha();
	void set_alpha(const real_t val);
	*/

	std::vector<real_t> model_set_test(std::vector<std::vector<real_t>> X);
	real_t model_test(std::vector<real_t> x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	void save(std::string file_name);

	bool is_initialized();
	void initialize();

	MLPPTanhReg(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, MLPPReg::RegularizationType p_reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t p_lambda = 0.5, real_t p_alpha = 0.5);

	MLPPTanhReg();
	~MLPPTanhReg();

protected:
	real_t cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	real_t evaluatev(std::vector<real_t> x);
	real_t propagatev(std::vector<real_t> x);

	std::vector<real_t> evaluatem(std::vector<std::vector<real_t>> X);
	std::vector<real_t> propagatem(std::vector<std::vector<real_t>> X);

	void forward_pass();

	static void _bind_methods();

	std::vector<std::vector<real_t>> _input_set;
	std::vector<real_t> _output_set;
	std::vector<real_t> _z;
	std::vector<real_t> _y_hat;
	std::vector<real_t> _weights;
	real_t _bias;

	int _n;
	int _k;

	// Regularization Params
	MLPPReg::RegularizationType _reg;
	real_t _lambda;
	real_t _alpha; /* This is the controlling param for Elastic Net*/

	bool _initialized;
};

#endif /* TanhReg_hpp */
