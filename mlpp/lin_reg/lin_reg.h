
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

	Ref<MLPPVector> model_set_test(const Ref<MLPPMatrix> &X);
	real_t model_test(const Ref<MLPPVector> &x);

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

	void save(const String &file_name);

	bool is_initialized();
	void initialize();

	MLPPLinReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t p_lambda = 0.5, real_t p_alpha = 0.5);

	MLPPLinReg();
	~MLPPLinReg();

protected:
	real_t cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);

	real_t evaluatev(const Ref<MLPPVector> &x);
	Ref<MLPPVector> evaluatem(const Ref<MLPPMatrix> &X);

	void forward_pass();

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPVector> _output_set;
	Ref<MLPPVector> _y_hat;
	Ref<MLPPVector> _weights;
	real_t _bias;

	int _n;
	int _k;

	// Regularization Params
	MLPPReg::RegularizationType _reg;
	int _lambda;
	int _alpha; /* This is the controlling param for Elastic Net*/

	bool _initialized;
};

#endif /* LinReg_hpp */
