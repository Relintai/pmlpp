
#ifndef MLPP_LOG_REG_H
#define MLPP_LOG_REG_H

//
//  LogReg.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../regularization/reg.h"

class MLPPLogReg : public Reference {
	GDCLASS(MLPPLogReg, Reference);

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

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void mle(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	void save(std::string file_name);

	bool is_initialized();
	void initialize();

	MLPPLogReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t p_lambda = 0.5, real_t p_alpha = 0.5);

	MLPPLogReg();
	~MLPPLogReg();

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
	real_t _learning_rate;

	// Regularization Params
	MLPPReg::RegularizationType _reg;
	real_t _lambda; /* Regularization Parameter */
	real_t _alpha; /* This is the controlling param for Elastic Net*/

	bool _initialized;
};

#endif /* LogReg_hpp */
