
#ifndef MLPP_SOFTMAX_REG_H
#define MLPP_SOFTMAX_REG_H

//
//  SoftmaxReg.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include "core/object/resource.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../regularization/reg.h"

class MLPPSoftmaxReg : public Resource {
	GDCLASS(MLPPSoftmaxReg, Resource);

public:
	Ref<MLPPMatrix> get_input_set() const;
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> get_output_set() const;
	void set_output_set(const Ref<MLPPMatrix> &val);

	MLPPReg::RegularizationType get_reg() const;
	void set_reg(const MLPPReg::RegularizationType val);

	real_t get_lambda() const;
	void set_lambda(const real_t val);

	real_t get_alpha() const;
	void set_alpha(const real_t val);

	Ref<MLPPMatrix> data_y_hat_get() const;
	void data_y_hat_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> data_weights_get() const;
	void data_weights_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> data_bias_get() const;
	void data_bias_set(const Ref<MLPPVector> &val);

	Ref<MLPPVector> model_test(const Ref<MLPPVector> &x);
	Ref<MLPPMatrix> model_set_test(const Ref<MLPPMatrix> &X);

	void train_gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void train_sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void train_mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	bool needs_init() const;
	void initialize();

	MLPPSoftmaxReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPMatrix> &p_output_set, MLPPReg::RegularizationType p_reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t p_lambda = 0.5, real_t p_alpha = 0.5);

	MLPPSoftmaxReg();
	~MLPPSoftmaxReg();

protected:
	real_t cost(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	Ref<MLPPVector> evaluatev(const Ref<MLPPVector> &x);
	Ref<MLPPMatrix> evaluatem(const Ref<MLPPMatrix> &X);

	void forward_pass();

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPMatrix> _output_set;
	// Regularization Params
	MLPPReg::RegularizationType _reg;
	real_t _lambda;
	real_t _alpha; /* This is the controlling param for Elastic Net*/

	Ref<MLPPMatrix> _y_hat;
	Ref<MLPPMatrix> _weights;
	Ref<MLPPVector> _bias;
};

#endif /* SoftmaxReg_hpp */
