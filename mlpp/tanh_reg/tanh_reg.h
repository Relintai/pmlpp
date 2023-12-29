
#ifndef MLPP_TANH_REG_H
#define MLPP_TANH_REG_H


#include "core/math/math_defs.h"

#include "core/object/resource.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../regularization/reg.h"

class MLPPTanhReg : public Resource {
	GDCLASS(MLPPTanhReg, Resource);

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

	Ref<MLPPVector> data_z_get() const;
	void data_z_set(const Ref<MLPPVector> &val);

	Ref<MLPPVector> data_y_hat_get() const;
	void data_y_hat_set(const Ref<MLPPVector> &val);

	Ref<MLPPVector> data_weights_get() const;
	void data_weights_set(const Ref<MLPPVector> &val);

	real_t data_bias_get() const;
	void data_bias_set(const real_t val);

	bool needs_init() const;
	void initialize();

	Ref<MLPPVector> model_set_test(const Ref<MLPPMatrix> &X);
	real_t model_test(const Ref<MLPPVector> &x);

	void train_gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void train_sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void train_mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	MLPPTanhReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t p_lambda = 0.5, real_t p_alpha = 0.5);

	MLPPTanhReg();
	~MLPPTanhReg();

protected:
	real_t cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);

	real_t evaluatev(const Ref<MLPPVector> &x);
	real_t propagatev(const Ref<MLPPVector> &x);

	Ref<MLPPVector> evaluatem(const Ref<MLPPMatrix> &X);
	Ref<MLPPVector> propagatem(const Ref<MLPPMatrix> &X);

	void forward_pass();

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPVector> _output_set;

	// Regularization Params
	MLPPReg::RegularizationType _reg;
	real_t _lambda;
	real_t _alpha; /* This is the controlling param for Elastic Net*/

	Ref<MLPPVector> _z;
	Ref<MLPPVector> _y_hat;
	Ref<MLPPVector> _weights;
	real_t _bias;
};

#endif /* TanhReg_hpp */
