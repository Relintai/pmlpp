
#ifndef MLPP_C_LOG_LOG_REG_H
#define MLPP_C_LOG_LOG_REG_H

//
//  CLogLogReg.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../regularization/reg.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPCLogLogReg : public Reference {
	GDCLASS(MLPPCLogLogReg, Reference);

public:
	Ref<MLPPVector> model_set_test(const Ref<MLPPMatrix> &X);
	real_t model_test(const Ref<MLPPVector> &x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void mle(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	MLPPCLogLogReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t p_lambda = 0.5, real_t p_alpha = 0.5);

	MLPPCLogLogReg();
	~MLPPCLogLogReg();

protected:
	void weight_initialization(int k);
	void bias_initialization();

	real_t cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);

	real_t evaluatev(const Ref<MLPPVector> &x);
	real_t propagatev(const Ref<MLPPVector> &x);

	Ref<MLPPVector> evaluatem(const Ref<MLPPMatrix> &X);
	Ref<MLPPVector> propagatem(const Ref<MLPPMatrix> &X);

	void forward_pass();

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPVector> _output_set;
	Ref<MLPPVector> _y_hat;
	Ref<MLPPVector> _z;
	Ref<MLPPVector> _weights;
	real_t bias;

	int _n;
	int _k;

	// Regularization Params
	MLPPReg::RegularizationType _reg;
	real_t _lambda;
	real_t _alpha; /* This is the controlling param for Elastic Net*/
};

#endif /* CLogLogReg_hpp */
