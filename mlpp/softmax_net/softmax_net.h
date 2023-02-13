#ifndef MLPP_SOFTMAX_NET_H
#define MLPP_SOFTMAX_NET_H

//
//  SoftmaxNet.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../regularization/reg.h"

class MLPPSoftmaxNet : public Reference {
	GDCLASS(MLPPSoftmaxNet, Reference);

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

	Ref<MLPPVector> model_test(const Ref<MLPPVector> &x);
	Ref<MLPPMatrix> model_set_test(const Ref<MLPPMatrix> &X);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	void save(const String &file_name);

	Ref<MLPPMatrix> get_embeddings(); // This class is used (mostly) for word2Vec. This function returns our embeddings.

	bool is_initialized();
	void initialize();

	MLPPSoftmaxNet(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPMatrix> &p_output_set, int p_n_hidden, MLPPReg::RegularizationType p_reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t p_lambda = 0.5, real_t p_alpha = 0.5);

	MLPPSoftmaxNet();
	~MLPPSoftmaxNet();

protected:
	real_t cost(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	Ref<MLPPVector> evaluatev(const Ref<MLPPVector> &x);

	struct PropagateVResult {
		Ref<MLPPVector> z2;
		Ref<MLPPVector> a2;
	};

	PropagateVResult propagatev(const Ref<MLPPVector> &x);

	Ref<MLPPMatrix> evaluatem(const Ref<MLPPMatrix> &X);

	struct PropagateMResult {
		Ref<MLPPMatrix> z2;
		Ref<MLPPMatrix> a2;
	};

	PropagateMResult propagatem(const Ref<MLPPMatrix> &X);

	void forward_pass();

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPMatrix> _output_set;
	Ref<MLPPMatrix> _y_hat;

	Ref<MLPPMatrix> _weights1;
	Ref<MLPPMatrix> _weights2;

	Ref<MLPPVector> _bias1;
	Ref<MLPPVector> _bias2;

	Ref<MLPPMatrix> _z2;
	Ref<MLPPMatrix> _a2;

	int _n;
	int _k;
	int _n_class;
	int _n_hidden;

	// Regularization Params
	MLPPReg::RegularizationType _reg;
	real_t _lambda;
	real_t _alpha; /* This is the controlling param for Elastic Net*/

	bool _initialized;
};

#endif /* SoftmaxNet_hpp */
