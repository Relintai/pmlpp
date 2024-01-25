#ifndef MLPP_SOFTMAX_NET_H
#define MLPP_SOFTMAX_NET_H

/*************************************************************************/
/*  softmax_net.h                                                        */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2023-present Péter Magyar.                              */
/* Copyright (c) 2022-2023 Marc Melikyan                                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifdef USING_SFW
#include "sfw.h"
#else
#include "core/math/math_defs.h"

#include "core/object/resource.h"
#endif

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../regularization/reg.h"

class MLPPSoftmaxNet : public Resource {
	GDCLASS(MLPPSoftmaxNet, Resource);

public:
	Ref<MLPPMatrix> get_input_set() const;
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> get_output_set() const;
	void set_output_set(const Ref<MLPPMatrix> &val);

	int get_n_hidden() const;
	void set_n_hidden(const int val);

	MLPPReg::RegularizationType get_reg() const;
	void set_reg(const MLPPReg::RegularizationType val);

	real_t get_lambda() const;
	void set_lambda(const real_t val);

	real_t get_alpha() const;
	void set_alpha(const real_t val);

	Ref<MLPPMatrix> data_y_hat_get() const;
	void data_y_hat_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> data_weights1_get() const;
	void data_weights1_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> data_weights2_get() const;
	void data_weights2_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> data_bias1_get() const;
	void data_bias1_set(const Ref<MLPPVector> &val);

	Ref<MLPPVector> data_bias2_get() const;
	void data_bias2_set(const Ref<MLPPVector> &val);

	Ref<MLPPMatrix> data_z2_get() const;
	void data_z2_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> data_a2_get() const;
	void data_a2_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> model_test(const Ref<MLPPVector> &x);
	Ref<MLPPMatrix> model_set_test(const Ref<MLPPMatrix> &X);

	void train_gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void train_sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void train_mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	Ref<MLPPMatrix> get_embeddings(); // This class is used (mostly) for word2Vec. This function returns our embeddings.

	bool needs_init() const;
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
	int _n_hidden;
	// Regularization Params
	MLPPReg::RegularizationType _reg;
	real_t _lambda;
	real_t _alpha; /* This is the controlling param for Elastic Net*/

	Ref<MLPPMatrix> _y_hat;

	Ref<MLPPMatrix> _weights1;
	Ref<MLPPMatrix> _weights2;

	Ref<MLPPVector> _bias1;
	Ref<MLPPVector> _bias2;

	Ref<MLPPMatrix> _z2;
	Ref<MLPPMatrix> _a2;
};

#endif /* SoftmaxNet_hpp */
