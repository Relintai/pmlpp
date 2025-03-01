#ifndef MLPP_GAN_H
#define MLPP_GAN_H

/*************************************************************************/
/*  gan.h                                                                */
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

#include "core/object/reference.h"
#endif

#include "../hidden_layer/hidden_layer.h"
#include "../output_layer/output_layer.h"

#include "../core/mlpp_tensor3.h"

#include "../core/activation.h"
#include "../core/utilities.h"

class MLPPGAN : public Reference {
	GDCLASS(MLPPGAN, Reference);

public:
	/*
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output_set();
	void set_output_set(const Ref<MLPPVector> &val);

	int get_k();
	void set_k(const int val);
	*/

	Ref<MLPPMatrix> generate_example(int n);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);

	real_t score();

	void save(const String &file_name);

	void add_layer(int n_hidden, MLPPActivation::ActivationFunction activation, MLPPUtilities::WeightDistributionType weight_init = MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT, MLPPReg::RegularizationType reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t lambda = 0.5, real_t alpha = 0.5);
	void add_output_layer(MLPPUtilities::WeightDistributionType weight_init = MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT, MLPPReg::RegularizationType reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t lambda = 0.5, real_t alpha = 0.5);

	MLPPGAN(real_t k, const Ref<MLPPMatrix> &output_set);

	MLPPGAN();
	~MLPPGAN();

protected:
	Ref<MLPPMatrix> model_set_test_generator(const Ref<MLPPMatrix> &X); // Evaluator for the generator of the gan.
	Ref<MLPPVector> model_set_test_discriminator(const Ref<MLPPMatrix> &X); // Evaluator for the discriminator of the gan.

	real_t cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);

	void forward_pass();

	void update_discriminator_parameters(const Ref<MLPPTensor3> &hidden_layer_updations, const Ref<MLPPVector> &output_layer_updation, real_t learning_rate);
	void update_generator_parameters(const Ref<MLPPTensor3> &hidden_layer_updations, real_t learning_rate);

	struct ComputeDiscriminatorGradientsResult {
		Ref<MLPPTensor3> cumulative_hidden_layer_w_grad; // Tensor containing ALL hidden grads.
		Ref<MLPPVector> output_w_grad;

		ComputeDiscriminatorGradientsResult() {
			cumulative_hidden_layer_w_grad.instance();
			output_w_grad.instance();
		}
	};

	ComputeDiscriminatorGradientsResult compute_discriminator_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set);
	Ref<MLPPTensor3> compute_generator_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set);

	void print_ui(int epoch, real_t cost_prev, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set);

	static void _bind_methods();

	Ref<MLPPMatrix> _output_set;
	Ref<MLPPVector> _y_hat;

	Vector<Ref<MLPPHiddenLayer>> _network;
	Ref<MLPPOutputLayer> _output_layer;

	int _n;
	int _k;
};

#endif /* GAN_hpp */