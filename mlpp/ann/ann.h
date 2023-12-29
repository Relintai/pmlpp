#ifndef MLPP_ANN_H
#define MLPP_ANN_H

/*************************************************************************/
/*  ann.h                                                                */
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

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_tensor3.h"
#include "../lin_alg/mlpp_vector.h"

#include "../hidden_layer/hidden_layer.h"
#include "../output_layer/output_layer.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

class MLPPANN : public Reference {
	GDCLASS(MLPPANN, Reference);

public:
	enum SchedulerType {
		SCHEDULER_TYPE_NONE = 0,
		SCHEDULER_TYPE_TIME,
		SCHEDULER_TYPE_EPOCH,
		SCHEDULER_TYPE_STEP,
		SCHEDULER_TYPE_EXPONENTIAL,
	};

public:
	Ref<MLPPVector> model_set_test(const Ref<MLPPMatrix> &X);
	real_t model_test(const Ref<MLPPVector> &x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);
	void momentum(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool nag, bool ui = false);
	void adagrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t e, bool ui = false);
	void adadelta(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t e, bool ui = false);
	void adam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui = false);
	void adamax(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui = false);
	void nadam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui = false);
	void amsgrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui = false);

	real_t score();
	void save(const String &file_name);

	void set_learning_rate_scheduler(SchedulerType type, real_t decay_constant);
	void set_learning_rate_scheduler_drop(SchedulerType type, real_t decay_constant, real_t drop_rate);

	void add_layer(int n_hidden, MLPPActivation::ActivationFunction activation, MLPPUtilities::WeightDistributionType weight_init = MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT, MLPPReg::RegularizationType reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t lambda = 0.5, real_t alpha = 0.5);
	void add_output_layer(MLPPActivation::ActivationFunction activation, MLPPCost::CostTypes loss, MLPPUtilities::WeightDistributionType weight_init = MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT, MLPPReg::RegularizationType reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t lambda = 0.5, real_t alpha = 0.5);

	MLPPANN(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set);

	MLPPANN();
	~MLPPANN();

protected:
	real_t apply_learning_rate_scheduler(real_t learning_rate, real_t decay_constant, real_t epoch, real_t drop_rate);

	real_t cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);

	void forward_pass();
	void update_parameters(const Vector<Ref<MLPPMatrix>> &hidden_layer_updations, const Ref<MLPPVector> &output_layer_updation, real_t learning_rate);

	struct ComputeGradientsResult {
		Vector<Ref<MLPPMatrix>> cumulative_hidden_layer_w_grad;
		Ref<MLPPVector> output_w_grad;

		ComputeGradientsResult() {
			output_w_grad.instance();
		}
	};

	ComputeGradientsResult compute_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &_output_set);

	void print_ui(int epoch, real_t cost_prev, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &p_output_set);

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPVector> _output_set;
	Ref<MLPPVector> _y_hat;

	Vector<Ref<MLPPHiddenLayer>> _network;
	Ref<MLPPOutputLayer> _output_layer;

	int _n;
	int _k;

	SchedulerType _lr_scheduler;
	real_t _decay_constant;
	real_t _drop_rate;
};

VARIANT_ENUM_CAST(MLPPANN::SchedulerType);

#endif /* ANN_hpp */