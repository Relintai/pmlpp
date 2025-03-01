/*************************************************************************/
/*  c_log_log_reg.cpp                                                    */
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

#include "c_log_log_reg.h"
#include "../core/activation.h"
#include "../core/cost.h"
#include "../core/reg.h"
#include "../core/utilities.h"

#include <random>

Ref<MLPPVector> MLPPCLogLogReg::model_set_test(const Ref<MLPPMatrix> &X) {
	return evaluatem(X);
}

real_t MLPPCLogLogReg::model_test(const Ref<MLPPVector> &x) {
	return evaluatev(x);
}

void MLPPCLogLogReg::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	MLPPActivation avn;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPVector> error = _y_hat->subn(_output_set);

		// Calculating the weight gradients
		_weights->sub(_input_set->transposen()->mult_vec(error->hadamard_productn(avn.cloglog_derivv(_z)))->scalar_multiplyn(learning_rate / _n));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients

		bias -= learning_rate * error->hadamard_productn(avn.cloglog_derivv(_z))->sum_elements() / _n;

		forward_pass();

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_y_hat, _output_set));
			MLPPUtilities::print_ui_vb(_weights, bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPCLogLogReg::mle(real_t learning_rate, int max_epoch, bool ui) {
	MLPPActivation avn;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPVector> error = _y_hat->subn(_output_set);

		_weights->add(_input_set->transposen()->mult_vec(error->hadamard_productn(avn.cloglog_derivv(_z)))->scalar_multiplyn(learning_rate / _n));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		bias += learning_rate * error->hadamard_productn(avn.cloglog_derivv(_z))->sum_elements() / _n;

		forward_pass();

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_y_hat, _output_set));
			MLPPUtilities::print_ui_vb(_weights, bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPCLogLogReg::sgd(real_t learning_rate, int max_epoch, bool p_) {
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(_n - 1));

	forward_pass();

	Ref<MLPPVector> input_set_row_tmp;
	input_set_row_tmp.instance();
	input_set_row_tmp->resize(_input_set->size().x);

	Ref<MLPPVector> y_hat_row_tmp;
	y_hat_row_tmp.instance();
	y_hat_row_tmp->resize(1);

	Ref<MLPPVector> output_set_row_tmp;
	output_set_row_tmp.instance();
	output_set_row_tmp->resize(1);

	while (true) {
		int output_index = distribution(generator);

		_input_set->row_get_into_mlpp_vector(output_index, input_set_row_tmp);
		real_t output_element_set = _output_set->element_get(output_index);
		output_set_row_tmp->element_set(0, output_element_set);

		real_t y_hat = evaluatev(input_set_row_tmp);
		y_hat_row_tmp->element_set(0, y_hat);

		real_t z = propagatev(input_set_row_tmp);

		cost_prev = cost(y_hat_row_tmp, output_set_row_tmp);

		real_t error = y_hat - output_element_set;

		// Weight Updation

		_weights->sub(input_set_row_tmp->scalar_multiplyn(learning_rate * error * Math::exp(z - Math::exp(z))));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Bias updation
		bias -= learning_rate * error * exp(z - exp(z));

		y_hat = evaluatev(input_set_row_tmp);

		if (p_) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat_row_tmp, output_set_row_tmp));
			MLPPUtilities::print_ui_vb(_weights, bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPCLogLogReg::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool p_) {
	MLPPActivation avn;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_batch);
			Ref<MLPPVector> z = propagatem(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			Ref<MLPPVector> error = y_hat->subn(current_output_batch);

			// Calculating the weight gradients
			_weights->sub(current_input_batch->transposen()->mult_vec(error->hadamard_productn(avn.cloglog_derivv(z)))->scalar_multiplyn(learning_rate / _n));

			_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients
			bias -= learning_rate * error->hadamard_productn(avn.cloglog_derivv(z))->sum_elements() / _n;

			forward_pass();

			y_hat = evaluatem(current_input_batch);

			if (p_) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_batch));
				MLPPUtilities::print_ui_vb(_weights, bias);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

real_t MLPPCLogLogReg::score() {
	MLPPUtilities util;

	return util.performance_vec(_y_hat, _output_set);
}

MLPPCLogLogReg::MLPPCLogLogReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = _input_set->size().y;
	_k = _input_set->size().x;
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.instance();
	_y_hat->resize(_n);

	MLPPUtilities utilities;

	_weights.instance();
	_weights->resize(_k);
	utilities.weight_initializationv(_weights);

	bias = utilities.bias_initializationr();
}

MLPPCLogLogReg::MLPPCLogLogReg() {
}
MLPPCLogLogReg::~MLPPCLogLogReg() {
}

real_t MLPPCLogLogReg::cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;

	return mlpp_cost.msev(y_hat, y) + regularization.reg_termv(_weights, _lambda, _alpha, _reg);
}

real_t MLPPCLogLogReg::evaluatev(const Ref<MLPPVector> &x) {
	MLPPActivation avn;

	return avn.cloglog_normr(_weights->dot(x) + bias);
}

real_t MLPPCLogLogReg::propagatev(const Ref<MLPPVector> &x) {
	return _weights->dot(x) + bias;
}

Ref<MLPPVector> MLPPCLogLogReg::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPActivation avn;

	return avn.cloglog_normv(X->mult_vec(_weights)->scalar_addn(bias));
}

Ref<MLPPVector> MLPPCLogLogReg::propagatem(const Ref<MLPPMatrix> &X) {
	return X->mult_vec(_weights)->scalar_addn(bias);
}

// cloglog ( wTx + b )
void MLPPCLogLogReg::forward_pass() {
	MLPPActivation avn;

	_z = propagatem(_input_set);
	_y_hat = avn.cloglog_normv(_z);
}

void MLPPCLogLogReg::_bind_methods() {
}
