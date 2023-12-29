/*************************************************************************/
/*  log_reg.cpp                                                          */
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

#include "log_reg.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

/*
Ref<MLPPMatrix> MLPPLogReg::get_input_set() {
	return _input_set;
}
void MLPPLogReg::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;

	_initialized = false;
}

Ref<MLPPVector> MLPPLogReg::get_output_set() {
	return _output_set;
}
void MLPPLogReg::set_output_set(const Ref<MLPPVector> &val) {
	_output_set = val;

	_initialized = false;
}

MLPPReg::RegularizationType MLPPLogReg::get_reg() {
	return _reg;
}
void MLPPLogReg::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;

	_initialized = false;
}

real_t MLPPLogReg::get_lambda() {
	return _lambda;
}
void MLPPLogReg::set_lambda(const real_t val) {
	_lambda = val;

	_initialized = false;
}

real_t MLPPLogReg::get_alpha() {
	return _alpha;
}
void MLPPLogReg::set_alpha(const real_t val) {
	_alpha = val;

	_initialized = false;
}
*/

Ref<MLPPVector> MLPPLogReg::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPVector>());

	return evaluatem(X);
}

real_t MLPPLogReg::model_test(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!_initialized, 0);

	return evaluatev(x);
}

void MLPPLogReg::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPVector> error = _y_hat->subn(_output_set);

		// Calculating the weight gradients
		_weights->sub(_input_set->transposen()->mult_vec(error)->scalar_multiplyn(learning_rate / _n));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		_bias -= learning_rate * error->sum_elements() / _n;

		forward_pass();

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_y_hat, _output_set));
			MLPPUtilities::print_ui_vb(_weights, _bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPLogReg::mle(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPVector> error = _output_set->subn(_y_hat);

		// Calculating the weight gradients
		_weights->add(_input_set->transposen()->mult_vec(error)->scalar_multiplyn(learning_rate / _n));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		_bias += learning_rate * error->sum_elements() / _n;

		forward_pass();

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_y_hat, _output_set));
			MLPPUtilities::print_ui_vb(_weights, _bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPLogReg::sgd(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(_n - 1));

	Ref<MLPPVector> input_row_tmp;
	input_row_tmp.instance();
	input_row_tmp->resize(_input_set->size().x);

	Ref<MLPPVector> y_hat_tmp;
	y_hat_tmp.instance();
	y_hat_tmp->resize(1);

	Ref<MLPPVector> output_element_set_tmp;
	output_element_set_tmp.instance();
	output_element_set_tmp->resize(1);

	while (true) {
		int output_index = distribution(generator);

		_input_set->row_get_into_mlpp_vector(output_index, input_row_tmp);
		real_t output_element_set = _output_set->element_get(output_index);
		output_element_set_tmp->element_set(0, output_element_set);

		real_t y_hat = evaluatev(input_row_tmp);
		y_hat_tmp->element_set(0, y_hat);

		cost_prev = cost(y_hat_tmp, output_element_set_tmp);

		real_t error = y_hat - output_element_set;

		// Weight updation
		_weights->sub(input_row_tmp->scalar_multiplyn(learning_rate * error));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Bias updation
		_bias -= learning_rate * error;

		y_hat = evaluatev(input_row_tmp);

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat_tmp, output_element_set_tmp));
			MLPPUtilities::print_ui_vb(_weights, _bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPLogReg::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	ERR_FAIL_COND(!_initialized);

	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch bacthes = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_mini_batch_input_entry = bacthes.input_sets[i];
			Ref<MLPPVector> current_mini_batch_output_entry = bacthes.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_mini_batch_input_entry);
			cost_prev = cost(y_hat, current_mini_batch_output_entry);

			Ref<MLPPVector> error = y_hat->subn(current_mini_batch_output_entry);

			// Calculating the weight gradients
			_weights->sub(current_mini_batch_input_entry->transposen()->mult_vec(error)->scalar_multiplyn(learning_rate / current_mini_batch_output_entry->size()));
			_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients
			_bias -= learning_rate * error->sum_elements() / current_mini_batch_output_entry->size();
			y_hat = evaluatem(current_mini_batch_input_entry);

			if (UI) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_mini_batch_output_entry));
				MLPPUtilities::print_ui_vb(_weights, _bias);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

real_t MLPPLogReg::score() {
	ERR_FAIL_COND_V(!_initialized, 0);

	MLPPUtilities util;
	return util.performance_vec(_y_hat, _output_set);
}

void MLPPLogReg::save(std::string file_name) {
	//ERR_FAIL_COND(!_initialized);

	//MLPPUtilities util;
	//util.saveParameters(file_name, _weights, _bias);
}

bool MLPPLogReg::is_initialized() {
	return _initialized;
}
void MLPPLogReg::initialize() {
	if (_initialized) {
		return;
	}

	//ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	_initialized = true;
}

MLPPLogReg::MLPPLogReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = p_input_set->size().y;
	_k = p_input_set->size().x;
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.instance();
	_y_hat->resize(_n);

	_weights.instance();
	_weights->resize(_k);

	MLPPUtilities utils;

	utils.weight_initializationv(_weights);
	_bias = utils.bias_initializationr();

	_initialized = true;
}

MLPPLogReg::MLPPLogReg() {
	_initialized = false;
}
MLPPLogReg::~MLPPLogReg() {
}

real_t MLPPLogReg::cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPReg regularization;
	class MLPPCost cost;

	return cost.log_lossv(y_hat, y) + regularization.reg_termv(_weights, _lambda, _alpha, _reg);
}

real_t MLPPLogReg::evaluatev(const Ref<MLPPVector> &x) {
	MLPPActivation avn;

	return avn.sigmoid_normr(_weights->dot(x) + _bias);
}

Ref<MLPPVector> MLPPLogReg::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPActivation avn;

	return avn.sigmoid_normv(X->mult_vec(_weights)->scalar_addn(_bias));
}

// sigmoid ( wTx + b )
void MLPPLogReg::forward_pass() {
	_y_hat = evaluatem(_input_set);
}

void MLPPLogReg::_bind_methods() {
	/*
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPLogReg::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPLogReg::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPLogReg::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPLogReg::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_reg"), &MLPPLogReg::get_reg);
	ClassDB::bind_method(D_METHOD("set_reg", "val"), &MLPPLogReg::set_reg);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reg"), "set_reg", "get_reg");

	ClassDB::bind_method(D_METHOD("get_lambda"), &MLPPLogReg::get_lambda);
	ClassDB::bind_method(D_METHOD("set_lambda", "val"), &MLPPLogReg::set_lambda);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lambda"), "set_lambda", "get_lambda");

	ClassDB::bind_method(D_METHOD("get_alpha"), &MLPPLogReg::get_alpha);
	ClassDB::bind_method(D_METHOD("set_alpha", "val"), &MLPPLogReg::set_alpha);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alpha"), "set_alpha", "get_alpha");

	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPLogReg::model_test);
	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPLogReg::model_set_test);

	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPLogReg::gradient_descent, false);
	ClassDB::bind_method(D_METHOD("sgd", "learning_rate", "max_epoch", "ui"), &MLPPLogReg::sgd, false);
	ClassDB::bind_method(D_METHOD("mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPLogReg::mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPLogReg::score);

	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPLogReg::save);

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPLogReg::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPLogReg::initialize);
	*/
}
