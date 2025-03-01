/*************************************************************************/
/*  probit_reg.cpp                                                       */
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

#include "probit_reg.h"

#include "../core/activation.h"
#include "../core/cost.h"
#include "../core/reg.h"
#include "../core/utilities.h"

#include <random>

Ref<MLPPMatrix> MLPPProbitReg::get_input_set() {
	return _input_set;
}
void MLPPProbitReg::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;
}

Ref<MLPPVector> MLPPProbitReg::get_output_set() {
	return _output_set;
}
void MLPPProbitReg::set_output_set(const Ref<MLPPVector> &val) {
	_output_set = val;
}

MLPPReg::RegularizationType MLPPProbitReg::get_reg() {
	return _reg;
}
void MLPPProbitReg::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;
}

real_t MLPPProbitReg::get_lambda() {
	return _lambda;
}
void MLPPProbitReg::set_lambda(const real_t val) {
	_lambda = val;
}

real_t MLPPProbitReg::get_alpha() {
	return _alpha;
}
void MLPPProbitReg::set_alpha(const real_t val) {
	_alpha = val;
}

Ref<MLPPVector> MLPPProbitReg::data_z_get() const {
	return _z;
}
void MLPPProbitReg::data_z_set(const Ref<MLPPVector> &val) {
	if (!val.is_valid()) {
		return;
	}

	_z = val;
}

Ref<MLPPVector> MLPPProbitReg::data_y_hat_get() const {
	return _y_hat;
}
void MLPPProbitReg::data_y_hat_set(const Ref<MLPPVector> &val) {
	if (!val.is_valid()) {
		return;
	}

	_y_hat = val;
}

Ref<MLPPVector> MLPPProbitReg::data_weights_get() const {
	return _weights;
}
void MLPPProbitReg::data_weights_set(const Ref<MLPPVector> &val) {
	if (!val.is_valid()) {
		return;
	}

	_weights = val;
}

real_t MLPPProbitReg::data_bias_get() const {
	return _bias;
}
void MLPPProbitReg::data_bias_set(const real_t val) {
	_bias = val;
}

Ref<MLPPVector> MLPPProbitReg::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(needs_init(), Ref<MLPPVector>());

	return evaluatem(X);
}

real_t MLPPProbitReg::model_test(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(needs_init(), 0);

	return evaluatev(x);
}

void MLPPProbitReg::train_gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(needs_init());

	MLPPActivation avn;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	int n = _input_set->size().y;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPVector> error = _y_hat->subn(_output_set);

		// Calculating the weight gradients
		_weights->sub(_input_set->transposen()->mult_vec(error->hadamard_productn(avn.gaussian_cdf_derivv(_z)))->scalar_multiplyn(learning_rate / n));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		_bias -= learning_rate * error->hadamard_productn(avn.gaussian_cdf_derivv(_z))->sum_elements() / n;

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

void MLPPProbitReg::train_mle(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(needs_init());

	MLPPActivation avn;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	int n = _input_set->size().y;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPVector> error = _output_set->subn(_y_hat);

		// Calculating the weight gradients
		_weights->add(_input_set->transposen()->mult_vec(error->hadamard_productn(avn.gaussian_cdf_derivv(_z)))->scalar_multiplyn(learning_rate / n));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		_bias += learning_rate * error->hadamard_productn(avn.gaussian_cdf_derivv(_z))->sum_elements() / n;

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

void MLPPProbitReg::train_sgd(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(needs_init());

	// NOTE: ∂y_hat/∂z is sparse
	MLPPActivation avn;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	int n = _input_set->size().y;

	Ref<MLPPVector> input_set_row_tmp;
	input_set_row_tmp.instance();
	input_set_row_tmp->resize(_input_set->size().x);

	Ref<MLPPVector> output_set_tmp;
	output_set_tmp.instance();
	output_set_tmp->resize(1);

	Ref<MLPPVector> y_hat_tmp;
	y_hat_tmp.instance();
	y_hat_tmp->resize(1);

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(n - 1));

	while (true) {
		int output_index = distribution(generator);

		_input_set->row_get_into_mlpp_vector(output_index, input_set_row_tmp);
		real_t output_set_entry = _output_set->element_get(output_index);

		real_t y_hat = evaluatev(input_set_row_tmp);
		real_t z = propagatev(input_set_row_tmp);

		y_hat_tmp->element_set(0, y_hat);
		output_set_tmp->element_set(0, output_set_entry);

		cost_prev = cost(y_hat_tmp, output_set_tmp);

		real_t error = y_hat - output_set_entry;

		// Weight Updation
		_weights->sub(input_set_row_tmp->scalar_multiplyn(learning_rate * error * ((1 / Math::sqrt(2 * Math_PI)) * Math::exp(-z * z / 2))));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Bias updation
		_bias -= learning_rate * error * ((1 / Math::sqrt(2 * Math_PI)) * Math::exp(-z * z / 2));

		y_hat = evaluatev(input_set_row_tmp);

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat_tmp, output_set_tmp));
			MLPPUtilities::print_ui_vb(_weights, _bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPProbitReg::train_mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	ERR_FAIL_COND(needs_init());

	MLPPActivation avn;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	int n = _input_set->size().y;

	Ref<MLPPVector> z_tmp;
	z_tmp.instance();
	z_tmp->resize(1);

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;

	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input = batches.input_sets[i];
			Ref<MLPPVector> current_output = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input);
			real_t z = propagatev(current_output);

			z_tmp->element_set(0, z);

			cost_prev = cost(y_hat, current_output);

			Ref<MLPPVector> error = y_hat->subn(current_output);

			// Calculating the weight gradients
			_weights->sub(current_input->transposen()->mult_vec(error->hadamard_productn(avn.gaussian_cdf_derivv(z_tmp)))->scalar_multiplyn(learning_rate / batches.input_sets.size()));
			_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients

			_bias -= learning_rate * error->hadamard_productn(avn.gaussian_cdf_derivv(z_tmp))->sum_elements() / batches.input_sets.size();
			y_hat = evaluatev(current_input);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output));
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

real_t MLPPProbitReg::score() {
	ERR_FAIL_COND_V(needs_init(), 0);

	MLPPUtilities util;

	return util.performance_vec(_y_hat, _output_set);
}

bool MLPPProbitReg::needs_init() const {
	if (!_input_set.is_valid()) {
		return true;
	}

	if (!_output_set.is_valid()) {
		return true;
	}

	int n = _input_set->size().y;
	int k = _input_set->size().x;

	if (_y_hat->size() != n) {
		return true;
	}

	if (_weights->size() != k) {
		return true;
	}

	return false;
}

void MLPPProbitReg::initialize() {
	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	int n = _input_set->size().y;
	int k = _input_set->size().x;

	_y_hat->resize(n);

	MLPPUtilities util;
	_weights->resize(k);

	util.weight_initializationv(_weights);
	_bias = util.bias_initializationr();
}

MLPPProbitReg::MLPPProbitReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;

	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_z.instance();
	_y_hat.instance();
	_weights.instance();
	_bias = 0;

	initialize();
}

MLPPProbitReg::MLPPProbitReg() {
	// Regularization Params
	_reg = MLPPReg::REGULARIZATION_TYPE_NONE;
	_lambda = 0.5;
	_alpha = 0.5;

	_z.instance();
	_y_hat.instance();
	_weights.instance();
	_bias = 0;
}
MLPPProbitReg::~MLPPProbitReg() {
}

real_t MLPPProbitReg::cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPReg regularization;
	class MLPPCost cost;

	return cost.msev(y_hat, y) + regularization.reg_termv(_weights, _lambda, _alpha, _reg);
}

Ref<MLPPVector> MLPPProbitReg::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPActivation avn;

	return avn.gaussian_cdf_normv(X->mult_vec(_weights)->scalar_addn(_bias));
}

Ref<MLPPVector> MLPPProbitReg::propagatem(const Ref<MLPPMatrix> &X) {
	return X->mult_vec(_weights)->scalar_addn(_bias);
}

real_t MLPPProbitReg::evaluatev(const Ref<MLPPVector> &x) {
	MLPPActivation avn;

	return avn.gaussian_cdf_normr(_weights->dot(x) + _bias);
}

real_t MLPPProbitReg::propagatev(const Ref<MLPPVector> &x) {
	return _weights->dot(x) + _bias;
}

// gaussianCDF ( wTx + b )
void MLPPProbitReg::forward_pass() {
	MLPPActivation avn;

	_z = propagatem(_input_set);
	_y_hat = avn.gaussian_cdf_normv(_z);
}

void MLPPProbitReg::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPProbitReg::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPProbitReg::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPProbitReg::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPProbitReg::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_reg"), &MLPPProbitReg::get_reg);
	ClassDB::bind_method(D_METHOD("set_reg", "val"), &MLPPProbitReg::set_reg);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reg"), "set_reg", "get_reg");

	ClassDB::bind_method(D_METHOD("get_lambda"), &MLPPProbitReg::get_lambda);
	ClassDB::bind_method(D_METHOD("set_lambda", "val"), &MLPPProbitReg::set_lambda);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lambda"), "set_lambda", "get_lambda");

	ClassDB::bind_method(D_METHOD("get_alpha"), &MLPPProbitReg::get_alpha);
	ClassDB::bind_method(D_METHOD("set_alpha", "val"), &MLPPProbitReg::set_alpha);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alpha"), "set_alpha", "get_alpha");

	ADD_GROUP("Data", "data");
	ClassDB::bind_method(D_METHOD("data_z_get"), &MLPPProbitReg::data_z_get);
	ClassDB::bind_method(D_METHOD("data_z_set", "val"), &MLPPProbitReg::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "data_z", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "data_z_set", "data_z_get");

	ClassDB::bind_method(D_METHOD("data_y_hat_get"), &MLPPProbitReg::data_y_hat_get);
	ClassDB::bind_method(D_METHOD("data_y_hat_set", "val"), &MLPPProbitReg::data_y_hat_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "data_y_hat", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "data_y_hat_set", "data_y_hat_get");

	ClassDB::bind_method(D_METHOD("data_weights_get"), &MLPPProbitReg::data_weights_get);
	ClassDB::bind_method(D_METHOD("data_weights_set", "val"), &MLPPProbitReg::data_weights_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "data_weights", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "data_weights_set", "data_weights_get");

	ClassDB::bind_method(D_METHOD("data_bias_get"), &MLPPProbitReg::data_bias_get);
	ClassDB::bind_method(D_METHOD("data_bias_set", "val"), &MLPPProbitReg::data_bias_set);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "data_bias"), "data_bias_set", "data_bias_get");

	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPProbitReg::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPProbitReg::model_test);

	ClassDB::bind_method(D_METHOD("train_gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPProbitReg::train_gradient_descent, 0, false);
	ClassDB::bind_method(D_METHOD("train_mle", "learning_rate", "max_epoch", "ui"), &MLPPProbitReg::train_mle, 0, false);
	ClassDB::bind_method(D_METHOD("train_sgd", "learning_rate", "max_epoch", "ui"), &MLPPProbitReg::train_sgd, 0, false);
	ClassDB::bind_method(D_METHOD("train_mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPProbitReg::train_mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPProbitReg::score);

	ClassDB::bind_method(D_METHOD("needs_init"), &MLPPProbitReg::needs_init);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPProbitReg::initialize);
}
