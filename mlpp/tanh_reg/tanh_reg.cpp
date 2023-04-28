//
//  TanhReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "tanh_reg.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <random>

Ref<MLPPMatrix> MLPPTanhReg::get_input_set() {
	return _input_set;
}
void MLPPTanhReg::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;

	_initialized = false;
}

Ref<MLPPMatrix> MLPPTanhReg::get_output_set() {
	return _output_set;
}
void MLPPTanhReg::set_output_set(const Ref<MLPPMatrix> &val) {
	_output_set = val;

	_initialized = false;
}

MLPPReg::RegularizationType MLPPTanhReg::get_reg() {
	return _reg;
}
void MLPPTanhReg::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;
}

real_t MLPPTanhReg::get_lambda() {
	return _lambda;
}
void MLPPTanhReg::set_lambda(const real_t val) {
	_lambda = val;
}

real_t MLPPTanhReg::get_alpha() {
	return _alpha;
}
void MLPPTanhReg::set_alpha(const real_t val) {
	_alpha = val;
}

Ref<MLPPVector> MLPPTanhReg::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPVector>());

	return evaluatem(X);
}

real_t MLPPTanhReg::model_test(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!_initialized, 0);

	return evaluatev(x);
}

void MLPPTanhReg::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPVector> error = _y_hat->subn(_output_set);

		_weights->sub(_input_set->transposen()->mult_vec(error->hadamard_productn(avn.tanh_derivv(_z)))->scalar_multiplyn(learning_rate / _n));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		_bias -= learning_rate * error->hadamard_productn(avn.tanh_derivv(_z))->sum_elements() / _n;

		forward_pass();

		// UI PORTION
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

void MLPPTanhReg::sgd(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(_n - 1));

	Ref<MLPPVector> input_set_row_tmp;
	input_set_row_tmp.instance();
	input_set_row_tmp->resize(_input_set->size().x);

	Ref<MLPPVector> output_set_row_tmp;
	output_set_row_tmp.instance();
	output_set_row_tmp->resize(1);

	Ref<MLPPVector> y_hat_row_tmp;
	y_hat_row_tmp.instance();
	y_hat_row_tmp->resize(1);

	while (true) {
		int output_index = distribution(generator);

		_input_set->get_row_into_mlpp_vector(output_index, input_set_row_tmp);
		real_t output_set_entry = _output_set->get_element(output_index);
		output_set_row_tmp->set_element(0, output_set_entry);

		real_t y_hat = evaluatev(input_set_row_tmp);
		y_hat_row_tmp->set_element(0, y_hat);

		cost_prev = cost(y_hat_row_tmp, output_set_row_tmp);

		real_t error = y_hat - output_set_entry;

		// Weight Updation
		_weights->subn(input_set_row_tmp->scalar_multiplyn(learning_rate * error * (1 - y_hat * y_hat)));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Bias updation
		_bias -= learning_rate * error * (1 - y_hat * y_hat);

		y_hat = evaluatev(input_set_row_tmp);

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat_row_tmp, output_set_row_tmp));
			MLPPUtilities::print_ui_vb(_weights, _bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPTanhReg::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch_entry = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch_entry = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_batch_entry);
			Ref<MLPPVector> z = propagatem(current_input_batch_entry);
			cost_prev = cost(y_hat, current_output_batch_entry);

			Ref<MLPPVector> error = y_hat->subn(current_output_batch_entry);

			// Calculating the weight gradients

			_weights->sub(current_input_batch_entry->transposen()->mult_vec(error->hadamard_productn(avn.tanh_derivv(z)))->scalar_multiplyn(learning_rate / _n));
			_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients
			_bias -= learning_rate * error->hadamard_productn(avn.tanh_derivv(_z))->sum_elements() / _n;

			forward_pass();

			y_hat = evaluatem(current_input_batch_entry);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_batch_entry));
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

real_t MLPPTanhReg::score() {
	ERR_FAIL_COND_V(!_initialized, 0);

	MLPPUtilities util;

	return util.performance_vec(_y_hat, _output_set);
}

void MLPPTanhReg::save(const String &file_name) {
	//MLPPUtilities util;

	//util.saveParameters(file_name, _weights, _bias);
}

bool MLPPTanhReg::is_initialized() {
	return _initialized;
}
void MLPPTanhReg::initialize() {
	if (_initialized) {
		return;
	}

	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	_n = _input_set->size().y;
	_k = _input_set->size().x;

	_y_hat->resize(_n);
	_weights->resize(_k);

	MLPPUtilities utils;

	utils.weight_initializationv(_weights);

	_bias = utils.bias_initializationr();

	_initialized = true;
}

MLPPTanhReg::MLPPTanhReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.instance();
	_weights.instance();

	_initialized = false;

	initialize();
}

MLPPTanhReg::MLPPTanhReg() {
	_initialized = false;

	_y_hat.instance();
	_weights.instance();
}
MLPPTanhReg::~MLPPTanhReg() {
}

real_t MLPPTanhReg::cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;

	return mlpp_cost.msev(y_hat, y) + regularization.reg_termv(_weights, _lambda, _alpha, _reg);
}

real_t MLPPTanhReg::evaluatev(const Ref<MLPPVector> &x) {
	MLPPActivation avn;

	return avn.tanh_normr(_weights->dot(x) + _bias);
}

real_t MLPPTanhReg::propagatev(const Ref<MLPPVector> &x) {
	return _weights->dot(x) + _bias;
}

Ref<MLPPVector> MLPPTanhReg::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPActivation avn;

	return avn.tanh_normv(X->mult_vec(_weights)->scalar_addn(_bias));
}

Ref<MLPPVector> MLPPTanhReg::propagatem(const Ref<MLPPMatrix> &X) {
	return X->mult_vec(_weights)->scalar_addn(_bias);
}

// Tanh ( wTx + b )
void MLPPTanhReg::forward_pass() {
	MLPPActivation avn;

	_z = propagatem(_input_set);
	_y_hat = avn.tanh_normv(_z);
}

void MLPPTanhReg::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPTanhReg::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPTanhReg::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPTanhReg::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPTanhReg::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_reg"), &MLPPTanhReg::get_reg);
	ClassDB::bind_method(D_METHOD("set_reg", "val"), &MLPPTanhReg::set_reg);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reg"), "set_reg", "get_reg");

	ClassDB::bind_method(D_METHOD("get_lambda"), &MLPPTanhReg::get_lambda);
	ClassDB::bind_method(D_METHOD("set_lambda", "val"), &MLPPTanhReg::set_lambda);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lambda"), "set_lambda", "get_lambda");

	ClassDB::bind_method(D_METHOD("get_alpha"), &MLPPTanhReg::get_alpha);
	ClassDB::bind_method(D_METHOD("set_alpha", "val"), &MLPPTanhReg::set_alpha);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alpha"), "set_alpha", "get_alpha");

	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPTanhReg::model_test);
	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPTanhReg::model_set_test);

	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPTanhReg::gradient_descent, false);
	ClassDB::bind_method(D_METHOD("sgd", "learning_rate", "max_epoch", "ui"), &MLPPTanhReg::sgd, false);
	ClassDB::bind_method(D_METHOD("mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPTanhReg::mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPTanhReg::score);

	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPTanhReg::save);

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPTanhReg::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPTanhReg::initialize);
}
