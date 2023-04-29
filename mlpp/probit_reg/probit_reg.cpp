//
//  ProbitReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "probit_reg.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <random>

Ref<MLPPMatrix> MLPPProbitReg::get_input_set() {
	return _input_set;
}
void MLPPProbitReg::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;

	_initialized = false;
}

Ref<MLPPVector> MLPPProbitReg::get_output_set() {
	return _output_set;
}
void MLPPProbitReg::set_output_set(const Ref<MLPPVector> &val) {
	_output_set = val;

	_initialized = false;
}

MLPPReg::RegularizationType MLPPProbitReg::get_reg() {
	return _reg;
}
void MLPPProbitReg::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;

	_initialized = false;
}

real_t MLPPProbitReg::get_lambda() {
	return _lambda;
}
void MLPPProbitReg::set_lambda(const real_t val) {
	_lambda = val;

	_initialized = false;
}

real_t MLPPProbitReg::get_alpha() {
	return _alpha;
}
void MLPPProbitReg::set_alpha(const real_t val) {
	_alpha = val;

	_initialized = false;
}

Ref<MLPPVector> MLPPProbitReg::model_set_test(const Ref<MLPPMatrix> &X) {
	return evaluatem(X);
}

real_t MLPPProbitReg::model_test(const Ref<MLPPVector> &x) {
	return evaluatev(x);
}

void MLPPProbitReg::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPVector> error = alg.subtractionnv(_y_hat, _output_set);

		// Calculating the weight gradients
		_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate / _n, alg.mat_vec_multnv(alg.transposenm(_input_set), alg.hadamard_productnv(error, avn.gaussian_cdf_derivv(_z)))));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		_bias -= learning_rate * alg.sum_elementsv(alg.hadamard_productnv(error, avn.gaussian_cdf_derivv(_z))) / _n;

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

void MLPPProbitReg::mle(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPVector> error = alg.subtractionnv(_output_set, _y_hat);

		// Calculating the weight gradients
		_weights = alg.additionnv(_weights, alg.scalar_multiplynv(learning_rate / _n, alg.mat_vec_multnv(alg.transposenm(_input_set), alg.hadamard_productnv(error, avn.gaussian_cdf_derivv(_z)))));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		_bias += learning_rate * alg.sum_elementsv(alg.hadamard_productnv(error, avn.gaussian_cdf_derivv(_z))) / _n;

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

void MLPPProbitReg::sgd(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	// NOTE: ∂y_hat/∂z is sparse
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

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
	std::uniform_int_distribution<int> distribution(0, int(_n - 1));

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
		_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate * error * ((1 / Math::sqrt(2 * Math_PI)) * Math::exp(-z * z / 2)), input_set_row_tmp));
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

void MLPPProbitReg::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	Ref<MLPPVector> z_tmp;
	z_tmp.instance();
	z_tmp->resize(1);

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;

	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input = batches.input_sets[i];
			Ref<MLPPVector> current_output = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input);
			real_t z = propagatev(current_output);

			z_tmp->element_set(0, z);

			cost_prev = cost(y_hat, current_output);

			Ref<MLPPVector> error = alg.subtractionnv(y_hat, current_output);

			// Calculating the weight gradients
			_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate / batches.input_sets.size(), alg.mat_vec_multnv(alg.transposenm(current_input), alg.hadamard_productnv(error, avn.gaussian_cdf_derivv(z_tmp)))));
			_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elementsv(alg.hadamard_productnv(error, avn.gaussian_cdf_derivv(z_tmp))) / batches.input_sets.size();
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
	MLPPUtilities util;

	return util.performance_vec(_y_hat, _output_set);
}

void MLPPProbitReg::save(const String &file_name) {
	MLPPUtilities util;

	//util.saveParameters(file_name, _weights, _bias);
}

bool MLPPProbitReg::is_initialized() {
	return _initialized;
}
void MLPPProbitReg::initialize() {
	if (_initialized) {
		return;
	}

	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	_n = _input_set->size().y;
	_k = _input_set->size().x;

	if (!_y_hat.is_valid()) {
		_y_hat.instance();
	}

	_y_hat->resize(_n);

	MLPPUtilities util;

	if (!_weights.is_valid()) {
		_weights.instance();
	}

	_weights->resize(_k);

	util.weight_initializationv(_weights);
	_bias = util.bias_initializationr();

	_initialized = true;
}

MLPPProbitReg::MLPPProbitReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;

	_n = _input_set->size().y;
	_k = _input_set->size().x;

	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.instance();
	_y_hat->resize(_n);

	MLPPUtilities util;

	_weights.instance();
	_weights->resize(_k);

	util.weight_initializationv(_weights);
	_bias = util.bias_initializationr();

	_initialized = true;
}

MLPPProbitReg::MLPPProbitReg() {
	_y_hat.instance();

	_bias = 0;

	_n = 0;
	_k = 0;

	// Regularization Params
	_reg = MLPPReg::REGULARIZATION_TYPE_NONE;
	_lambda = 0.5;
	_alpha = 0.5;

	_initialized = false;
}
MLPPProbitReg::~MLPPProbitReg() {
}

real_t MLPPProbitReg::cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPReg regularization;
	class MLPPCost cost;

	return cost.msev(y_hat, y) + regularization.reg_termv(_weights, _lambda, _alpha, _reg);
}

Ref<MLPPVector> MLPPProbitReg::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	return avn.gaussian_cdf_normv(alg.scalar_addnv(_bias, alg.mat_vec_multnv(X, _weights)));
}

Ref<MLPPVector> MLPPProbitReg::propagatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;

	return alg.scalar_addnv(_bias, alg.mat_vec_multnv(X, _weights));
}

real_t MLPPProbitReg::evaluatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	return avn.gaussian_cdf_normr(alg.dotnv(_weights, x) + _bias);
}

real_t MLPPProbitReg::propagatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;

	return alg.dotnv(_weights, x) + _bias;
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

	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPProbitReg::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPProbitReg::model_test);

	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPProbitReg::gradient_descent, 0, false);
	ClassDB::bind_method(D_METHOD("mle", "learning_rate", "max_epoch", "ui"), &MLPPProbitReg::mle, 0, false);
	ClassDB::bind_method(D_METHOD("sgd", "learning_rate", "max_epoch", "ui"), &MLPPProbitReg::sgd, 0, false);
	ClassDB::bind_method(D_METHOD("mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPProbitReg::mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPProbitReg::score);

	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPProbitReg::save);

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPProbitReg::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPProbitReg::initialize);
}
