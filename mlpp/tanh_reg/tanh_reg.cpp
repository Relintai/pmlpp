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

#include <iostream>
#include <random>

/*
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

	_initialized = false;
}

real_t MLPPTanhReg::get_lambda() {
	return _lambda;
}
void MLPPTanhReg::set_lambda(const real_t val) {
	_lambda = val;

	_initialized = false;
}

real_t MLPPTanhReg::get_alpha() {
	return _alpha;
}
void MLPPTanhReg::set_alpha(const real_t val) {
	_alpha = val;

	_initialized = false;
}
*/

std::vector<real_t> MLPPTanhReg::model_set_test(std::vector<std::vector<real_t>> X) {
	return evaluatem(X);
}

real_t MLPPTanhReg::model_test(std::vector<real_t> x) {
	return evaluatev(x);
}

void MLPPTanhReg::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		std::vector<real_t> error = alg.subtraction(_y_hat, _output_set);

		_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate / _n, alg.mat_vec_mult(alg.transpose(_input_set), alg.hadamard_product(error, avn.tanh(_z, 1)))));
		//_reg
		_weights = regularization.regWeights(_weights, _lambda, _alpha, "None");

		// Calculating the bias gradients
		_bias -= learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.tanh(_z, 1))) / _n;

		forward_pass();

		// UI PORTION
		if (ui) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost(_y_hat, _output_set));
			MLPPUtilities::UI(_weights, _bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPTanhReg::sgd(real_t learning_rate, int max_epoch, bool ui) {
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(_n - 1));

	while (true) {
		int outputIndex = distribution(generator);

		real_t y_hat = evaluatev(_input_set[outputIndex]);
		cost_prev = cost({ _y_hat }, { _output_set[outputIndex] });

		real_t error = y_hat - _output_set[outputIndex];

		// Weight Updation
		_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate * error * (1 - y_hat * y_hat), _input_set[outputIndex]));
		//_reg
		_weights = regularization.regWeights(_weights, _lambda, _alpha, "None");

		// Bias updation
		_bias -= learning_rate * error * (1 - y_hat * y_hat);

		y_hat = evaluatev(_input_set[outputIndex]);

		if (ui) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost({ _y_hat }, { _output_set[outputIndex] }));
			MLPPUtilities::UI(_weights, _bias);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPTanhReg::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto inputMiniBatches = std::get<0>(batches);
	auto outputMiniBatches = std::get<1>(batches);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = evaluatem(inputMiniBatches[i]);
			std::vector<real_t> z = propagatem(inputMiniBatches[i]);
			cost_prev = cost(y_hat, outputMiniBatches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate / _n, alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), alg.hadamard_product(error, avn.tanh(z, 1)))));
			//_reg
			_weights = regularization.regWeights(_weights, _lambda, _alpha, "None");

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.tanh(_z, true))) / _n;

			forward_pass();

			y_hat = evaluatem(inputMiniBatches[i]);

			if (ui) {
				MLPPUtilities::CostInfo(epoch, cost_prev, cost(y_hat, outputMiniBatches[i]));
				MLPPUtilities::UI(_weights, _bias);
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
	MLPPUtilities util;

	return util.performance(_y_hat, _output_set);
}

void MLPPTanhReg::save(std::string file_name) {
	MLPPUtilities util;

	util.saveParameters(file_name, _weights, _bias);
}

bool MLPPTanhReg::is_initialized() {
	return _initialized;
}
void MLPPTanhReg::initialize() {
	if (_initialized) {
		return;
	}

	//ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	_initialized = true;
}

MLPPTanhReg::MLPPTanhReg(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = _input_set.size();
	_k = _input_set[0].size();
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.resize(_n);
	_weights = MLPPUtilities::weightInitialization(_k);
	_bias = MLPPUtilities::biasInitialization();
}

MLPPTanhReg::MLPPTanhReg() {
}
MLPPTanhReg::~MLPPTanhReg() {
}

real_t MLPPTanhReg::cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPReg regularization;
	class MLPPCost cost;

	//_reg
	return cost.MSE(y_hat, y) + regularization.regTerm(_weights, _lambda, _alpha, "None");
}

real_t MLPPTanhReg::evaluatev(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.tanh(alg.dot(_weights, x) + _bias);
}

real_t MLPPTanhReg::propagatev(std::vector<real_t> x) {
	MLPPLinAlg alg;
	return alg.dot(_weights, x) + _bias;
}

std::vector<real_t> MLPPTanhReg::evaluatem(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.tanh(alg.scalarAdd(_bias, alg.mat_vec_mult(X, _weights)));
}

std::vector<real_t> MLPPTanhReg::propagatem(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	return alg.scalarAdd(_bias, alg.mat_vec_mult(X, _weights));
}

// Tanh ( wTx + b )
void MLPPTanhReg::forward_pass() {
	MLPPActivation avn;

	_z = propagatem(_input_set);
	_y_hat = avn.tanh(_z);
}

void MLPPTanhReg::_bind_methods() {
	/*
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
	*/
}
