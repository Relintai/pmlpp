//
//  LogReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "log_reg.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
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

std::vector<real_t> MLPPLogReg::model_set_test(std::vector<std::vector<real_t>> X) {
	ERR_FAIL_COND_V(!_initialized, std::vector<real_t>());

	return evaluatem(X);
}

real_t MLPPLogReg::model_test(std::vector<real_t> x) {
	ERR_FAIL_COND_V(!_initialized, 0);

	return evaluatev(x);
}

void MLPPLogReg::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		std::vector<real_t> error = alg.subtraction(_y_hat, _output_set);

		// Calculating the weight gradients
		_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate / _n, alg.mat_vec_mult(alg.transpose(_input_set), error)));
		_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		_bias -= learning_rate * alg.sum_elements(error) / _n;
		forward_pass();

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

void MLPPLogReg::mle(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		std::vector<real_t> error = alg.subtraction(_output_set, _y_hat);

		// Calculating the weight gradients
		_weights = alg.addition(_weights, alg.scalarMultiply(learning_rate / _n, alg.mat_vec_mult(alg.transpose(_input_set), error)));
		_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		_bias += learning_rate * alg.sum_elements(error) / _n;

		forward_pass();

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

void MLPPLogReg::sgd(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

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
		cost_prev = cost({ y_hat }, { _output_set[outputIndex] });

		real_t error = y_hat - _output_set[outputIndex];

		// Weight updation
		_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate * error, _input_set[outputIndex]));
		_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

		// Bias updation
		_bias -= learning_rate * error;

		y_hat = evaluatev(_input_set[outputIndex]);

		if (ui) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost({ y_hat }, { _output_set[outputIndex] }));
			MLPPUtilities::UI(_weights, _bias);
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

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	auto bacthes = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto inputMiniBatches = std::get<0>(bacthes);
	auto outputMiniBatches = std::get<1>(bacthes);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = evaluatem(inputMiniBatches[i]);
			cost_prev = cost(y_hat, outputMiniBatches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate / outputMiniBatches[i].size(), alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), error)));
			_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size();
			y_hat = evaluatem(inputMiniBatches[i]);

			if (UI) {
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

real_t MLPPLogReg::score() {
	ERR_FAIL_COND_V(!_initialized, 0);

	MLPPUtilities util;
	return util.performance(_y_hat, _output_set);
}

void MLPPLogReg::save(std::string file_name) {
	ERR_FAIL_COND(!_initialized);

	MLPPUtilities util;
	util.saveParameters(file_name, _weights, _bias);
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

MLPPLogReg::MLPPLogReg(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, std::string p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = p_input_set.size();
	_k = p_input_set[0].size();
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.resize(_n);
	_weights = MLPPUtilities::weightInitialization(_k);
	_bias = MLPPUtilities::biasInitialization();

	_initialized = true;
}

MLPPLogReg::MLPPLogReg() {
	_initialized = false;
}
MLPPLogReg::~MLPPLogReg() {
}

real_t MLPPLogReg::cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPReg regularization;
	class MLPPCost cost;

	return cost.LogLoss(y_hat, y) + regularization.regTerm(_weights, _lambda, _alpha, _reg);
}

real_t MLPPLogReg::evaluatev(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	return avn.sigmoid(alg.dot(_weights, x) + _bias);
}

std::vector<real_t> MLPPLogReg::evaluatem(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	return avn.sigmoid(alg.scalarAdd(_bias, alg.mat_vec_mult(X, _weights)));
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
