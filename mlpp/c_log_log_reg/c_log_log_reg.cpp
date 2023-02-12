//
//  CLogLogReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "c_log_log_reg.h"
#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

std::vector<real_t> MLPPCLogLogReg::model_set_test(std::vector<std::vector<real_t>> X) {
	return evaluatem(X);
}

real_t MLPPCLogLogReg::model_test(std::vector<real_t> x) {
	return evaluatev(x);
}

void MLPPCLogLogReg::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		std::vector<real_t> error = alg.subtraction(_y_hat, _output_set);

		// Calculating the weight gradients
		_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate / _n, alg.mat_vec_mult(alg.transpose(_input_set), alg.hadamard_product(error, avn.cloglog(_z, true)))));
		_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		bias -= learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.cloglog(_z, true))) / _n;

		forward_pass();

		if (ui) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost(_y_hat, _output_set));
			MLPPUtilities::UI(_weights, bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPCLogLogReg::mle(real_t learning_rate, int max_epoch, bool ui) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		std::vector<real_t> error = alg.subtraction(_y_hat, _output_set);

		_weights = alg.addition(_weights, alg.scalarMultiply(learning_rate / _n, alg.mat_vec_mult(alg.transpose(_input_set), alg.hadamard_product(error, avn.cloglog(_z, true)))));
		_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		bias += learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.cloglog(_z, true))) / _n;

		forward_pass();

		if (ui) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost(_y_hat, _output_set));
			MLPPUtilities::UI(_weights, bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPCLogLogReg::sgd(real_t learning_rate, int max_epoch, bool p_) {
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(_n - 1));
		int outputIndex = distribution(generator);

		real_t y_hat = evaluatev(_input_set[outputIndex]);
		real_t z = propagatev(_input_set[outputIndex]);
		cost_prev = cost({ y_hat }, { _output_set[outputIndex] });

		real_t error = y_hat - _output_set[outputIndex];

		// Weight Updation
		_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate * error * exp(z - exp(z)), _input_set[outputIndex]));
		_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

		// Bias updation
		bias -= learning_rate * error * exp(z - exp(z));

		y_hat = evaluatev(_input_set[outputIndex]);

		if (p_) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost({ y_hat }, { _output_set[outputIndex] }));
			MLPPUtilities::UI(_weights, bias);
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
			_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate / _n, alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), alg.hadamard_product(error, avn.cloglog(z, 1)))));
			_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.cloglog(z, 1))) / _n;

			forward_pass();

			y_hat = evaluatem(inputMiniBatches[i]);

			if (p_) {
				MLPPUtilities::CostInfo(epoch, cost_prev, cost(y_hat, outputMiniBatches[i]));
				MLPPUtilities::UI(_weights, bias);
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
	return util.performance(_y_hat, _output_set);
}

MLPPCLogLogReg::MLPPCLogLogReg(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, std::string p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = _input_set.size();
	_k = _input_set[0].size();
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.resize(_n);

	_weights = MLPPUtilities::weightInitialization(_k);
	bias = MLPPUtilities::biasInitialization();
}

MLPPCLogLogReg::MLPPCLogLogReg() {
}
MLPPCLogLogReg::~MLPPCLogLogReg() {
}

real_t MLPPCLogLogReg::cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPReg regularization;
	class MLPPCost cost;
	return cost.MSE(y_hat, y) + regularization.regTerm(_weights, _lambda, _alpha, _reg);
}

real_t MLPPCLogLogReg::evaluatev(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.cloglog(alg.dot(_weights, x) + bias);
}

real_t MLPPCLogLogReg::propagatev(std::vector<real_t> x) {
	MLPPLinAlg alg;
	return alg.dot(_weights, x) + bias;
}

std::vector<real_t> MLPPCLogLogReg::evaluatem(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.cloglog(alg.scalarAdd(bias, alg.mat_vec_mult(X, _weights)));
}

std::vector<real_t> MLPPCLogLogReg::propagatem(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	return alg.scalarAdd(bias, alg.mat_vec_mult(X, _weights));
}

// cloglog ( wTx + b )
void MLPPCLogLogReg::forward_pass() {
	MLPPActivation avn;

	_z = propagatem(_input_set);
	_y_hat = avn.cloglog(_z);
}

void MLPPCLogLogReg::_bind_methods() {
}
