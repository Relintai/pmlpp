//
//  ExpReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "exp_reg.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../stat/stat.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

std::vector<real_t> MLPPExpReg::model_set_test(std::vector<std::vector<real_t>> X) {
	return evaluatem(X);
}

real_t MLPPExpReg::model_test(std::vector<real_t> x) {
	return evaluatev(x);
}

void MLPPExpReg::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		std::vector<real_t> error = alg.subtraction(_y_hat, _output_set);

		for (int i = 0; i < _k; i++) {
			// Calculating the weight gradient
			real_t sum = 0;
			for (int j = 0; j < _n; j++) {
				sum += error[j] * _input_set[j][i] * std::pow(_weights[i], _input_set[j][i] - 1);
			}
			real_t w_gradient = sum / _n;

			// Calculating the initial gradient
			real_t sum2 = 0;
			for (int j = 0; j < _n; j++) {
				sum2 += error[j] * std::pow(_weights[i], _input_set[j][i]);
			}

			real_t i_gradient = sum2 / _n;

			// Weight/initial updation
			_weights[i] -= learning_rate * w_gradient;
			_initial[i] -= learning_rate * i_gradient;
		}

		_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradient
		real_t sum = 0;
		for (int j = 0; j < _n; j++) {
			sum += (_y_hat[j] - _output_set[j]);
		}
		real_t b_gradient = sum / _n;

		// bias updation
		_bias -= learning_rate * b_gradient;

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

void MLPPExpReg::sgd(real_t learning_rate, int max_epoch, bool ui) {
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(_n - 1));

	while (true) {
		int output_index = distribution(generator);

		real_t y_hat = evaluatev(_input_set[output_index]);
		cost_prev = cost({ y_hat }, { _output_set[output_index] });

		for (int i = 0; i < _k; i++) {
			// Calculating the weight gradients

			real_t w_gradient = (y_hat - _output_set[output_index]) * _input_set[output_index][i] * std::pow(_weights[i], _input_set[output_index][i] - 1);
			real_t i_gradient = (y_hat - _output_set[output_index]) * std::pow(_weights[i], _input_set[output_index][i]);

			// Weight/initial updation
			_weights[i] -= learning_rate * w_gradient;
			_initial[i] -= learning_rate * i_gradient;
		}

		_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		real_t b_gradient = (y_hat - _output_set[output_index]);

		// Bias updation
		_bias -= learning_rate * b_gradient;
		y_hat = evaluatev(_input_set[output_index]);

		if (ui) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost({ y_hat }, { _output_set[output_index] }));
			MLPPUtilities::UI(_weights, _bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPExpReg::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = evaluatem(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);
			std::vector<real_t> error = alg.subtraction(y_hat, output_mini_batches[i]);

			for (int j = 0; j < _k; j++) {
				// Calculating the weight gradient
				real_t sum = 0;
				for (uint32_t k = 0; k < output_mini_batches[i].size(); k++) {
					sum += error[k] * input_mini_batches[i][k][j] * std::pow(_weights[j], input_mini_batches[i][k][j] - 1);
				}
				real_t w_gradient = sum / output_mini_batches[i].size();

				// Calculating the initial gradient
				real_t sum2 = 0;
				for (uint32_t k = 0; k < output_mini_batches[i].size(); k++) {
					sum2 += error[k] * std::pow(_weights[j], input_mini_batches[i][k][j]);
				}

				real_t i_gradient = sum2 / output_mini_batches[i].size();

				// Weight/initial updation
				_weights[j] -= learning_rate * w_gradient;
				_initial[j] -= learning_rate * i_gradient;
			}

			_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradient
			real_t sum = 0;
			for (uint32_t j = 0; j < output_mini_batches[i].size(); j++) {
				sum += (y_hat[j] - output_mini_batches[i][j]);
			}

			//real_t b_gradient = sum / output_mini_batches[i].size();
			y_hat = evaluatem(input_mini_batches[i]);

			if (ui) {
				MLPPUtilities::CostInfo(epoch, cost_prev, cost(y_hat, output_mini_batches[i]));
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

real_t MLPPExpReg::score() {
	MLPPUtilities util;

	return util.performance(_y_hat, _output_set);
}

void MLPPExpReg::save(std::string file_name) {
	MLPPUtilities util;

	util.saveParameters(file_name, _weights, _initial, _bias);
}

MLPPExpReg::MLPPExpReg(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, std::string p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = p_input_set.size();
	_k = p_input_set[0].size();
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.resize(_n);
	_weights = MLPPUtilities::weightInitialization(_k);
	_initial = MLPPUtilities::weightInitialization(_k);
	_bias = MLPPUtilities::biasInitialization();
}

MLPPExpReg::MLPPExpReg() {
}
MLPPExpReg::~MLPPExpReg() {
}

real_t MLPPExpReg::cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;

	return mlpp_cost.MSE(y_hat, y) + regularization.regTerm(_weights, _lambda, _alpha, _reg);
}

real_t MLPPExpReg::evaluatev(std::vector<real_t> x) {
	real_t y_hat = 0;

	for (uint32_t i = 0; i < x.size(); i++) {
		y_hat += _initial[i] * std::pow(_weights[i], x[i]);
	}

	return y_hat + _bias;
}

std::vector<real_t> MLPPExpReg::evaluatem(std::vector<std::vector<real_t>> X) {
	std::vector<real_t> y_hat;
	y_hat.resize(X.size());

	for (uint32_t i = 0; i < X.size(); i++) {
		y_hat[i] = 0;
		for (uint32_t j = 0; j < X[i].size(); j++) {
			y_hat[i] += _initial[j] * std::pow(_weights[j], X[i][j]);
		}
		y_hat[i] += _bias;
	}

	return y_hat;
}

// a * w^x + b
void MLPPExpReg::forward_pass() {
	_y_hat = evaluatem(_input_set);
}

void MLPPExpReg::_bind_methods() {
}
