//
//  LinReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "lin_reg.h"

#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../stat/stat.h"
#include "../utilities/utilities.h"

#include <cmath>
#include <iostream>
#include <random>

/*
Ref<MLPPMatrix> MLPPLinReg::get_input_set() {
	return _input_set;
}
void MLPPLinReg::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;

	_initialized = false;
}

Ref<MLPPVector> MLPPLinReg::get_output_set() {
	return _output_set;
}
void MLPPLinReg::set_output_set(const Ref<MLPPVector> &val) {
	_output_set = val;

	_initialized = false;
}

MLPPReg::RegularizationType MLPPLinReg::get_reg() {
	return _reg;
}
void MLPPLinReg::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;

	_initialized = false;
}

real_t MLPPLinReg::get_lambda() {
	return _lambda;
}
void MLPPLinReg::set_lambda(const real_t val) {
	_lambda = val;

	_initialized = false;
}

real_t MLPPLinReg::get_alpha() {
	return _alpha;
}
void MLPPLinReg::set_alpha(const real_t val) {
	_alpha = val;

	_initialized = false;
}
*/

std::vector<real_t> MLPPLinReg::model_set_test(std::vector<std::vector<real_t>> X) {
	ERR_FAIL_COND_V(!_initialized, std::vector<real_t>());

	return evaluatem(X);
}

real_t MLPPLinReg::model_test(std::vector<real_t> x) {
	ERR_FAIL_COND_V(!_initialized, 0);

	return evaluatev(x);
}

void MLPPLinReg::newton_raphson(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		std::vector<real_t> error = alg.subtraction(_y_hat, _output_set);

		// Calculating the weight gradients (2nd derivative)
		std::vector<real_t> first_derivative = alg.mat_vec_mult(alg.transpose(_input_set), error);
		std::vector<std::vector<real_t>> second_derivative = alg.matmult(alg.transpose(_input_set), _input_set);
		_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate / _n, alg.mat_vec_mult(alg.transpose(alg.inverse(second_derivative)), first_derivative)));
		_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients (2nd derivative)
		_bias -= learning_rate * alg.sum_elements(error) / _n; // We keep this the same. The 2nd derivative is just [1].

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

void MLPPLinReg::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
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

void MLPPLinReg::sgd(real_t learning_rate, int max_epoch, bool ui) {
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

void MLPPLinReg::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	ERR_FAIL_COND(!_initialized);

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

			// Calculating the weight gradients
			_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate / output_mini_batches[i].size(), alg.mat_vec_mult(alg.transpose(input_mini_batches[i]), error)));
			_weights = regularization.regWeights(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elements(error) / output_mini_batches[i].size();
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

void MLPPLinReg::momentum(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Momentum.
	std::vector<real_t> v = alg.zerovec(_weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = evaluatem(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, output_mini_batches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / output_mini_batches[i].size(), alg.mat_vec_mult(alg.transpose(input_mini_batches[i]), error));
			std::vector<real_t> reg_deriv_term = regularization.regDerivTerm(_weights, _lambda, _alpha, _reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, reg_deriv_term); // Weight_grad_final

			v = alg.addition(alg.scalarMultiply(gamma, v), alg.scalarMultiply(learning_rate, weight_grad));

			_weights = alg.subtraction(_weights, v);

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elements(error) / output_mini_batches[i].size(); // As normal
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

void MLPPLinReg::nag(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Momentum.
	std::vector<real_t> v = alg.zerovec(_weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			_weights = alg.subtraction(_weights, alg.scalarMultiply(gamma, v)); // "Aposterori" calculation

			std::vector<real_t> y_hat = evaluatem(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, output_mini_batches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / output_mini_batches[i].size(), alg.mat_vec_mult(alg.transpose(input_mini_batches[i]), error));
			std::vector<real_t> reg_deriv_term = regularization.regDerivTerm(_weights, _lambda, _alpha, _reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, reg_deriv_term); // Weight_grad_final

			v = alg.addition(alg.scalarMultiply(gamma, v), alg.scalarMultiply(learning_rate, weight_grad));

			_weights = alg.subtraction(_weights, v);

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elements(error) / output_mini_batches[i].size(); // As normal
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

void MLPPLinReg::adagrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t e, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Adagrad.
	std::vector<real_t> v = alg.zerovec(_weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = evaluatem(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, output_mini_batches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / output_mini_batches[i].size(), alg.mat_vec_mult(alg.transpose(input_mini_batches[i]), error));
			std::vector<real_t> reg_deriv_term = regularization.regDerivTerm(_weights, _lambda, _alpha, _reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, reg_deriv_term); // Weight_grad_final

			v = alg.hadamard_product(weight_grad, weight_grad);

			_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate, alg.elementWiseDivision(weight_grad, alg.sqrt(alg.scalarAdd(e, v)))));

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elements(error) / output_mini_batches[i].size(); // As normal
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

void MLPPLinReg::adadelta(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t e, bool ui) {
	ERR_FAIL_COND(!_initialized);

	// Adagrad upgrade. Momentum is applied.
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Adagrad.
	std::vector<real_t> v = alg.zerovec(_weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = evaluatem(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, output_mini_batches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / output_mini_batches[i].size(), alg.mat_vec_mult(alg.transpose(input_mini_batches[i]), error));
			std::vector<real_t> reg_deriv_term = regularization.regDerivTerm(_weights, _lambda, _alpha, _reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, reg_deriv_term); // Weight_grad_final

			v = alg.addition(alg.scalarMultiply(b1, v), alg.scalarMultiply(1 - b1, alg.hadamard_product(weight_grad, weight_grad)));

			_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate, alg.elementWiseDivision(weight_grad, alg.sqrt(alg.scalarAdd(e, v)))));

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elements(error) / output_mini_batches[i].size(); // As normal
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

void MLPPLinReg::adam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<real_t> m = alg.zerovec(_weights.size());

	std::vector<real_t> v = alg.zerovec(_weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = evaluatem(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, output_mini_batches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / output_mini_batches[i].size(), alg.mat_vec_mult(alg.transpose(input_mini_batches[i]), error));
			std::vector<real_t> reg_deriv_term = regularization.regDerivTerm(_weights, _lambda, _alpha, _reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, reg_deriv_term); // Weight_grad_final

			m = alg.addition(alg.scalarMultiply(b1, m), alg.scalarMultiply(1 - b1, weight_grad));
			v = alg.addition(alg.scalarMultiply(b2, v), alg.scalarMultiply(1 - b2, alg.exponentiate(weight_grad, 2)));

			std::vector<real_t> m_hat = alg.scalarMultiply(1 / (1 - pow(b1, epoch)), m);
			std::vector<real_t> v_hat = alg.scalarMultiply(1 / (1 - pow(b2, epoch)), v);

			_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate, alg.elementWiseDivision(m_hat, alg.scalarAdd(e, alg.sqrt(v_hat)))));

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elements(error) / output_mini_batches[i].size(); // As normal
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

void MLPPLinReg::adamax(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	std::vector<real_t> m = alg.zerovec(_weights.size());

	std::vector<real_t> u = alg.zerovec(_weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = evaluatem(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, output_mini_batches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / output_mini_batches[i].size(), alg.mat_vec_mult(alg.transpose(input_mini_batches[i]), error));
			std::vector<real_t> reg_deriv_term = regularization.regDerivTerm(_weights, _lambda, _alpha, _reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, reg_deriv_term); // Weight_grad_final

			m = alg.addition(alg.scalarMultiply(b1, m), alg.scalarMultiply(1 - b1, weight_grad));
			u = alg.max(alg.scalarMultiply(b2, u), alg.abs(weight_grad));

			std::vector<real_t> m_hat = alg.scalarMultiply(1 / (1 - pow(b1, epoch)), m);

			_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate, alg.elementWiseDivision(m_hat, u)));

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elements(error) / output_mini_batches[i].size(); // As normal
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

void MLPPLinReg::nadam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<real_t> m = alg.zerovec(_weights.size());
	std::vector<real_t> v = alg.zerovec(_weights.size());
	std::vector<real_t> m_final = alg.zerovec(_weights.size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = evaluatem(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, output_mini_batches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / output_mini_batches[i].size(), alg.mat_vec_mult(alg.transpose(input_mini_batches[i]), error));
			std::vector<real_t> reg_deriv_term = regularization.regDerivTerm(_weights, _lambda, _alpha, _reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, reg_deriv_term); // Weight_grad_final

			m = alg.addition(alg.scalarMultiply(b1, m), alg.scalarMultiply(1 - b1, weight_grad));
			v = alg.addition(alg.scalarMultiply(b2, v), alg.scalarMultiply(1 - b2, alg.exponentiate(weight_grad, 2)));
			m_final = alg.addition(alg.scalarMultiply(b1, m), alg.scalarMultiply((1 - b1) / (1 - pow(b1, epoch)), weight_grad));

			std::vector<real_t> m_hat = alg.scalarMultiply(1 / (1 - pow(b1, epoch)), m);
			std::vector<real_t> v_hat = alg.scalarMultiply(1 / (1 - pow(b2, epoch)), v);

			_weights = alg.subtraction(_weights, alg.scalarMultiply(learning_rate, alg.elementWiseDivision(m_final, alg.scalarAdd(e, alg.sqrt(v_hat)))));

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elements(error) / output_mini_batches[i].size(); // As normal
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

void MLPPLinReg::normal_equation() {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPStat stat;
	std::vector<real_t> x_means;
	std::vector<std::vector<real_t>> _input_setT = alg.transpose(_input_set);

	x_means.resize(_input_setT.size());
	for (uint32_t i = 0; i < _input_setT.size(); i++) {
		x_means[i] = (stat.mean(_input_setT[i]));
	}

	std::vector<real_t> temp;
	temp.resize(_k);
	temp = alg.mat_vec_mult(alg.inverse(alg.matmult(alg.transpose(_input_set), _input_set)), alg.mat_vec_mult(alg.transpose(_input_set), _output_set));

	ERR_FAIL_COND_MSG(std::isnan(temp[0]), "ERR: Resulting matrix was noninvertible/degenerate, and so the normal equation could not be performed. Try utilizing gradient descent.");

	if (_reg == "Ridge") {
		_weights = alg.mat_vec_mult(alg.inverse(alg.addition(alg.matmult(alg.transpose(_input_set), _input_set), alg.scalarMultiply(_lambda, alg.identity(_k)))), alg.mat_vec_mult(alg.transpose(_input_set), _output_set));
	} else {
		_weights = alg.mat_vec_mult(alg.inverse(alg.matmult(alg.transpose(_input_set), _input_set)), alg.mat_vec_mult(alg.transpose(_input_set), _output_set));
	}

	_bias = stat.mean(_output_set) - alg.dot(_weights, x_means);

	forward_pass();
}

real_t MLPPLinReg::score() {
	ERR_FAIL_COND_V(!_initialized, 0);

	MLPPUtilities util;

	return util.performance(_y_hat, _output_set);
}

void MLPPLinReg::save(std::string fileName) {
	ERR_FAIL_COND(!_initialized);

	MLPPUtilities util;

	util.saveParameters(fileName, _weights, _bias);
}

bool MLPPLinReg::is_initialized() {
	return _initialized;
}
void MLPPLinReg::initialize() {
	if (_initialized) {
		return;
	}

	//ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	_initialized = true;
}

MLPPLinReg::MLPPLinReg(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, std::string p_reg, real_t p_lambda, real_t p_alpha) {
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

MLPPLinReg::MLPPLinReg() {
	_initialized = false;
}
MLPPLinReg::~MLPPLinReg() {
}

real_t MLPPLinReg::cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;

	return mlpp_cost.MSE(y_hat, y) + regularization.regTerm(_weights, _lambda, _alpha, _reg);
}

real_t MLPPLinReg::evaluatev(std::vector<real_t> x) {
	MLPPLinAlg alg;

	return alg.dot(_weights, x) + _bias;
}

std::vector<real_t> MLPPLinReg::evaluatem(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;

	return alg.scalarAdd(_bias, alg.mat_vec_mult(X, _weights));
}

// wTx + b
void MLPPLinReg::forward_pass() {
	_y_hat = evaluatem(_input_set);
}

void MLPPLinReg::_bind_methods() {
	/*
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPLinReg::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPLinReg::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPLinReg::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPLinReg::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_reg"), &MLPPLinReg::get_reg);
	ClassDB::bind_method(D_METHOD("set_reg", "val"), &MLPPLinReg::set_reg);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reg"), "set_reg", "get_reg");

	ClassDB::bind_method(D_METHOD("get_lambda"), &MLPPLinReg::get_lambda);
	ClassDB::bind_method(D_METHOD("set_lambda", "val"), &MLPPLinReg::set_lambda);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lambda"), "set_lambda", "get_lambda");

	ClassDB::bind_method(D_METHOD("get_alpha"), &MLPPLinReg::get_alpha);
	ClassDB::bind_method(D_METHOD("set_alpha", "val"), &MLPPLinReg::set_alpha);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alpha"), "set_alpha", "get_alpha");

	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPLinReg::model_test);
	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPLinReg::model_set_test);

	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPLinReg::gradient_descent, false);
	ClassDB::bind_method(D_METHOD("sgd", "learning_rate", "max_epoch", "ui"), &MLPPLinReg::sgd, false);
	ClassDB::bind_method(D_METHOD("mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPLinReg::mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPLinReg::score);

	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPLinReg::save);

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPLinReg::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPLinReg::initialize);
	*/
}
