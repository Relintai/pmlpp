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

Ref<MLPPVector> MLPPExpReg::model_set_test(const Ref<MLPPMatrix> &X) {
	return evaluatem(X);
}

real_t MLPPExpReg::model_test(const Ref<MLPPVector> &x) {
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

		Ref<MLPPVector> error = alg.subtractionnv(_y_hat, _output_set);

		for (int i = 0; i < _k; i++) {
			// Calculating the weight gradient
			real_t sum = 0;
			for (int j = 0; j < _n; j++) {
				sum += error->get_element(j) * _input_set->get_element(j, i) * Math::pow(_weights->get_element(i), _input_set->get_element(j, i) - 1);
			}
			real_t w_gradient = sum / _n;

			// Calculating the initial gradient
			real_t sum2 = 0;
			for (int j = 0; j < _n; j++) {
				sum2 += error->get_element(j) * Math::pow(_weights->get_element(i), _input_set->get_element(j, i));
			}

			real_t i_gradient = sum2 / _n;

			// Weight/initial updation
			_weights->set_element(i, _weights->get_element(i) - learning_rate * w_gradient);
			_initial->set_element(i, _initial->get_element(i) - learning_rate * i_gradient);
		}

		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradient
		real_t sum = 0;
		for (int j = 0; j < _n; j++) {
			sum += (_y_hat->get_element(j) - _output_set->get_element(j));
		}
		real_t b_gradient = sum / _n;

		// bias updation
		_bias -= learning_rate * b_gradient;

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

void MLPPExpReg::sgd(real_t learning_rate, int max_epoch, bool ui) {
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(_n - 1));

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

		_input_set->get_row_into_mlpp_vector(output_index, input_set_row_tmp);
		real_t output_set_element = _output_set->get_element(output_index);
		output_set_row_tmp->set_element(0, output_set_element);

		real_t y_hat = evaluatev(input_set_row_tmp);
		y_hat_row_tmp->set_element(0, y_hat);

		cost_prev = cost(y_hat_row_tmp, output_set_row_tmp);

		for (int i = 0; i < _k; i++) {
			// Calculating the weight gradients

			real_t w_gradient = (y_hat - output_set_element) * input_set_row_tmp->get_element(i) * Math::pow(_weights->get_element(i), _input_set->get_element(output_index, i) - 1);
			real_t i_gradient = (y_hat - output_set_element) * Math::pow(_weights->get_element(i), _input_set->get_element(output_index, i));

			// Weight/initial updation
			_weights->set_element(i, _weights->get_element(i) - learning_rate * w_gradient);
			_initial->set_element(i, _initial->get_element(i) - learning_rate * i_gradient);
		}

		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		real_t b_gradient = (y_hat - output_set_element);

		// Bias updation
		_bias -= learning_rate * b_gradient;
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

void MLPPExpReg::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	MLPPLinAlg alg;
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
			cost_prev = cost(y_hat, current_output_batch);
			Ref<MLPPVector> error = alg.subtractionnv(y_hat, current_output_batch);

			for (int j = 0; j < _k; j++) {
				// Calculating the weight gradient
				real_t sum = 0;
				for (int k = 0; k < current_output_batch->size(); k++) {
					sum += error->get_element(k) * current_input_batch->get_element(k, j) * Math::pow(_weights->get_element(j), current_input_batch->get_element(k, j) - 1);
				}
				real_t w_gradient = sum / current_output_batch->size();

				// Calculating the initial gradient
				real_t sum2 = 0;
				for (int k = 0; k < current_output_batch->size(); k++) {
					sum2 += error->get_element(k) * Math::pow(_weights->get_element(j), current_input_batch->get_element(k, j));
				}

				real_t i_gradient = sum2 / current_output_batch->size();

				// Weight/initial updation
				_weights->set_element(i, _weights->get_element(i) - learning_rate * w_gradient);
				_initial->set_element(i, _initial->get_element(i) - learning_rate * i_gradient);
			}

			_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradient
			real_t sum = 0;
			for (int j = 0; j < current_output_batch->size(); j++) {
				sum += (y_hat->get_element(j) - current_output_batch->get_element(j));
			}

			//real_t b_gradient = sum / output_mini_batches[i].size();
			y_hat = evaluatem(current_input_batch);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_batch));
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

real_t MLPPExpReg::score() {
	MLPPUtilities util;

	return util.performance_vec(_y_hat, _output_set);
}

void MLPPExpReg::save(const String &file_name) {
	MLPPUtilities util;

	//util.saveParameters(file_name, _weights, _initial, _bias);
}

MLPPExpReg::MLPPExpReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = p_input_set->size().y;
	_k = p_input_set->size().x;
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.instance();
	_y_hat->resize(_n);

	MLPPUtilities util;

	_weights.instance();
	_weights->resize(_k);

	util.weight_initializationv(_weights);

	_initial.instance();
	_initial->resize(_k);

	util.weight_initializationv(_initial);

	_bias = util.bias_initializationr();
}

MLPPExpReg::MLPPExpReg() {
}
MLPPExpReg::~MLPPExpReg() {
}

real_t MLPPExpReg::cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;

	return mlpp_cost.msev(y_hat, y) + regularization.reg_termv(_weights, _lambda, _alpha, _reg);
}

real_t MLPPExpReg::evaluatev(const Ref<MLPPVector> &x) {
	real_t y_hat = 0;

	for (int i = 0; i < x->size(); i++) {
		y_hat += _initial->get_element(i) * Math::pow(_weights->get_element(i), x->get_element(i));
	}

	return y_hat + _bias;
}

Ref<MLPPVector> MLPPExpReg::evaluatem(const Ref<MLPPMatrix> &X) {
	Ref<MLPPVector> y_hat;
	y_hat.instance();
	y_hat->resize(X->size().y);

	for (int i = 0; i < X->size().y; i++) {
		real_t y;

		for (int j = 0; j < X->size().x; j++) {
			y += _initial->get_element(j) * Math::pow(_weights->get_element(j), X->get_element(i, j));
		}

		y += _bias;

		y_hat->set_element(i, y);
	}

	return y_hat;
}

// a * w^x + b
void MLPPExpReg::forward_pass() {
	_y_hat = evaluatem(_input_set);
}

void MLPPExpReg::_bind_methods() {
}
