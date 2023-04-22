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

#include <random>

Ref<MLPPVector> MLPPCLogLogReg::model_set_test(const Ref<MLPPMatrix> &X) {
	return evaluatem(X);
}

real_t MLPPCLogLogReg::model_test(const Ref<MLPPVector> &x) {
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

		Ref<MLPPVector> error = alg.subtractionnv(_y_hat, _output_set);

		// Calculating the weight gradients
		_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate / _n, alg.mat_vec_multv(alg.transposenm(_input_set), alg.hadamard_productnv(error, avn.cloglog_derivv(_z)))));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		bias -= learning_rate * alg.sum_elementsv(alg.hadamard_productnv(error, avn.cloglog_derivv(_z))) / _n;

		forward_pass();

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_y_hat, _output_set));
			MLPPUtilities::print_ui_vb(_weights, bias);
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

		Ref<MLPPVector> error = alg.subtractionnv(_y_hat, _output_set);

		_weights = alg.additionnv(_weights, alg.scalar_multiplynv(learning_rate / _n, alg.mat_vec_multv(alg.transposenm(_input_set), alg.hadamard_productnv(error, avn.cloglog_derivv(_z)))));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		bias += learning_rate * alg.sum_elementsv(alg.hadamard_productnv(error, avn.cloglog_derivv(_z))) / _n;

		forward_pass();

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_y_hat, _output_set));
			MLPPUtilities::print_ui_vb(_weights, bias);
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

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(_n - 1));

	forward_pass();

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

		real_t z = propagatev(input_set_row_tmp);

		cost_prev = cost(y_hat_row_tmp, output_set_row_tmp);

		real_t error = y_hat - output_set_element;

		// Weight Updation
		_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate * error * Math::exp(z - Math::exp(z)), input_set_row_tmp));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Bias updation
		bias -= learning_rate * error * exp(z - exp(z));

		y_hat = evaluatev(input_set_row_tmp);

		if (p_) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat_row_tmp, output_set_row_tmp));
			MLPPUtilities::print_ui_vb(_weights, bias);
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
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_batch);
			Ref<MLPPVector> z = propagatem(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			Ref<MLPPVector> error = alg.subtractionnv(y_hat, current_output_batch);

			// Calculating the weight gradients
			_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate / _n, alg.mat_vec_multv(alg.transposenm(current_input_batch), alg.hadamard_productnv(error, avn.cloglog_derivv(z)))));
			_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elementsv(alg.hadamard_productnv(error, avn.cloglog_derivv(z))) / _n;

			forward_pass();

			y_hat = evaluatem(current_input_batch);

			if (p_) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_batch));
				MLPPUtilities::print_ui_vb(_weights, bias);
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

	return util.performance_vec(_y_hat, _output_set);
}

MLPPCLogLogReg::MLPPCLogLogReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = _input_set->size().y;
	_k = _input_set->size().x;
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.instance();
	_y_hat->resize(_n);

	MLPPUtilities utilities;

	_weights.instance();
	_weights->resize(_k);
	utilities.weight_initializationv(_weights);

	bias = utilities.bias_initializationr();
}

MLPPCLogLogReg::MLPPCLogLogReg() {
}
MLPPCLogLogReg::~MLPPCLogLogReg() {
}

real_t MLPPCLogLogReg::cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;

	return mlpp_cost.msev(y_hat, y) + regularization.reg_termv(_weights, _lambda, _alpha, _reg);
}

real_t MLPPCLogLogReg::evaluatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	return avn.cloglog_normr(alg.dotv(_weights, x) + bias);
}

real_t MLPPCLogLogReg::propagatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;

	return alg.dotv(_weights, x) + bias;
}

Ref<MLPPVector> MLPPCLogLogReg::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	return avn.cloglog_normv(alg.scalar_addnv(bias, alg.mat_vec_multv(X, _weights)));
}

Ref<MLPPVector> MLPPCLogLogReg::propagatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;

	return alg.scalar_addnv(bias, alg.mat_vec_multv(X, _weights));
}

// cloglog ( wTx + b )
void MLPPCLogLogReg::forward_pass() {
	MLPPActivation avn;

	_z = propagatem(_input_set);
	_y_hat = avn.cloglog_normv(_z);
}

void MLPPCLogLogReg::_bind_methods() {
}
