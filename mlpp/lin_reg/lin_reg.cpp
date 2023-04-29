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

Ref<MLPPVector> MLPPLinReg::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPVector>());

	return evaluatem(X);
}

real_t MLPPLinReg::model_test(const Ref<MLPPVector> &x) {
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

		Ref<MLPPVector> error = alg.subtractionnv(_y_hat, _output_set);

		// Calculating the weight gradients (2nd derivative)
		Ref<MLPPVector> first_derivative = alg.mat_vec_multnv(alg.transposenm(_input_set), error);
		Ref<MLPPMatrix> second_derivative = alg.matmultnm(alg.transposenm(_input_set), _input_set);
		_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate / _n, alg.mat_vec_multnv(alg.transposenm(alg.inversenm(second_derivative)), first_derivative)));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients (2nd derivative)
		_bias -= learning_rate * alg.sum_elementsv(error) / _n; // We keep this the same. The 2nd derivative is just [1].

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

void MLPPLinReg::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPVector> error = alg.subtractionnv(_y_hat, _output_set);

		// Calculating the weight gradients
		_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate / _n, alg.mat_vec_multnv(alg.transposenm(_input_set), error)));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		_bias -= learning_rate * alg.sum_elementsv(error) / _n;

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

void MLPPLinReg::sgd(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
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

	Ref<MLPPVector> y_hat_tmp;
	y_hat_tmp.instance();
	y_hat_tmp->resize(1);

	while (true) {
		int output_index = distribution(generator);

		_input_set->get_row_into_mlpp_vector(output_index, input_set_row_tmp);
		real_t output_element_set = _output_set->element_get(output_index);
		output_set_row_tmp->element_set(0, output_element_set);

		real_t y_hat = evaluatev(input_set_row_tmp);
		y_hat_tmp->element_set(0, output_element_set);

		cost_prev = cost(y_hat_tmp, output_set_row_tmp);

		real_t error = y_hat - output_element_set;

		// Weight updation
		_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate * error, input_set_row_tmp));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Bias updation
		_bias -= learning_rate * error;

		y_hat = evaluatev(input_set_row_tmp);

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat_tmp, output_set_row_tmp));
			MLPPUtilities::print_ui_vb(_weights, _bias);
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
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = alg.subtractionnv(y_hat, current_output_mini_batch);

			// Calculating the weight gradients
			_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate / current_output_mini_batch->size(), alg.mat_vec_multnv(alg.transposenm(current_input_mini_batch), error)));
			_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elementsv(error) / current_output_mini_batch->size();
			y_hat = evaluatem(current_input_mini_batch);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_mini_batch));
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

void MLPPLinReg::momentum(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Momentum.
	Ref<MLPPVector> v = alg.zerovecnv(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = alg.subtractionnv(y_hat, current_output_mini_batch);

			// Calculating the weight gradients
			Ref<MLPPVector> gradient = alg.scalar_multiplynv(1 / current_output_mini_batch->size(), alg.mat_vec_multnv(alg.transposenm(current_input_mini_batch), error));
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = alg.additionnv(gradient, reg_deriv_term); // Weight_grad_final

			v = alg.additionnv(alg.scalar_multiplynv(gamma, v), alg.scalar_multiplynv(learning_rate, weight_grad));

			_weights = alg.subtractionnv(_weights, v);

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elementsv(error) / current_output_mini_batch->size(); // As normal
			y_hat = evaluatem(current_input_mini_batch);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_mini_batch));
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

void MLPPLinReg::nag(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Momentum.
	Ref<MLPPVector> v = alg.zerovecnv(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(gamma, v)); // "Aposterori" calculation

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = alg.subtractionnv(y_hat, current_output_mini_batch);

			// Calculating the weight gradients
			Ref<MLPPVector> gradient = alg.scalar_multiplynv(1 / current_output_mini_batch->size(), alg.mat_vec_multnv(alg.transposenm(current_input_mini_batch), error));
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = alg.additionnv(gradient, reg_deriv_term); // Weight_grad_final

			v = alg.additionnv(alg.scalar_multiplynv(gamma, v), alg.scalar_multiplynv(learning_rate, weight_grad));

			_weights = alg.subtractionnv(_weights, v);

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elementsv(error) / current_output_mini_batch->size(); // As normal
			y_hat = evaluatem(current_input_mini_batch);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_mini_batch));
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

void MLPPLinReg::adagrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t e, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adagrad.
	Ref<MLPPVector> v = alg.zerovecnv(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = alg.subtractionnv(y_hat, current_output_mini_batch);

			// Calculating the weight gradients
			Ref<MLPPVector> gradient = alg.scalar_multiplynv(1 / current_output_mini_batch->size(), alg.mat_vec_multnv(alg.transposenm(current_input_mini_batch), error));
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = alg.additionnv(gradient, reg_deriv_term); // Weight_grad_final

			v = alg.hadamard_productnv(weight_grad, weight_grad);

			_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate, alg.division_element_wisenv(weight_grad, alg.sqrtnv(alg.scalar_addnv(e, v)))));

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elementsv(error) / current_output_mini_batch->size(); // As normal
			y_hat = evaluatem(current_input_mini_batch);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_mini_batch));
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

void MLPPLinReg::adadelta(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t e, bool ui) {
	ERR_FAIL_COND(!_initialized);

	// Adagrad upgrade. Momentum is applied.
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adagrad.
	Ref<MLPPVector> v = alg.zerovecnv(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = alg.subtractionnv(y_hat, current_output_mini_batch);

			// Calculating the weight gradients
			Ref<MLPPVector> gradient = alg.scalar_multiplynv(1 / current_output_mini_batch->size(), alg.mat_vec_multnv(alg.transposenm(current_input_mini_batch), error));
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = alg.additionnv(gradient, reg_deriv_term); // Weight_grad_final

			v = alg.additionnv(alg.scalar_multiplynv(b1, v), alg.scalar_multiplynv(1 - b1, alg.hadamard_productnv(weight_grad, weight_grad)));

			_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate, alg.division_element_wisenv(weight_grad, alg.sqrtnv(alg.scalar_addnv(e, v)))));

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elementsv(error) / current_output_mini_batch->size(); // As normal
			y_hat = evaluatem(current_input_mini_batch);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_mini_batch));
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

void MLPPLinReg::adam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adam.
	Ref<MLPPVector> m = alg.zerovecnv(_weights->size());
	Ref<MLPPVector> v = alg.zerovecnv(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = alg.subtractionnv(y_hat, current_output_mini_batch);

			// Calculating the weight gradients
			Ref<MLPPVector> gradient = alg.scalar_multiplynv(1 / current_output_mini_batch->size(), alg.mat_vec_multnv(alg.transposenm(current_input_mini_batch), error));
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = alg.additionnv(gradient, reg_deriv_term); // Weight_grad_final

			m = alg.additionnv(alg.scalar_multiplynv(b1, m), alg.scalar_multiplynv(1 - b1, weight_grad));
			v = alg.additionnv(alg.scalar_multiplynv(b2, v), alg.scalar_multiplynv(1 - b2, alg.exponentiatenv(weight_grad, 2)));

			Ref<MLPPVector> m_hat = alg.scalar_multiplynv(1 / (1 - Math::pow(b1, epoch)), m);
			Ref<MLPPVector> v_hat = alg.scalar_multiplynv(1 / (1 - Math::pow(b2, epoch)), v);

			_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate, alg.division_element_wisenvnm(m_hat, alg.scalar_addnv(e, alg.sqrtnv(v_hat)))));

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elementsv(error) / current_output_mini_batch->size(); // As normal
			y_hat = evaluatem(current_input_mini_batch);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_mini_batch));
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

void MLPPLinReg::adamax(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	Ref<MLPPVector> m = alg.zerovecnv(_weights->size());
	Ref<MLPPVector> u = alg.zerovecnv(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = alg.subtractionnv(y_hat, current_output_mini_batch);

			// Calculating the weight gradients
			Ref<MLPPVector> gradient = alg.scalar_multiplynv(1 / current_output_mini_batch->size(), alg.mat_vec_multnv(alg.transposenm(current_input_mini_batch), error));
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = alg.additionnv(gradient, reg_deriv_term); // Weight_grad_final

			m = alg.additionnv(alg.scalar_multiplynv(b1, m), alg.scalar_multiplynv(1 - b1, weight_grad));
			u = alg.maxnvv(alg.scalar_multiplynv(b2, u), alg.absv(weight_grad));

			Ref<MLPPVector> m_hat = alg.scalar_multiplynv(1 / (1 - Math::pow(b1, epoch)), m);

			_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate, alg.division_element_wisenv(m_hat, u)));

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elementsv(error) / current_output_mini_batch->size(); // As normal
			y_hat = evaluatem(current_input_mini_batch);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_mini_batch));
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

void MLPPLinReg::nadam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adam.
	Ref<MLPPVector> m = alg.zerovecnv(_weights->size());
	Ref<MLPPVector> v = alg.zerovecnv(_weights->size());
	Ref<MLPPVector> m_final = alg.zerovecnv(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = alg.subtractionnv(y_hat, current_output_mini_batch);

			// Calculating the weight gradients
			Ref<MLPPVector> gradient = alg.scalar_multiplynv(1 / current_output_mini_batch->size(), alg.mat_vec_multnv(alg.transposenm(current_input_mini_batch), error));
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = alg.additionnv(gradient, reg_deriv_term); // Weight_grad_final

			m = alg.additionnv(alg.scalar_multiplynv(b1, m), alg.scalar_multiplynv(1 - b1, weight_grad));
			v = alg.additionnv(alg.scalar_multiplynv(b2, v), alg.scalar_multiplynv(1 - b2, alg.exponentiatenv(weight_grad, 2)));
			m_final = alg.additionnv(alg.scalar_multiplynv(b1, m), alg.scalar_multiplynv((1 - b1) / (1 - Math::pow(b1, epoch)), weight_grad));

			Ref<MLPPVector> m_hat = alg.scalar_multiplynv(1 / (1 - Math::pow(b1, epoch)), m);
			Ref<MLPPVector> v_hat = alg.scalar_multiplynv(1 / (1 - Math::pow(b2, epoch)), v);

			_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate, alg.division_element_wisenv(m_final, alg.scalar_addnv(e, alg.sqrtnv(v_hat)))));

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elementsv(error) / current_output_mini_batch->size(); // As normal
			y_hat = evaluatem(current_input_mini_batch);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_mini_batch));
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

void MLPPLinReg::normal_equation() {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPStat stat;

	Ref<MLPPMatrix> input_set_t = alg.transposenm(_input_set);

	Ref<MLPPVector> input_set_t_row_tmp;
	input_set_t_row_tmp.instance();
	input_set_t_row_tmp->resize(input_set_t->size().x);

	Ref<MLPPVector> x_means;
	x_means.instance();
	x_means->resize(input_set_t->size().y);

	for (int i = 0; i < input_set_t->size().y; i++) {
		input_set_t->get_row_into_mlpp_vector(i, input_set_t_row_tmp);

		x_means->element_set(i, stat.meanv(input_set_t_row_tmp));
	}

	Ref<MLPPVector> temp;
	//temp.resize(_k);
	temp = alg.mat_vec_multnv(alg.inversenm(alg.matmultnm(alg.transposenm(_input_set), _input_set)), alg.mat_vec_multnv(alg.transposenm(_input_set), _output_set));

	ERR_FAIL_COND_MSG(Math::is_nan(temp->element_get(0)), "ERR: Resulting matrix was noninvertible/degenerate, and so the normal equation could not be performed. Try utilizing gradient descent.");

	if (_reg == MLPPReg::REGULARIZATION_TYPE_RIDGE) {
		_weights = alg.mat_vec_multnv(alg.inversenm(alg.additionnm(alg.matmultnm(alg.transposenm(_input_set), _input_set), alg.scalar_multiplynm(_lambda, alg.identitym(_k)))), alg.mat_vec_multnv(alg.transposenm(_input_set), _output_set));
	} else {
		_weights = alg.mat_vec_multnv(alg.inversenm(alg.matmultnm(alg.transposenm(_input_set), _input_set)), alg.mat_vec_multnv(alg.transposenm(_input_set), _output_set));
	}

	_bias = stat.meanv(_output_set) - alg.dotnv(_weights, x_means);

	forward_pass();
}

real_t MLPPLinReg::score() {
	ERR_FAIL_COND_V(!_initialized, 0);

	MLPPUtilities util;

	return util.performance_vec(_y_hat, _output_set);
}

void MLPPLinReg::save(const String &file_name) {
	ERR_FAIL_COND(!_initialized);

	//MLPPUtilities util;

	//util.saveParameters(fileName, _weights, _bias);
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

MLPPLinReg::MLPPLinReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = p_input_set->size().y;
	_k = p_input_set->size().x;
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.instance();
	_y_hat->resize(_n);

	_weights.instance();
	_weights->resize(_k);

	MLPPUtilities utils;

	utils.weight_initializationv(_weights);
	_bias = utils.bias_initializationr();

	_initialized = true;
}

MLPPLinReg::MLPPLinReg() {
	_initialized = false;
}
MLPPLinReg::~MLPPLinReg() {
}

real_t MLPPLinReg::cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;

	return mlpp_cost.msev(y_hat, y) + regularization.reg_termv(_weights, _lambda, _alpha, _reg);
}

real_t MLPPLinReg::evaluatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;

	return alg.dotnv(_weights, x) + _bias;
}

Ref<MLPPVector> MLPPLinReg::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;

	return alg.scalar_addnv(_bias, alg.mat_vec_multnv(X, _weights));
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
