/*************************************************************************/
/*  lin_reg.cpp                                                          */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2023-present Péter Magyar.                              */
/* Copyright (c) 2022-2023 Marc Melikyan                                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "lin_reg.h"

#include "../core/cost.h"
#include "../core/reg.h"
#include "../core/stat.h"
#include "../core/utilities.h"

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

	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPVector> error = _y_hat->subn(_output_set);

		// Calculating the weight gradients (2nd derivative)

		Ref<MLPPVector> first_derivative = _input_set->transposen()->mult_vec(error);
		Ref<MLPPMatrix> second_derivative = _input_set->transposen()->multn(_input_set);

		_weights->sub(second_derivative->inverse()->transposen()->mult_vec(first_derivative)->scalar_multiplyn(learning_rate / _n));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients (2nd derivative)
		_bias -= learning_rate * error->sum_elements() / _n; // We keep this the same. The 2nd derivative is just [1].

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

	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPVector> error = _y_hat->subn(_output_set);

		// Calculating the weight gradients
		_weights->sub(_input_set->transposen()->mult_vec(error)->scalar_multiplyn(learning_rate / _n));
		_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		_bias -= learning_rate * error->sum_elements() / _n;

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

		_input_set->row_get_into_mlpp_vector(output_index, input_set_row_tmp);
		real_t output_element_set = _output_set->element_get(output_index);
		output_set_row_tmp->element_set(0, output_element_set);

		real_t y_hat = evaluatev(input_set_row_tmp);
		y_hat_tmp->element_set(0, output_element_set);

		cost_prev = cost(y_hat_tmp, output_set_row_tmp);

		real_t error = y_hat - output_element_set;

		// Weight updation
		_weights->sub(input_set_row_tmp->scalar_multiplyn(learning_rate * error));
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

			Ref<MLPPVector> error = y_hat->subn(current_output_mini_batch);

			// Calculating the weight gradients
			_weights->sub(current_input_mini_batch->transposen()->mult_vec(error)->scalar_multiplyn(learning_rate / current_output_mini_batch->size()));
			_weights = regularization.reg_weightsv(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients
			_bias -= learning_rate * error->sum_elements() / current_output_mini_batch->size();
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

	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Momentum.
	Ref<MLPPVector> v = MLPPVector::create_vec_zero(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = y_hat->subn(current_output_mini_batch);

			// Calculating the weight gradients

			Ref<MLPPVector> gradient = current_input_mini_batch->transposen()->mult_vec(error)->scalar_multiplyn(1 / current_output_mini_batch->size());

			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = gradient->addn(reg_deriv_term); // Weight_grad_final

			v = v->scalar_multiplyn(gamma)->addn(weight_grad->scalar_multiplyn(learning_rate));
			_weights->sub(v);

			// Calculating the bias gradients
			_bias -= learning_rate * error->sum_elements() / current_output_mini_batch->size(); // As normal
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

	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Momentum.
	Ref<MLPPVector> v = MLPPVector::create_vec_zero(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			_weights->sub(v->scalar_multiplyn(gamma)); // "Aposterori" calculation

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = y_hat->subn(current_output_mini_batch);

			// Calculating the weight gradients

			Ref<MLPPVector> gradient = current_input_mini_batch->transposen()->mult_vec(error)->scalar_multiplyn(1 / current_output_mini_batch->size());
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = gradient->addn(reg_deriv_term); // Weight_grad_final

			v = v->scalar_multiplyn(gamma)->addn(weight_grad->scalar_multiplyn(learning_rate));

			_weights->sub(v);

			// Calculating the bias gradients
			_bias -= learning_rate * error->sum_elements() / current_output_mini_batch->size(); // As normal
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

	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adagrad.
	Ref<MLPPVector> v = MLPPVector::create_vec_zero(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = y_hat->subn(current_output_mini_batch);

			// Calculating the weight gradients
			Ref<MLPPVector> gradient = current_input_mini_batch->transposen()->mult_vec(error)->scalar_multiplyn(1 / current_output_mini_batch->size());
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = gradient->addn(reg_deriv_term); // Weight_grad_final

			v = weight_grad->hadamard_productn(weight_grad);
			_weights->sub(weight_grad->division_element_wisen(v->scalar_addn(e)->sqrtn())->scalar_multiplyn(learning_rate));

			// Calculating the bias gradients
			_bias -= learning_rate * error->sum_elements() / current_output_mini_batch->size(); // As normal
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
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adagrad.
	Ref<MLPPVector> v = MLPPVector::create_vec_zero(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = y_hat->subn(current_output_mini_batch);

			// Calculating the weight gradients
			Ref<MLPPVector> gradient = current_input_mini_batch->transposen()->mult_vec(error)->scalar_multiplyn(1 / current_output_mini_batch->size());
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = gradient->addn(reg_deriv_term); // Weight_grad_final

			v = v->scalar_multiplyn(b1)->addn(weight_grad->hadamard_productn(weight_grad)->scalar_multiplyn(1 - b1));
			_weights->sub(weight_grad->division_element_wisen(v->scalar_addn(e)->sqrtn())->scalar_multiplyn(learning_rate));

			// Calculating the bias gradients
			_bias -= learning_rate * error->sum_elements() / current_output_mini_batch->size(); // As normal
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

	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adam.
	Ref<MLPPVector> m = MLPPVector::create_vec_zero(_weights->size());
	Ref<MLPPVector> v = MLPPVector::create_vec_zero(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = y_hat->subn(current_output_mini_batch);

			// Calculating the weight gradients
			Ref<MLPPVector> gradient = current_input_mini_batch->transposen()->mult_vec(error)->scalar_multiplyn(1 / current_output_mini_batch->size());
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = gradient->addn(reg_deriv_term); // Weight_grad_final

			m = m->scalar_multiplyn(b1)->addn(weight_grad->scalar_multiplyn(1 - b1));
			v = v->scalar_multiplyn(b2)->addn(weight_grad->exponentiaten(2)->scalar_multiplyn(1 - b2));

			Ref<MLPPVector> m_hat = m->scalar_multiplyn(1 / (1 - Math::pow(b1, epoch)));
			Ref<MLPPVector> v_hat = v->scalar_multiplyn(1 / (1 - Math::pow(b2, epoch)));

			_weights->sub(m_hat->division_element_wisen(v_hat->sqrtn()->scalar_addn(e))->scalar_multiplyn(learning_rate));

			// Calculating the bias gradients
			_bias -= learning_rate * error->sum_elements() / current_output_mini_batch->size(); // As normal
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

	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	Ref<MLPPVector> m = MLPPVector::create_vec_zero(_weights->size());
	Ref<MLPPVector> u = MLPPVector::create_vec_zero(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = y_hat->subn(current_output_mini_batch);

			// Calculating the weight gradients
			Ref<MLPPVector> gradient = current_input_mini_batch->transposen()->mult_vec(error)->scalar_multiplyn(1 / current_output_mini_batch->size());
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = gradient->addn(reg_deriv_term); // Weight_grad_final

			m = m->scalar_multiplyn(b1)->addn(weight_grad->scalar_multiplyn(1 - b1));
			u = u->scalar_multiplyn(b2)->maxn(weight_grad->absn());

			Ref<MLPPVector> m_hat = m->scalar_multiplyn(1 / (1 - Math::pow(b1, epoch)));
			_weights->sub(m_hat->division_element_wisen(u)->scalar_multiplyn(learning_rate));

			// Calculating the bias gradients
			_bias -= learning_rate * error->sum_elements() / current_output_mini_batch->size(); // As normal
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

	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adam.
	Ref<MLPPVector> m = MLPPVector::create_vec_zero(_weights->size());
	Ref<MLPPVector> v = MLPPVector::create_vec_zero(_weights->size());
	Ref<MLPPVector> m_final = MLPPVector::create_vec_zero(_weights->size());

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_mini_batch);
			cost_prev = cost(y_hat, current_output_mini_batch);

			Ref<MLPPVector> error = y_hat->subn(current_output_mini_batch);

			// Calculating the weight gradients
			Ref<MLPPVector> gradient = current_input_mini_batch->transposen()->mult_vec(error)->scalar_multiplyn(1 / current_output_mini_batch->size());
			Ref<MLPPVector> reg_deriv_term = regularization.reg_deriv_termv(_weights, _lambda, _alpha, _reg);
			Ref<MLPPVector> weight_grad = gradient->addn(reg_deriv_term); // Weight_grad_final

			m = m->scalar_multiplyn(b1)->addn(weight_grad->scalar_multiplyn(1 - b1));
			v = v->scalar_multiplyn(b2)->addn(weight_grad->exponentiaten(2)->scalar_multiplyn(1 - b2));

			m_final = m->scalar_multiplyn(b1)->addn(weight_grad->scalar_multiplyn((1 - b1) / (1 - Math::pow(b1, epoch))));

			Ref<MLPPVector> m_hat = m->scalar_multiplyn(1 / (1 - Math::pow(b1, epoch)));
			Ref<MLPPVector> v_hat = v->scalar_multiplyn(1 / (1 - Math::pow(b2, epoch)));

			_weights->sub(m_final->division_element_wisen(v_hat->sqrtn()->scalar_addn(e))->scalar_multiplyn(learning_rate));

			// Calculating the bias gradients
			_bias -= learning_rate * error->sum_elements() / current_output_mini_batch->size(); // As normal
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

	MLPPStat stat;

	Ref<MLPPMatrix> input_set_t = _input_set->transposen();

	Ref<MLPPVector> input_set_t_row_tmp;
	input_set_t_row_tmp.instance();
	input_set_t_row_tmp->resize(input_set_t->size().x);

	Ref<MLPPVector> x_means;
	x_means.instance();
	x_means->resize(input_set_t->size().y);

	for (int i = 0; i < input_set_t->size().y; i++) {
		input_set_t->row_get_into_mlpp_vector(i, input_set_t_row_tmp);

		x_means->element_set(i, stat.meanv(input_set_t_row_tmp));
	}

	Ref<MLPPVector> temp;
	//temp.resize(_k);

	temp = input_set_t->multn(_input_set)->inverse()->mult_vec(input_set_t->mult_vec(_output_set));

	ERR_FAIL_COND_MSG(Math::is_nan(temp->element_get(0)), "ERR: Resulting matrix was noninvertible/degenerate, and so the normal equation could not be performed. Try utilizing gradient descent.");

	if (_reg == MLPPReg::REGULARIZATION_TYPE_RIDGE) {
		_weights = _input_set->transposen()->multn(_input_set)->addn(MLPPMatrix::create_identity_mat(_k)->scalar_multiplyn(_lambda))->inverse()->mult_vec(_input_set->transposen()->mult_vec(_output_set));
	} else {
		_weights = _input_set->transposen()->multn(_input_set)->inverse()->mult_vec(_input_set->transposen()->mult_vec(_output_set));
	}

	_bias = stat.meanv(_output_set) - _weights->dot(x_means);

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
	return _weights->dot(x) + _bias;
}

Ref<MLPPVector> MLPPLinReg::evaluatem(const Ref<MLPPMatrix> &X) {
	return X->mult_vec(_weights)->scalar_addn(_bias);
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
