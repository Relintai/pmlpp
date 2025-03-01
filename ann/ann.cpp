/*************************************************************************/
/*  ann.cpp                                                              */
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

#include "ann.h"

#include "../core/activation.h"
#include "../core/cost.h"
#include "../core/lin_alg.h"
#include "../core/reg.h"
#include "../core/utilities.h"

#ifdef USING_SFW
#include "sfw.h"
#else
#include "core/log/logger.h"
#endif

#include <random>

Ref<MLPPVector> MLPPANN::model_set_test(const Ref<MLPPMatrix> &X) {
	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[0];

		layer->set_input(X);
		layer->forward_pass();

		for (int i = 1; i < _network.size(); i++) {
			layer = _network[i];
			Ref<MLPPHiddenLayer> prev_layer = _network[i - 1];

			layer->set_input(prev_layer->get_a());
			layer->forward_pass();
		}

		_output_layer->set_input(_network.write[_network.size() - 1]->get_a());
	} else {
		_output_layer->set_input(X);
	}

	_output_layer->forward_pass();

	return _output_layer->get_a();
}

real_t MLPPANN::model_test(const Ref<MLPPVector> &x) {
	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[0];

		layer->test(x);

		for (int i = 1; i < _network.size(); i++) {
			layer = _network[i];
			Ref<MLPPHiddenLayer> prev_layer = _network[i - 1];

			layer->test(prev_layer->get_a_test());
		}

		_output_layer->test(_network.write[_network.size() - 1]->get_a_test());
	} else {
		_output_layer->test(x);
	}

	return _output_layer->get_a_test();
}

void MLPPANN::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	MLPPCost mlpp_cost;
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	real_t initial_learning_rate = learning_rate;

	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		cost_prev = cost(_y_hat, _output_set);

		ComputeGradientsResult grads = compute_gradients(_y_hat, _output_set);
		grads.cumulative_hidden_layer_w_grad = alg.scalar_multiplynvt(learning_rate / _n, grads.cumulative_hidden_layer_w_grad);

		grads.output_w_grad->scalar_multiply(learning_rate / _n);

		update_parameters(grads.cumulative_hidden_layer_w_grad, grads.output_w_grad, learning_rate); // subject to change. may want bias to have this matrix too.

		forward_pass();

		if (ui) {
			print_ui(epoch, cost_prev, _y_hat, _output_set);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPANN::sgd(real_t learning_rate, int max_epoch, bool ui) {
	MLPPCost mlpp_cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

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
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		int output_index = distribution(generator);

		_input_set->row_get_into_mlpp_vector(output_index, input_set_row_tmp);
		real_t output_element_set = _output_set->element_get(output_index);
		output_set_row_tmp->element_set(0, output_element_set);

		real_t y_hat = model_test(input_set_row_tmp);
		y_hat_row_tmp->element_set(0, y_hat);

		cost_prev = cost(y_hat_row_tmp, output_set_row_tmp);

		ComputeGradientsResult grads = compute_gradients(y_hat_row_tmp, output_set_row_tmp);

		grads.cumulative_hidden_layer_w_grad = alg.scalar_multiplynvt(learning_rate / _n, grads.cumulative_hidden_layer_w_grad);
		grads.output_w_grad->scalar_multiply(learning_rate / _n);

		update_parameters(grads.cumulative_hidden_layer_w_grad, grads.output_w_grad, learning_rate); // subject to change. may want bias to have this matrix too.
		y_hat = model_test(input_set_row_tmp);

		if (ui) {
			print_ui(epoch, cost_prev, y_hat_row_tmp, output_set_row_tmp);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPANN::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	MLPPCost mlpp_cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = model_set_test(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			ComputeGradientsResult grads = compute_gradients(y_hat, current_output_batch);

			grads.cumulative_hidden_layer_w_grad = alg.scalar_multiplynvt(learning_rate / _n, grads.cumulative_hidden_layer_w_grad);
			grads.output_w_grad->scalar_multiply(learning_rate / _n);

			update_parameters(grads.cumulative_hidden_layer_w_grad, grads.output_w_grad, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(current_input_batch);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, current_output_batch);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPANN::momentum(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool nag, bool ui) {
	class MLPPCost mlpp_cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adam.
	Vector<Ref<MLPPMatrix>> v_hidden;

	Ref<MLPPVector> v_output;
	v_output.instance();

	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = model_set_test(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			ComputeGradientsResult grads = compute_gradients(y_hat, current_output_batch);

			if (!_network.empty() && v_hidden.empty()) { // Initing our tensor
				alg.resizevt(v_hidden, grads.cumulative_hidden_layer_w_grad);
			}

			if (v_output->size() == 0) {
				v_output->resize(grads.output_w_grad->size());
			}

			if (nag) { // "Aposterori" calculation
				update_parameters(v_hidden, v_output, 0); // DON'T update bias.
			}

			v_hidden = alg.additionnvt(alg.scalar_multiplynvt(gamma, v_hidden), alg.scalar_multiplynvt(learning_rate / _n, grads.cumulative_hidden_layer_w_grad));
			v_output = v_output->scalar_multiplyn(gamma)->addn(grads.output_w_grad->scalar_multiplyn(learning_rate / _n));

			update_parameters(v_hidden, v_output, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(current_input_batch);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, current_output_batch);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPANN::adagrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t e, bool ui) {
	MLPPCost mlpp_cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adam.
	Vector<Ref<MLPPMatrix>> v_hidden;

	Ref<MLPPVector> v_output;
	v_output.instance();

	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = model_set_test(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			ComputeGradientsResult grads = compute_gradients(y_hat, current_output_batch);

			if (!_network.empty() && v_hidden.empty()) { // Initing our tensor
				alg.resizevt(v_hidden, grads.cumulative_hidden_layer_w_grad);
			}

			if (v_output->size() == 0) {
				v_output->resize(grads.output_w_grad->size());
			}

			v_hidden = alg.additionnvt(v_hidden, alg.exponentiatenvt(grads.cumulative_hidden_layer_w_grad, 2));
			v_output->add(grads.output_w_grad->exponentiaten(2));

			Vector<Ref<MLPPMatrix>> hidden_layer_updations = alg.scalar_multiplynvt(learning_rate / _n, alg.division_element_wisenvnvt(grads.cumulative_hidden_layer_w_grad, alg.scalar_addnvt(e, alg.sqrtnvt(v_hidden))));

			Ref<MLPPVector> output_layer_updation = grads.output_w_grad->division_element_wisen(v_output->sqrtn()->scalar_addn(e))->scalar_multiplyn(learning_rate / _n);

			update_parameters(hidden_layer_updations, output_layer_updation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(current_input_batch);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, current_output_batch);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPANN::adadelta(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t e, bool ui) {
	MLPPCost mlpp_cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adam.
	Vector<Ref<MLPPMatrix>> v_hidden;

	Ref<MLPPVector> v_output;
	v_output.instance();

	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = model_set_test(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			ComputeGradientsResult grads = compute_gradients(y_hat, current_output_batch);

			if (!_network.empty() && v_hidden.empty()) { // Initing our tensor
				alg.resizevt(v_hidden, grads.cumulative_hidden_layer_w_grad);
			}

			if (v_output->size() == 0) {
				v_output->resize(grads.output_w_grad->size());
			}

			v_hidden = alg.additionnvt(alg.scalar_multiplynvt(1 - b1, v_hidden), alg.scalar_multiplynvt(b1, alg.exponentiatenvt(grads.cumulative_hidden_layer_w_grad, 2)));
			v_output->add(grads.output_w_grad->exponentiaten(2));

			Vector<Ref<MLPPMatrix>> hidden_layer_updations = alg.scalar_multiplynvt(learning_rate / _n, alg.division_element_wisenvnvt(grads.cumulative_hidden_layer_w_grad, alg.scalar_addnvt(e, alg.sqrtnvt(v_hidden))));
			Ref<MLPPVector> output_layer_updation = grads.output_w_grad->division_element_wisen(v_output->sqrtn()->scalar_addn(e))->scalar_multiplyn(learning_rate / _n);

			update_parameters(hidden_layer_updations, output_layer_updation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(current_input_batch);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, current_output_batch);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPANN::adam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui) {
	MLPPCost mlpp_cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adam.
	Vector<Ref<MLPPMatrix>> m_hidden;
	Vector<Ref<MLPPMatrix>> v_hidden;

	Ref<MLPPVector> m_output;
	Ref<MLPPVector> v_output;
	m_output.instance();
	v_output.instance();

	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = model_set_test(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			ComputeGradientsResult grads = compute_gradients(y_hat, current_output_batch);

			if (!_network.empty() && m_hidden.empty() && v_hidden.empty()) { // Initing our tensor
				alg.resizevt(m_hidden, grads.cumulative_hidden_layer_w_grad);
				alg.resizevt(v_hidden, grads.cumulative_hidden_layer_w_grad);
			}

			if (m_output->size() == 0 && v_output->size()) {
				m_output->resize(grads.output_w_grad->size());
				v_output->resize(grads.output_w_grad->size());
			}

			m_hidden = alg.additionnvt(alg.scalar_multiplynvt(b1, m_hidden), alg.scalar_multiplynvt(1 - b1, grads.cumulative_hidden_layer_w_grad));
			v_hidden = alg.additionnvt(alg.scalar_multiplynvt(b2, v_hidden), alg.scalar_multiplynvt(1 - b2, alg.exponentiatenvt(grads.cumulative_hidden_layer_w_grad, 2)));

			m_output = m_output->scalar_multiplyn(b1)->addn(grads.output_w_grad->scalar_multiplyn(1 - b1));
			v_output = v_output->scalar_multiplyn(b2)->addn(grads.output_w_grad->exponentiaten(2)->scalar_multiplyn(1 - b2));

			Vector<Ref<MLPPMatrix>> m_hidden_hat = alg.scalar_multiplynvt(1 / (1 - Math::pow(b1, epoch)), m_hidden);
			Vector<Ref<MLPPMatrix>> v_hidden_hat = alg.scalar_multiplynvt(1 / (1 - Math::pow(b2, epoch)), v_hidden);

			Ref<MLPPVector> m_output_hat = m_output->scalar_multiplyn(1 / (1 - Math::pow(b1, epoch)));
			Ref<MLPPVector> v_output_hat = v_output->scalar_multiplyn(1 / (1 - Math::pow(b2, epoch)));

			Vector<Ref<MLPPMatrix>> hidden_layer_updations = alg.scalar_multiplynvt(learning_rate / _n, alg.division_element_wisenvnvt(m_hidden_hat, alg.scalar_addnvt(e, alg.sqrtnvt(v_hidden_hat))));
			Ref<MLPPVector> output_layer_updation = m_output_hat->division_element_wisen(v_output_hat->sqrtn()->scalar_addn(e))->scalar_multiplyn(learning_rate / _n);

			update_parameters(hidden_layer_updations, output_layer_updation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(current_input_batch);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, current_output_batch);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
	forward_pass();
}

void MLPPANN::adamax(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui) {
	MLPPCost mlpp_cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adam.
	Vector<Ref<MLPPMatrix>> m_hidden;
	Vector<Ref<MLPPMatrix>> u_hidden;

	Ref<MLPPVector> m_output;
	Ref<MLPPVector> u_output;
	m_output.instance();
	u_output.instance();

	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = model_set_test(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			ComputeGradientsResult grads = compute_gradients(y_hat, current_output_batch);

			if (!_network.empty() && m_hidden.empty() && u_hidden.empty()) { // Initing our tensor
				alg.resizevt(m_hidden, grads.cumulative_hidden_layer_w_grad);
				alg.resizevt(u_hidden, grads.cumulative_hidden_layer_w_grad);
			}

			if (m_output->size() == 0 && u_output->size() == 0) {
				m_output->resize(grads.output_w_grad->size());
				u_output->resize(grads.output_w_grad->size());
			}

			m_hidden = alg.additionnvt(alg.scalar_multiplynvt(b1, m_hidden), alg.scalar_multiplynvt(1 - b1, grads.cumulative_hidden_layer_w_grad));
			u_hidden = alg.maxnvt(alg.scalar_multiplynvt(b2, u_hidden), alg.absnvt(grads.cumulative_hidden_layer_w_grad));

			m_output->addb(m_output->scalar_multiplyn(b1), grads.output_w_grad->scalar_multiplyn(1 - b1));
			u_output->maxb(u_output->scalar_multiplyn(b2), grads.output_w_grad->absn());

			Vector<Ref<MLPPMatrix>> m_hidden_hat = alg.scalar_multiplynvt(1 / (1 - Math::pow(b1, epoch)), m_hidden);
			Ref<MLPPVector> m_output_hat = m_output->scalar_multiplyn(1 / (1 - Math::pow(b1, epoch)));

			Vector<Ref<MLPPMatrix>> hidden_layer_updations = alg.scalar_multiplynvt(learning_rate / _n, alg.division_element_wisenvnvt(m_hidden_hat, alg.scalar_addnvt(e, u_hidden)));
			Ref<MLPPVector> output_layer_updation = m_output_hat->division_element_wisen(u_output->scalar_addn(e))->scalar_multiplyn(learning_rate / _n);

			update_parameters(hidden_layer_updations, output_layer_updation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(current_input_batch);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, current_output_batch);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
	forward_pass();
}

void MLPPANN::nadam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui) {
	MLPPCost mlpp_cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adam.
	Vector<Ref<MLPPMatrix>> m_hidden;
	Vector<Ref<MLPPMatrix>> v_hidden;

	Ref<MLPPVector> m_output;
	Ref<MLPPVector> v_output;
	m_output.instance();
	v_output.instance();

	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = model_set_test(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			ComputeGradientsResult grads = compute_gradients(y_hat, current_output_batch);

			if (!_network.empty() && m_hidden.empty() && v_hidden.empty()) { // Initing our tensor
				alg.resizevt(m_hidden, grads.cumulative_hidden_layer_w_grad);
				alg.resizevt(v_hidden, grads.cumulative_hidden_layer_w_grad);
			}

			if (m_output->size() == 0 && v_output->size() == 0) {
				m_output->resize(grads.output_w_grad->size());
				v_output->resize(grads.output_w_grad->size());
			}

			m_hidden = alg.additionnvt(alg.scalar_multiplynvt(b1, m_hidden), alg.scalar_multiplynvt(1 - b1, grads.cumulative_hidden_layer_w_grad));
			v_hidden = alg.additionnvt(alg.scalar_multiplynvt(b2, v_hidden), alg.scalar_multiplynvt(1 - b2, alg.exponentiatenvt(grads.cumulative_hidden_layer_w_grad, 2)));

			m_output->addb(m_output->scalar_multiplyn(b1), grads.output_w_grad->scalar_multiplyn(1 - b1));
			v_output->addb(v_output->scalar_multiplyn(b2), grads.output_w_grad->exponentiaten(2)->scalar_multiplyn(1 - b2));

			Vector<Ref<MLPPMatrix>> m_hidden_hat = alg.scalar_multiplynvt(1 / (1 - Math::pow(b1, epoch)), m_hidden);
			Vector<Ref<MLPPMatrix>> v_hidden_hat = alg.scalar_multiplynvt(1 / (1 - Math::pow(b2, epoch)), v_hidden);
			Vector<Ref<MLPPMatrix>> m_hidden_final = alg.additionnvt(alg.scalar_multiplynvt(b1, m_hidden_hat), alg.scalar_multiplynvt((1 - b1) / (1 - Math::pow(b1, epoch)), grads.cumulative_hidden_layer_w_grad));

			Ref<MLPPVector> m_output_hat = m_output->scalar_multiplyn(1 / (1.0 - Math::pow(b1, epoch)));
			Ref<MLPPVector> v_output_hat = v_output->scalar_multiplyn(1 / (1.0 - Math::pow(b2, epoch)));
			Ref<MLPPVector> m_output_final = m_output_hat->scalar_multiplyn(b1)->addn(grads.output_w_grad->scalar_multiplyn((1 - b1) / (1.0 - Math::pow(b1, epoch))));

			Vector<Ref<MLPPMatrix>> hidden_layer_updations = alg.scalar_multiplynvt(learning_rate / _n, alg.division_element_wisenvnvt(m_hidden_final, alg.scalar_addnvt(e, alg.sqrtnvt(v_hidden_hat))));
			Ref<MLPPVector> output_layer_updation = m_output_final->division_element_wisen(v_output_hat->sqrtn()->scalar_addn(e))->scalar_multiplyn(learning_rate / _n);

			update_parameters(hidden_layer_updations, output_layer_updation, learning_rate); // subject to change. may want bias to have this matrix too.

			y_hat = model_set_test(current_input_batch);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, current_output_batch);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPANN::amsgrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui) {
	MLPPCost mlpp_cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	// Initializing necessary components for Adam.
	Vector<Ref<MLPPMatrix>> m_hidden;
	Vector<Ref<MLPPMatrix>> v_hidden;

	Vector<Ref<MLPPMatrix>> v_hidden_hat;

	Ref<MLPPVector> m_output;
	Ref<MLPPVector> v_output;
	m_output.instance();
	v_output.instance();

	Ref<MLPPVector> v_output_hat;
	v_output_hat.instance();

	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = model_set_test(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			ComputeGradientsResult grads = compute_gradients(y_hat, current_output_batch);

			if (!_network.empty() && m_hidden.size() == 0 && v_hidden.size() == 0) { // Initing our tensor
				alg.resizevt(m_hidden, grads.cumulative_hidden_layer_w_grad);
				alg.resizevt(v_hidden, grads.cumulative_hidden_layer_w_grad);
				alg.resizevt(v_hidden_hat, grads.cumulative_hidden_layer_w_grad);
			}

			if (m_output->size() == 0 && v_output->size() == 0) {
				m_output->resize(grads.output_w_grad->size());
				v_output->resize(grads.output_w_grad->size());
				v_output_hat->resize(grads.output_w_grad->size());
			}

			m_hidden = alg.additionnvt(alg.scalar_multiplynvt(b1, m_hidden), alg.scalar_multiplynvt(1 - b1, grads.cumulative_hidden_layer_w_grad));
			v_hidden = alg.additionnvt(alg.scalar_multiplynvt(b2, v_hidden), alg.scalar_multiplynvt(1 - b2, alg.exponentiatenvt(grads.cumulative_hidden_layer_w_grad, 2)));

			m_output->addb(m_output->scalar_multiplyn(b1), grads.output_w_grad->scalar_multiplyn(1 - b1));
			v_output->addb(v_output->scalar_multiplyn(b2), grads.output_w_grad->exponentiaten(2)->scalar_multiplyn(1 - b2));

			v_hidden_hat = alg.maxnvt(v_hidden_hat, v_hidden);
			v_output_hat->max(v_output);

			Vector<Ref<MLPPMatrix>> hidden_layer_updations = alg.scalar_multiplynvt(learning_rate / _n, alg.division_element_wisenvnvt(m_hidden, alg.scalar_addnvt(e, alg.sqrtnvt(v_hidden_hat))));
			Ref<MLPPVector> output_layer_updation = m_output->division_element_wisen(v_output_hat->sqrtn()->scalar_addn(e))->scalar_multiplyn(learning_rate / _n);

			update_parameters(hidden_layer_updations, output_layer_updation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(current_input_batch);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, current_output_batch);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

real_t MLPPANN::score() {
	MLPPUtilities util;

	forward_pass();

	return util.performance_vec(_y_hat, _output_set);
}

void MLPPANN::save(const String &file_name) {
	MLPPUtilities util;

	/*
	if (!_network.empty()) {
		util.saveParameters(file_name, _network[0].weights, _network[0].bias, false, 1);
		for (uint32_t i = 1; i < _network.size(); i++) {
			util.saveParameters(file_name, _network[i].weights, _network[i].bias, true, i + 1);
		}
		util.saveParameters(file_name, _output_layer->weights, _output_layer->bias, true, _network.size() + 1);
	} else {
		util.saveParameters(file_name, _output_layer->weights, _output_layer->bias, false, _network.size() + 1);
	}
	*/
}

void MLPPANN::set_learning_rate_scheduler(SchedulerType type, real_t decay_constant) {
	_lr_scheduler = type;
	_decay_constant = decay_constant;
}

void MLPPANN::set_learning_rate_scheduler_drop(SchedulerType type, real_t decay_constant, real_t drop_rate) {
	_lr_scheduler = type;
	_decay_constant = decay_constant;
	_drop_rate = drop_rate;
}

void MLPPANN::add_layer(int n_hidden, MLPPActivation::ActivationFunction activation, MLPPUtilities::WeightDistributionType weight_init, MLPPReg::RegularizationType reg, real_t lambda, real_t alpha) {
	if (_network.empty()) {
		_network.push_back(Ref<MLPPHiddenLayer>(memnew(MLPPHiddenLayer(n_hidden, activation, _input_set, weight_init, reg, lambda, alpha))));
		_network.write[0]->forward_pass();
	} else {
		_network.push_back(Ref<MLPPHiddenLayer>(memnew(MLPPHiddenLayer(n_hidden, activation, _network.write[_network.size() - 1]->get_a(), weight_init, reg, lambda, alpha))));
		_network.write[_network.size() - 1]->forward_pass();
	}
}

void MLPPANN::add_output_layer(MLPPActivation::ActivationFunction activation, MLPPCost::CostTypes loss, MLPPUtilities::WeightDistributionType weight_init, MLPPReg::RegularizationType reg, real_t lambda, real_t alpha) {
	if (!_network.empty()) {
		_output_layer = Ref<MLPPOutputLayer>(memnew(MLPPOutputLayer(_network.write[_network.size() - 1]->get_n_hidden(), activation, loss, _network.write[_network.size() - 1]->get_a(), weight_init, reg, lambda, alpha)));
	} else {
		_output_layer = Ref<MLPPOutputLayer>(memnew(MLPPOutputLayer(_k, activation, loss, _input_set, weight_init, reg, lambda, alpha)));
	}
}

MLPPANN::MLPPANN(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set) {
	_input_set = p_input_set;
	_output_set = p_output_set;

	_n = _input_set->size().y;
	_k = _input_set->size().x;
	_lr_scheduler = SCHEDULER_TYPE_NONE;
	_decay_constant = 0;
	_drop_rate = 0;
}

MLPPANN::MLPPANN() {
}

MLPPANN::~MLPPANN() {
}

// https://en.wikipedia.org/wiki/Learning_rate
// Learning Rate Decay (C2W2L09) - Andrew Ng - Deep Learning Specialization
real_t MLPPANN::apply_learning_rate_scheduler(real_t learning_rate, real_t decay_constant, real_t epoch, real_t drop_rate) {
	if (_lr_scheduler == SCHEDULER_TYPE_TIME) {
		return learning_rate / (1 + decay_constant * epoch);
	} else if (_lr_scheduler == SCHEDULER_TYPE_EPOCH) {
		return learning_rate * (decay_constant / std::sqrt(epoch));
	} else if (_lr_scheduler == SCHEDULER_TYPE_STEP) {
		return learning_rate * Math::pow(decay_constant, int((1 + epoch) / drop_rate)); // Utilizing an explicit int conversion implicitly takes the floor.
	} else if (_lr_scheduler == SCHEDULER_TYPE_EXPONENTIAL) {
		return learning_rate * Math::exp(-decay_constant * epoch);
	}

	return learning_rate;
}

real_t MLPPANN::cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;

	real_t total_reg_term = 0;

	if (!_network.empty()) {
		for (int i = 0; i < _network.size() - 1; i++) {
			Ref<MLPPHiddenLayer> layer = _network[i];

			total_reg_term += regularization.reg_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg());
		}
	}

	return mlpp_cost.run_cost_norm_vector(_output_layer->get_cost(), y_hat, y) + total_reg_term + regularization.reg_termv(_output_layer->get_weights(), _output_layer->get_lambda(), _output_layer->get_alpha(), _output_layer->get_reg());
}

void MLPPANN::forward_pass() {
	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[0];

		layer->set_input(_input_set);
		layer->forward_pass();

		for (int i = 1; i < _network.size(); i++) {
			layer = _network[i];
			Ref<MLPPHiddenLayer> prev_layer = _network[i - 1];

			layer->set_input(prev_layer->get_a());
			layer->forward_pass();
		}

		_output_layer->set_input(_network.write[_network.size() - 1]->get_a());
	} else {
		_output_layer->set_input(_input_set);
	}

	_output_layer->forward_pass();

	_y_hat = _output_layer->get_a();
}

void MLPPANN::update_parameters(const Vector<Ref<MLPPMatrix>> &hidden_layer_updations, const Ref<MLPPVector> &output_layer_updation, real_t learning_rate) {
	_output_layer->set_weights(_output_layer->get_weights()->subn(output_layer_updation));
	_output_layer->set_bias(_output_layer->get_bias() - learning_rate * _output_layer->get_delta()->sum_elements() / _n);

	Ref<MLPPMatrix> slice;

	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[_network.size() - 1];

		slice = hidden_layer_updations[0];

		layer->set_weights(layer->get_weights()->subn(slice));
		layer->set_bias(layer->get_bias()->subtract_matrix_rowsn(layer->get_delta()->scalar_multiplyn(learning_rate / _n)));

		for (int i = _network.size() - 2; i >= 0; i--) {
			layer = _network[i];

			slice = hidden_layer_updations[(_network.size() - 2) - i + 1];

			layer->set_weights(layer->get_weights()->subn(slice));
			layer->set_bias(layer->get_bias()->subtract_matrix_rowsn(layer->get_delta()->scalar_multiplyn(learning_rate / _n)));
		}
	}
}

MLPPANN::ComputeGradientsResult MLPPANN::compute_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &_output_set) {
	// std::cout << "BEGIN" << std::endl;
	MLPPCost mlpp_cost;
	MLPPActivation avn;

	MLPPReg regularization;

	ComputeGradientsResult res;

	_output_layer->set_delta(mlpp_cost.run_cost_deriv_vector(_output_layer->get_cost(), y_hat, _output_set)->hadamard_productn(avn.run_activation_deriv_vector(_output_layer->get_activation(), _output_layer->get_z())));

	res.output_w_grad = _output_layer->get_input()->transposen()->mult_vec(_output_layer->get_delta());
	res.output_w_grad->add(regularization.reg_deriv_termv(_output_layer->get_weights(), _output_layer->get_lambda(), _output_layer->get_alpha(), _output_layer->get_reg()));

	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[_network.size() - 1];
		layer->set_delta(_output_layer->get_delta()->outer_product(_output_layer->get_weights())->hadamard_productn(avn.run_activation_deriv_matrix(layer->get_activation(), layer->get_z())));

		Ref<MLPPMatrix> hidden_layer_w_grad = layer->get_input()->transposen()->multn(layer->get_delta());

		// Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		res.cumulative_hidden_layer_w_grad.push_back(hidden_layer_w_grad->addn(regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg())));

		for (int i = _network.size() - 2; i >= 0; i--) {
			layer = _network[i];
			Ref<MLPPHiddenLayer> next_layer = _network[i + 1];

			layer->set_delta(next_layer->get_delta()->multn(next_layer->get_weights()->transposen())->hadamard_productn(avn.run_activation_deriv_matrix(layer->get_activation(), layer->get_z())));
			hidden_layer_w_grad = layer->get_input()->transposen()->multn(layer->get_delta());
			res.cumulative_hidden_layer_w_grad.push_back(hidden_layer_w_grad->addn(regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}

	return res;
}

void MLPPANN::print_ui(int epoch, real_t cost_prev, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &p_output_set) {
	MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, p_output_set));

	PLOG_MSG("Layer " + itos(_network.size() + 1) + ": ");
	MLPPUtilities::print_ui_vb(_output_layer->get_weights(), _output_layer->get_bias());

	if (!_network.empty()) {
		for (int i = _network.size() - 1; i >= 0; i--) {
			Ref<MLPPHiddenLayer> layer = _network[i];

			PLOG_MSG("Layer " + itos(i + 1) + ": ");
			MLPPUtilities::print_ui_mb(layer->get_weights(), layer->get_bias());
		}
	}
}

void MLPPANN::_bind_methods() {
}
