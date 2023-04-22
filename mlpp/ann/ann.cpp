//
//  ANN.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "ann.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"
#include "core/log/logger.h"

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

		grads.cumulative_hidden_layer_w_grad = alg.scalar_multiply_vm(learning_rate / _n, grads.cumulative_hidden_layer_w_grad);
		grads.output_w_grad = alg.scalar_multiplynv(learning_rate / _n, grads.output_w_grad);
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

		_input_set->get_row_into_mlpp_vector(output_index, input_set_row_tmp);
		real_t output_set_element = _output_set->get_element(output_index);
		output_set_row_tmp->set_element(0, output_set_element);

		real_t y_hat = model_test(input_set_row_tmp);
		y_hat_row_tmp->set_element(0, y_hat);

		cost_prev = cost(y_hat_row_tmp, output_set_row_tmp);

		ComputeGradientsResult grads = compute_gradients(y_hat_row_tmp, output_set_row_tmp);

		grads.cumulative_hidden_layer_w_grad = alg.scalar_multiply_vm(learning_rate / _n, grads.cumulative_hidden_layer_w_grad);
		grads.output_w_grad = alg.scalar_multiplynv(learning_rate / _n, grads.output_w_grad);

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

			grads.cumulative_hidden_layer_w_grad = alg.scalar_multiply_vm(learning_rate / _n, grads.cumulative_hidden_layer_w_grad);
			grads.output_w_grad = alg.scalar_multiplynv(learning_rate / _n, grads.output_w_grad);

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
				v_hidden = alg.resize_vt(v_hidden, grads.cumulative_hidden_layer_w_grad);
			}

			if (v_output->size() == 0) {
				v_output->resize(grads.output_w_grad->size());
			}

			if (nag) { // "Aposterori" calculation
				update_parameters(v_hidden, v_output, 0); // DON'T update bias.
			}

			v_hidden = alg.addition_vt(alg.scalar_multiply_vm(gamma, v_hidden), alg.scalar_multiply_vm(learning_rate / _n, grads.cumulative_hidden_layer_w_grad));
			v_output = alg.additionnv(alg.scalar_multiplynv(gamma, v_output), alg.scalar_multiplynv(learning_rate / _n, grads.output_w_grad));

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
				v_hidden = alg.resize_vt(v_hidden, grads.cumulative_hidden_layer_w_grad);
			}

			if (v_output->size() == 0) {
				v_output->resize(grads.output_w_grad->size());
			}

			v_hidden = alg.addition_vt(v_hidden, alg.exponentiate_vt(grads.cumulative_hidden_layer_w_grad, 2));
			v_output = alg.additionnv(v_output, alg.exponentiatenv(grads.output_w_grad, 2));

			Vector<Ref<MLPPMatrix>> hidden_layer_updations = alg.scalar_multiply_vm(learning_rate / _n, alg.element_wise_divisionnv_vt(grads.cumulative_hidden_layer_w_grad, alg.scalar_add_vm(e, alg.sqrt_vt(v_hidden))));
			Ref<MLPPVector> output_layer_updation = alg.scalar_multiplynv(learning_rate / _n, alg.element_wise_divisionnv(grads.output_w_grad, alg.scalar_addnv(e, alg.sqrtnv(v_output))));

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
				v_hidden = alg.resize_vt(v_hidden, grads.cumulative_hidden_layer_w_grad);
			}

			if (v_output->size() == 0) {
				v_output->resize(grads.output_w_grad->size());
			}

			v_hidden = alg.addition_vt(alg.scalar_multiply_vm(1 - b1, v_hidden), alg.scalar_multiply_vm(b1, alg.exponentiate_vt(grads.cumulative_hidden_layer_w_grad, 2)));
			v_output = alg.additionnv(v_output, alg.exponentiatenv(grads.output_w_grad, 2));

			Vector<Ref<MLPPMatrix>> hidden_layer_updations = alg.scalar_multiply_vm(learning_rate / _n, alg.element_wise_divisionnv_vt(grads.cumulative_hidden_layer_w_grad, alg.scalar_add_vm(e, alg.sqrt_vt(v_hidden))));
			Ref<MLPPVector> output_layer_updation = alg.scalar_multiplynv(learning_rate / _n, alg.element_wise_divisionnv(grads.output_w_grad, alg.scalar_addnv(e, alg.sqrtnv(v_output))));

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
				m_hidden = alg.resize_vt(m_hidden, grads.cumulative_hidden_layer_w_grad);
				v_hidden = alg.resize_vt(v_hidden, grads.cumulative_hidden_layer_w_grad);
			}

			if (m_output->size() == 0 && v_output->size()) {
				m_output->resize(grads.output_w_grad->size());
				v_output->resize(grads.output_w_grad->size());
			}

			m_hidden = alg.addition_vt(alg.scalar_multiply_vm(b1, m_hidden), alg.scalar_multiply_vm(1 - b1, grads.cumulative_hidden_layer_w_grad));
			v_hidden = alg.addition_vt(alg.scalar_multiply_vm(b2, v_hidden), alg.scalar_multiply_vm(1 - b2, alg.exponentiate_vt(grads.cumulative_hidden_layer_w_grad, 2)));

			m_output = alg.additionnv(alg.scalar_multiplynv(b1, m_output), alg.scalar_multiplynv(1 - b1, grads.output_w_grad));
			v_output = alg.additionnv(alg.scalar_multiplynv(b2, v_output), alg.scalar_multiplynv(1 - b2, alg.exponentiatenv(grads.output_w_grad, 2)));

			Vector<Ref<MLPPMatrix>> m_hidden_hat = alg.scalar_multiply_vm(1 / (1 - Math::pow(b1, epoch)), m_hidden);
			Vector<Ref<MLPPMatrix>> v_hidden_hat = alg.scalar_multiply_vm(1 / (1 - Math::pow(b2, epoch)), v_hidden);

			Ref<MLPPVector> m_output_hat = alg.scalar_multiplynv(1 / (1 - Math::pow(b1, epoch)), m_output);
			Ref<MLPPVector> v_output_hat = alg.scalar_multiplynv(1 / (1 - Math::pow(b2, epoch)), v_output);

			Vector<Ref<MLPPMatrix>> hidden_layer_updations = alg.scalar_multiply_vm(learning_rate / _n, alg.element_wise_divisionnv_vt(m_hidden_hat, alg.scalar_add_vm(e, alg.sqrt_vt(v_hidden_hat))));
			Ref<MLPPVector> output_layer_updation = alg.scalar_multiplynv(learning_rate / _n, alg.element_wise_divisionnv(m_output_hat, alg.scalar_addnv(e, alg.sqrtnv(v_output_hat))));

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
	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = model_set_test(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			ComputeGradientsResult grads = compute_gradients(y_hat, current_output_batch);

			if (!_network.empty() && m_hidden.empty() && u_hidden.empty()) { // Initing our tensor
				m_hidden = alg.resize_vt(m_hidden, grads.cumulative_hidden_layer_w_grad);
				u_hidden = alg.resize_vt(u_hidden, grads.cumulative_hidden_layer_w_grad);
			}

			if (m_output->size() == 0 && u_output->size() == 0) {
				m_output->resize(grads.output_w_grad->size());
				u_output->resize(grads.output_w_grad->size());
			}

			m_hidden = alg.addition_vt(alg.scalar_multiply_vm(b1, m_hidden), alg.scalar_multiply_vm(1 - b1, grads.cumulative_hidden_layer_w_grad));
			u_hidden = alg.max_vt(alg.scalar_multiply_vm(b2, u_hidden), alg.abs_vt(grads.cumulative_hidden_layer_w_grad));

			m_output = alg.additionnv(alg.scalar_multiplynv(b1, m_output), alg.scalar_multiplynv(1 - b1, grads.output_w_grad));
			u_output = alg.maxnvv(alg.scalar_multiplynv(b2, u_output), alg.absv(grads.output_w_grad));

			Vector<Ref<MLPPMatrix>> m_hidden_hat = alg.scalar_multiply_vm(1 / (1 - Math::pow(b1, epoch)), m_hidden);

			Ref<MLPPVector> m_output_hat = alg.scalar_multiplynv(1 / (1 - Math::pow(b1, epoch)), m_output);

			Vector<Ref<MLPPMatrix>> hidden_layer_updations = alg.scalar_multiply_vm(learning_rate / _n, alg.element_wise_divisionnv_vt(m_hidden_hat, alg.scalar_add_vm(e, u_hidden)));
			Ref<MLPPVector> output_layer_updation = alg.scalar_multiplynv(learning_rate / _n, alg.element_wise_divisionnv(m_output_hat, alg.scalar_addnv(e, u_output)));

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
	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = model_set_test(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			ComputeGradientsResult grads = compute_gradients(y_hat, current_output_batch);

			if (!_network.empty() && m_hidden.empty() && v_hidden.empty()) { // Initing our tensor
				m_hidden = alg.resize_vt(m_hidden, grads.cumulative_hidden_layer_w_grad);
				v_hidden = alg.resize_vt(v_hidden, grads.cumulative_hidden_layer_w_grad);
			}

			if (m_output->size() == 0 && v_output->size() == 0) {
				m_output->resize(grads.output_w_grad->size());
				v_output->resize(grads.output_w_grad->size());
			}

			m_hidden = alg.addition_vt(alg.scalar_multiply_vm(b1, m_hidden), alg.scalar_multiply_vm(1 - b1, grads.cumulative_hidden_layer_w_grad));
			v_hidden = alg.addition_vt(alg.scalar_multiply_vm(b2, v_hidden), alg.scalar_multiply_vm(1 - b2, alg.exponentiate_vt(grads.cumulative_hidden_layer_w_grad, 2)));

			m_output = alg.additionnv(alg.scalar_multiplynv(b1, m_output), alg.scalar_multiplynv(1 - b1, grads.output_w_grad));
			v_output = alg.additionnv(alg.scalar_multiplynv(b2, v_output), alg.scalar_multiplynv(1 - b2, alg.exponentiatenv(grads.output_w_grad, 2)));

			Vector<Ref<MLPPMatrix>> m_hidden_hat = alg.scalar_multiply_vm(1 / (1.0 - Math::pow(b1, epoch)), m_hidden);
			Vector<Ref<MLPPMatrix>> v_hidden_hat = alg.scalar_multiply_vm(1 / (1.0 - Math::pow(b2, epoch)), v_hidden);
			Vector<Ref<MLPPMatrix>> m_hidden_final = alg.addition_vt(alg.scalar_multiply_vm(b1, m_hidden_hat), alg.scalar_multiply_vm((1 - b1) / (1 - Math::pow(b1, epoch)), grads.cumulative_hidden_layer_w_grad));

			Ref<MLPPVector> m_output_hat = alg.scalar_multiplynv(1 / (1.0 - Math::pow(b1, epoch)), m_output);
			Ref<MLPPVector> v_output_hat = alg.scalar_multiplynv(1 / (1.0 - Math::pow(b2, epoch)), v_output);
			Ref<MLPPVector> m_output_final = alg.additionnv(alg.scalar_multiplynv(b1, m_output_hat), alg.scalar_multiplynv((1 - b1) / (1.0 - Math::pow(b1, epoch)), grads.output_w_grad));

			Vector<Ref<MLPPMatrix>> hidden_layer_updations = alg.scalar_multiply_vm(learning_rate / _n, alg.element_wise_divisionnv_vt(m_hidden_final, alg.scalar_add_vm(e, alg.sqrt_vt(v_hidden_hat))));
			Ref<MLPPVector> output_layer_updation = alg.scalar_multiplynv(learning_rate / _n, alg.element_wise_divisionnvnm(m_output_final, alg.scalar_addnv(e, alg.sqrtnv(v_output_hat))));

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

	Ref<MLPPVector> v_output_hat;
	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch = batches.output_sets[i];

			Ref<MLPPVector> y_hat = model_set_test(current_input_batch);
			cost_prev = cost(y_hat, current_output_batch);

			ComputeGradientsResult grads = compute_gradients(y_hat, current_output_batch);

			if (!_network.empty() && m_hidden.empty() && v_hidden.empty()) { // Initing our tensor
				m_hidden = alg.resize_vt(m_hidden, grads.cumulative_hidden_layer_w_grad);
				v_hidden = alg.resize_vt(v_hidden, grads.cumulative_hidden_layer_w_grad);
				v_hidden_hat = alg.resize_vt(v_hidden_hat, grads.cumulative_hidden_layer_w_grad);
			}

			if (m_output->size() == 0 && v_output->size() == 0) {
				m_output->resize(grads.output_w_grad->size());
				v_output->resize(grads.output_w_grad->size());
				v_output_hat->resize(grads.output_w_grad->size());
			}

			m_hidden = alg.addition_vt(alg.scalar_multiply_vm(b1, m_hidden), alg.scalar_multiply_vm(1 - b1, grads.cumulative_hidden_layer_w_grad));
			v_hidden = alg.addition_vt(alg.scalar_multiply_vm(b2, v_hidden), alg.scalar_multiply_vm(1 - b2, alg.exponentiate_vt(grads.cumulative_hidden_layer_w_grad, 2)));

			m_output = alg.additionnv(alg.scalar_multiplynv(b1, m_output), alg.scalar_multiplynv(1 - b1, grads.output_w_grad));
			v_output = alg.additionnv(alg.scalar_multiplynv(b2, v_output), alg.scalar_multiplynv(1 - b2, alg.exponentiatenv(grads.output_w_grad, 2)));

			v_hidden_hat = alg.max_vt(v_hidden_hat, v_hidden);
			v_output_hat = alg.maxnvv(v_output_hat, v_output);

			Vector<Ref<MLPPMatrix>> hidden_layer_updations = alg.scalar_multiply_vm(learning_rate / _n, alg.element_wise_divisionnv_vt(m_hidden, alg.scalar_add_vm(e, alg.sqrt_vt(v_hidden_hat))));
			Ref<MLPPVector> output_layer_updation = alg.scalar_multiplynv(learning_rate / _n, alg.element_wise_divisionnv(m_output, alg.scalar_addnv(e, alg.sqrtnv(v_output_hat))));

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
	MLPPLinAlg alg;

	_output_layer->set_weights(alg.subtractionnv(_output_layer->get_weights(), output_layer_updation));
	_output_layer->set_bias(_output_layer->get_bias() - learning_rate * alg.sum_elementsv(_output_layer->get_delta()) / _n);

	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[_network.size() - 1];

		layer->set_weights(alg.subtractionnm(layer->get_weights(), hidden_layer_updations[0]));
		layer->set_bias(alg.subtract_matrix_rows(layer->get_bias(), alg.scalar_multiplynm(learning_rate / _n, layer->get_delta())));

		for (int i = _network.size() - 2; i >= 0; i--) {
			layer = _network[i];

			layer->set_weights(alg.subtractionnm(layer->get_weights(), hidden_layer_updations[(_network.size() - 2) - i + 1]));
			layer->set_bias(alg.subtract_matrix_rows(layer->get_bias(), alg.scalar_multiplynm(learning_rate / _n, layer->get_delta())));
		}
	}
}

MLPPANN::ComputeGradientsResult MLPPANN::compute_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &_output_set) {
	// std::cout << "BEGIN" << std::endl;
	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	ComputeGradientsResult res;

	_output_layer->set_delta(alg.hadamard_productnv(mlpp_cost.run_cost_deriv_vector(_output_layer->get_cost(), y_hat, _output_set), avn.run_activation_deriv_vector(_output_layer->get_activation(), _output_layer->get_z())));

	res.output_w_grad = alg.mat_vec_multv(alg.transposenm(_output_layer->get_input()), _output_layer->get_delta());
	res.output_w_grad = alg.additionnv(res.output_w_grad, regularization.reg_deriv_termv(_output_layer->get_weights(), _output_layer->get_lambda(), _output_layer->get_alpha(), _output_layer->get_reg()));

	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[_network.size() - 1];

		layer->set_delta(alg.hadamard_productnm(alg.outer_product(_output_layer->get_delta(), _output_layer->get_weights()), avn.run_activation_deriv_vector(layer->get_activation(), layer->get_z())));

		Ref<MLPPMatrix> hidden_layer_w_grad = alg.matmultnm(alg.transposenm(layer->get_input()), layer->get_delta());

		res.cumulative_hidden_layer_w_grad.push_back(alg.additionnm(hidden_layer_w_grad, regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

		for (int i = _network.size() - 2; i >= 0; i--) {
			layer = _network[i];
			Ref<MLPPHiddenLayer> next_layer = _network[i + 1];

			layer->set_delta(alg.hadamard_productnm(alg.matmultnm(next_layer->get_delta(), alg.transposenm(next_layer->get_weights())), avn.run_activation_deriv_vector(layer->get_activation(), layer->get_z())));
			hidden_layer_w_grad = alg.matmultnm(alg.transposenm(layer->get_input()), layer->get_delta());
			res.cumulative_hidden_layer_w_grad.push_back(alg.additionnm(hidden_layer_w_grad, regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
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
