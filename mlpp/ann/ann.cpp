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

#include <cmath>
#include <iostream>
#include <random>

std::vector<real_t> MLPPANN::model_set_test(std::vector<std::vector<real_t>> X) {
	if (!_network.empty()) {
		_network[0].input = X;
		_network[0].forwardPass();

		for (uint32_t i = 1; i < _network.size(); i++) {
			_network[i].input = _network[i - 1].a;
			_network[i].forwardPass();
		}
		_output_layer->input = _network[_network.size() - 1].a;
	} else {
		_output_layer->input = X;
	}

	_output_layer->forwardPass();

	return _output_layer->a;
}

real_t MLPPANN::model_test(std::vector<real_t> x) {
	if (!_network.empty()) {
		_network[0].Test(x);
		for (uint32_t i = 1; i < _network.size(); i++) {
			_network[i].Test(_network[i - 1].a_test);
		}
		_output_layer->Test(_network[_network.size() - 1].a_test);
	} else {
		_output_layer->Test(x);
	}
	return _output_layer->a_test;
}

void MLPPANN::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	MLPPCost mlpp_cost;
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	real_t initial_learning_rate = learning_rate;

	alg.printMatrix(_network[_network.size() - 1].weights);
	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		cost_prev = cost(_y_hat, _output_set);

		auto grads = compute_gradients(_y_hat, _output_set);
		auto cumulative_hidden_layer_w_grad = std::get<0>(grads);
		auto output_w_grad = std::get<1>(grads);

		cumulative_hidden_layer_w_grad = alg.scalarMultiply(learning_rate / _n, cumulative_hidden_layer_w_grad);
		output_w_grad = alg.scalarMultiply(learning_rate / _n, output_w_grad);
		update_parameters(cumulative_hidden_layer_w_grad, output_w_grad, learning_rate); // subject to change. may want bias to have this matrix too.

		std::cout << learning_rate << std::endl;

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

	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(_n - 1));
		int outputIndex = distribution(generator);

		std::vector<real_t> y_hat = model_set_test({ _input_set[outputIndex] });
		cost_prev = cost({ y_hat }, { _output_set[outputIndex] });

		auto grads = compute_gradients(y_hat, { _output_set[outputIndex] });
		auto cumulative_hidden_layer_w_grad = std::get<0>(grads);
		auto output_w_grad = std::get<1>(grads);

		cumulative_hidden_layer_w_grad = alg.scalarMultiply(learning_rate / _n, cumulative_hidden_layer_w_grad);
		output_w_grad = alg.scalarMultiply(learning_rate / _n, output_w_grad);

		update_parameters(cumulative_hidden_layer_w_grad, output_w_grad, learning_rate); // subject to change. may want bias to have this matrix too.
		y_hat = model_set_test({ _input_set[outputIndex] });

		if (ui) {
			print_ui(epoch, cost_prev, y_hat, { _output_set[outputIndex] });
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

	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = model_set_test(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			auto grads = compute_gradients(y_hat, output_mini_batches[i]);
			auto cumulative_hidden_layer_w_grad = std::get<0>(grads);
			auto output_w_grad = std::get<1>(grads);

			cumulative_hidden_layer_w_grad = alg.scalarMultiply(learning_rate / _n, cumulative_hidden_layer_w_grad);
			output_w_grad = alg.scalarMultiply(learning_rate / _n, output_w_grad);

			update_parameters(cumulative_hidden_layer_w_grad, output_w_grad, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(input_mini_batches[i]);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, output_mini_batches[i]);
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

	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> v_hidden;

	std::vector<real_t> v_output;
	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = model_set_test(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			auto grads = compute_gradients(y_hat, output_mini_batches[i]);
			auto cumulative_hidden_layer_w_grad = std::get<0>(grads);
			auto output_w_grad = std::get<1>(grads);

			if (!_network.empty() && v_hidden.empty()) { // Initing our tensor
				v_hidden = alg.resize(v_hidden, cumulative_hidden_layer_w_grad);
			}

			if (v_output.empty()) {
				v_output.resize(output_w_grad.size());
			}

			if (nag) { // "Aposterori" calculation
				update_parameters(v_hidden, v_output, 0); // DON'T update bias.
			}

			v_hidden = alg.addition(alg.scalarMultiply(gamma, v_hidden), alg.scalarMultiply(learning_rate / _n, cumulative_hidden_layer_w_grad));

			v_output = alg.addition(alg.scalarMultiply(gamma, v_output), alg.scalarMultiply(learning_rate / _n, output_w_grad));

			update_parameters(v_hidden, v_output, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(input_mini_batches[i]);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, output_mini_batches[i]);
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

	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> v_hidden;

	std::vector<real_t> v_output;
	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);

		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = model_set_test(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			auto grads = compute_gradients(y_hat, output_mini_batches[i]);
			auto cumulative_hidden_layer_w_grad = std::get<0>(grads);
			auto output_w_grad = std::get<1>(grads);

			if (!_network.empty() && v_hidden.empty()) { // Initing our tensor
				v_hidden = alg.resize(v_hidden, cumulative_hidden_layer_w_grad);
			}

			if (v_output.empty()) {
				v_output.resize(output_w_grad.size());
			}

			v_hidden = alg.addition(v_hidden, alg.exponentiate(cumulative_hidden_layer_w_grad, 2));

			v_output = alg.addition(v_output, alg.exponentiate(output_w_grad, 2));

			std::vector<std::vector<std::vector<real_t>>> hidden_layer_updations = alg.scalarMultiply(learning_rate / _n, alg.elementWiseDivision(cumulative_hidden_layer_w_grad, alg.scalarAdd(e, alg.sqrt(v_hidden))));
			std::vector<real_t> output_layer_updation = alg.scalarMultiply(learning_rate / _n, alg.elementWiseDivision(output_w_grad, alg.scalarAdd(e, alg.sqrt(v_output))));

			update_parameters(hidden_layer_updations, output_layer_updation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(input_mini_batches[i]);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, output_mini_batches[i]);
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

	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> v_hidden;

	std::vector<real_t> v_output;
	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = model_set_test(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			auto grads = compute_gradients(y_hat, output_mini_batches[i]);
			auto cumulative_hidden_layer_w_grad = std::get<0>(grads);
			auto output_w_grad = std::get<1>(grads);

			if (!_network.empty() && v_hidden.empty()) { // Initing our tensor
				v_hidden = alg.resize(v_hidden, cumulative_hidden_layer_w_grad);
			}

			if (v_output.empty()) {
				v_output.resize(output_w_grad.size());
			}

			v_hidden = alg.addition(alg.scalarMultiply(1 - b1, v_hidden), alg.scalarMultiply(b1, alg.exponentiate(cumulative_hidden_layer_w_grad, 2)));

			v_output = alg.addition(v_output, alg.exponentiate(output_w_grad, 2));

			std::vector<std::vector<std::vector<real_t>>> hidden_layer_updations = alg.scalarMultiply(learning_rate / _n, alg.elementWiseDivision(cumulative_hidden_layer_w_grad, alg.scalarAdd(e, alg.sqrt(v_hidden))));
			std::vector<real_t> output_layer_updation = alg.scalarMultiply(learning_rate / _n, alg.elementWiseDivision(output_w_grad, alg.scalarAdd(e, alg.sqrt(v_output))));

			update_parameters(hidden_layer_updations, output_layer_updation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(input_mini_batches[i]);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, output_mini_batches[i]);
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

	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> m_hidden;
	std::vector<std::vector<std::vector<real_t>>> v_hidden;

	std::vector<real_t> m_output;
	std::vector<real_t> v_output;
	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = model_set_test(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			auto grads = compute_gradients(y_hat, output_mini_batches[i]);
			auto cumulative_hidden_layer_w_grad = std::get<0>(grads);
			auto output_w_grad = std::get<1>(grads);

			if (!_network.empty() && m_hidden.empty() && v_hidden.empty()) { // Initing our tensor
				m_hidden = alg.resize(m_hidden, cumulative_hidden_layer_w_grad);
				v_hidden = alg.resize(v_hidden, cumulative_hidden_layer_w_grad);
			}

			if (m_output.empty() && v_output.empty()) {
				m_output.resize(output_w_grad.size());
				v_output.resize(output_w_grad.size());
			}

			m_hidden = alg.addition(alg.scalarMultiply(b1, m_hidden), alg.scalarMultiply(1 - b1, cumulative_hidden_layer_w_grad));
			v_hidden = alg.addition(alg.scalarMultiply(b2, v_hidden), alg.scalarMultiply(1 - b2, alg.exponentiate(cumulative_hidden_layer_w_grad, 2)));

			m_output = alg.addition(alg.scalarMultiply(b1, m_output), alg.scalarMultiply(1 - b1, output_w_grad));
			v_output = alg.addition(alg.scalarMultiply(b2, v_output), alg.scalarMultiply(1 - b2, alg.exponentiate(output_w_grad, 2)));

			std::vector<std::vector<std::vector<real_t>>> m_hidden_hat = alg.scalarMultiply(1 / (1 - std::pow(b1, epoch)), m_hidden);
			std::vector<std::vector<std::vector<real_t>>> v_hidden_hat = alg.scalarMultiply(1 / (1 - std::pow(b2, epoch)), v_hidden);

			std::vector<real_t> m_output_hat = alg.scalarMultiply(1 / (1 - std::pow(b1, epoch)), m_output);
			std::vector<real_t> v_output_hat = alg.scalarMultiply(1 / (1 - std::pow(b2, epoch)), v_output);

			std::vector<std::vector<std::vector<real_t>>> hidden_layer_updations = alg.scalarMultiply(learning_rate / _n, alg.elementWiseDivision(m_hidden_hat, alg.scalarAdd(e, alg.sqrt(v_hidden_hat))));
			std::vector<real_t> output_layer_updation = alg.scalarMultiply(learning_rate / _n, alg.elementWiseDivision(m_output_hat, alg.scalarAdd(e, alg.sqrt(v_output_hat))));

			update_parameters(hidden_layer_updations, output_layer_updation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(input_mini_batches[i]);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, output_mini_batches[i]);
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

	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> m_hidden;
	std::vector<std::vector<std::vector<real_t>>> u_hidden;

	std::vector<real_t> m_output;
	std::vector<real_t> u_output;
	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = model_set_test(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			auto grads = compute_gradients(y_hat, output_mini_batches[i]);
			auto cumulative_hidden_layer_w_grad = std::get<0>(grads);
			auto output_w_grad = std::get<1>(grads);

			if (!_network.empty() && m_hidden.empty() && u_hidden.empty()) { // Initing our tensor
				m_hidden = alg.resize(m_hidden, cumulative_hidden_layer_w_grad);
				u_hidden = alg.resize(u_hidden, cumulative_hidden_layer_w_grad);
			}

			if (m_output.empty() && u_output.empty()) {
				m_output.resize(output_w_grad.size());
				u_output.resize(output_w_grad.size());
			}

			m_hidden = alg.addition(alg.scalarMultiply(b1, m_hidden), alg.scalarMultiply(1 - b1, cumulative_hidden_layer_w_grad));
			u_hidden = alg.max(alg.scalarMultiply(b2, u_hidden), alg.abs(cumulative_hidden_layer_w_grad));

			m_output = alg.addition(alg.scalarMultiply(b1, m_output), alg.scalarMultiply(1 - b1, output_w_grad));
			u_output = alg.max(alg.scalarMultiply(b2, u_output), alg.abs(output_w_grad));

			std::vector<std::vector<std::vector<real_t>>> m_hidden_hat = alg.scalarMultiply(1 / (1 - std::pow(b1, epoch)), m_hidden);

			std::vector<real_t> m_output_hat = alg.scalarMultiply(1 / (1 - std::pow(b1, epoch)), m_output);

			std::vector<std::vector<std::vector<real_t>>> hidden_layer_updations = alg.scalarMultiply(learning_rate / _n, alg.elementWiseDivision(m_hidden_hat, alg.scalarAdd(e, u_hidden)));
			std::vector<real_t> output_layer_updation = alg.scalarMultiply(learning_rate / _n, alg.elementWiseDivision(m_output_hat, alg.scalarAdd(e, u_output)));

			update_parameters(hidden_layer_updations, output_layer_updation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(input_mini_batches[i]);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, output_mini_batches[i]);
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

	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> m_hidden;
	std::vector<std::vector<std::vector<real_t>>> v_hidden;

	std::vector<real_t> m_output;
	std::vector<real_t> v_output;
	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = model_set_test(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			auto grads = compute_gradients(y_hat, output_mini_batches[i]);
			auto cumulative_hidden_layer_w_grad = std::get<0>(grads);
			auto output_w_grad = std::get<1>(grads);

			if (!_network.empty() && m_hidden.empty() && v_hidden.empty()) { // Initing our tensor
				m_hidden = alg.resize(m_hidden, cumulative_hidden_layer_w_grad);
				v_hidden = alg.resize(v_hidden, cumulative_hidden_layer_w_grad);
			}

			if (m_output.empty() && v_output.empty()) {
				m_output.resize(output_w_grad.size());
				v_output.resize(output_w_grad.size());
			}

			m_hidden = alg.addition(alg.scalarMultiply(b1, m_hidden), alg.scalarMultiply(1 - b1, cumulative_hidden_layer_w_grad));
			v_hidden = alg.addition(alg.scalarMultiply(b2, v_hidden), alg.scalarMultiply(1 - b2, alg.exponentiate(cumulative_hidden_layer_w_grad, 2)));

			m_output = alg.addition(alg.scalarMultiply(b1, m_output), alg.scalarMultiply(1 - b1, output_w_grad));
			v_output = alg.addition(alg.scalarMultiply(b2, v_output), alg.scalarMultiply(1 - b2, alg.exponentiate(output_w_grad, 2)));

			std::vector<std::vector<std::vector<real_t>>> m_hidden_hat = alg.scalarMultiply(1 / (1 - std::pow(b1, epoch)), m_hidden);
			std::vector<std::vector<std::vector<real_t>>> v_hidden_hat = alg.scalarMultiply(1 / (1 - std::pow(b2, epoch)), v_hidden);
			std::vector<std::vector<std::vector<real_t>>> m_hidden_final = alg.addition(alg.scalarMultiply(b1, m_hidden_hat), alg.scalarMultiply((1 - b1) / (1 - std::pow(b1, epoch)), cumulative_hidden_layer_w_grad));

			std::vector<real_t> m_output_hat = alg.scalarMultiply(1 / (1 - std::pow(b1, epoch)), m_output);
			std::vector<real_t> v_output_hat = alg.scalarMultiply(1 / (1 - std::pow(b2, epoch)), v_output);
			std::vector<real_t> m_output_final = alg.addition(alg.scalarMultiply(b1, m_output_hat), alg.scalarMultiply((1 - b1) / (1 - std::pow(b1, epoch)), output_w_grad));

			std::vector<std::vector<std::vector<real_t>>> hidden_layer_updations = alg.scalarMultiply(learning_rate / _n, alg.elementWiseDivision(m_hidden_final, alg.scalarAdd(e, alg.sqrt(v_hidden_hat))));
			std::vector<real_t> output_layer_updation = alg.scalarMultiply(learning_rate / _n, alg.elementWiseDivision(m_output_final, alg.scalarAdd(e, alg.sqrt(v_output_hat))));

			update_parameters(hidden_layer_updations, output_layer_updation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(input_mini_batches[i]);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, output_mini_batches[i]);
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

	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto input_mini_batches = std::get<0>(batches);
	auto output_mini_batches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> m_hidden;
	std::vector<std::vector<std::vector<real_t>>> v_hidden;

	std::vector<std::vector<std::vector<real_t>>> v_hidden_hat;

	std::vector<real_t> m_output;
	std::vector<real_t> v_output;

	std::vector<real_t> v_output_hat;
	while (true) {
		learning_rate = apply_learning_rate_scheduler(initial_learning_rate, _decay_constant, epoch, _drop_rate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = model_set_test(input_mini_batches[i]);
			cost_prev = cost(y_hat, output_mini_batches[i]);

			auto grads = compute_gradients(y_hat, output_mini_batches[i]);
			auto cumulative_hidden_layer_w_grad = std::get<0>(grads);
			auto output_w_grad = std::get<1>(grads);

			if (!_network.empty() && m_hidden.empty() && v_hidden.empty()) { // Initing our tensor
				m_hidden = alg.resize(m_hidden, cumulative_hidden_layer_w_grad);
				v_hidden = alg.resize(v_hidden, cumulative_hidden_layer_w_grad);
				v_hidden_hat = alg.resize(v_hidden_hat, cumulative_hidden_layer_w_grad);
			}

			if (m_output.empty() && v_output.empty()) {
				m_output.resize(output_w_grad.size());
				v_output.resize(output_w_grad.size());
				v_output_hat.resize(output_w_grad.size());
			}

			m_hidden = alg.addition(alg.scalarMultiply(b1, m_hidden), alg.scalarMultiply(1 - b1, cumulative_hidden_layer_w_grad));
			v_hidden = alg.addition(alg.scalarMultiply(b2, v_hidden), alg.scalarMultiply(1 - b2, alg.exponentiate(cumulative_hidden_layer_w_grad, 2)));

			m_output = alg.addition(alg.scalarMultiply(b1, m_output), alg.scalarMultiply(1 - b1, output_w_grad));
			v_output = alg.addition(alg.scalarMultiply(b2, v_output), alg.scalarMultiply(1 - b2, alg.exponentiate(output_w_grad, 2)));

			v_hidden_hat = alg.max(v_hidden_hat, v_hidden);

			v_output_hat = alg.max(v_output_hat, v_output);

			std::vector<std::vector<std::vector<real_t>>> hidden_layer_updations = alg.scalarMultiply(learning_rate / _n, alg.elementWiseDivision(m_hidden, alg.scalarAdd(e, alg.sqrt(v_hidden_hat))));
			std::vector<real_t> output_layer_updation = alg.scalarMultiply(learning_rate / _n, alg.elementWiseDivision(m_output, alg.scalarAdd(e, alg.sqrt(v_output_hat))));

			update_parameters(hidden_layer_updations, output_layer_updation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = model_set_test(input_mini_batches[i]);

			if (ui) {
				print_ui(epoch, cost_prev, y_hat, output_mini_batches[i]);
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
	return util.performance(_y_hat, _output_set);
}

void MLPPANN::save(std::string fileName) {
	MLPPUtilities util;
	if (!_network.empty()) {
		util.saveParameters(fileName, _network[0].weights, _network[0].bias, false, 1);
		for (uint32_t i = 1; i < _network.size(); i++) {
			util.saveParameters(fileName, _network[i].weights, _network[i].bias, true, i + 1);
		}
		util.saveParameters(fileName, _output_layer->weights, _output_layer->bias, true, _network.size() + 1);
	} else {
		util.saveParameters(fileName, _output_layer->weights, _output_layer->bias, false, _network.size() + 1);
	}
}

void MLPPANN::set_learning_rate_scheduler(std::string type, real_t decay_constant) {
	_lr_scheduler = type;
	_decay_constant = decay_constant;
}

void MLPPANN::set_learning_rate_scheduler_drop(std::string type, real_t decay_constant, real_t drop_rate) {
	_lr_scheduler = type;
	_decay_constant = decay_constant;
	_drop_rate = drop_rate;
}

// https://en.wikipedia.org/wiki/Learning_rate
// Learning Rate Decay (C2W2L09) - Andrew Ng - Deep Learning Specialization
real_t MLPPANN::apply_learning_rate_scheduler(real_t learning_rate, real_t decay_constant, real_t epoch, real_t drop_rate) {
	if (_lr_scheduler == "Time") {
		return learning_rate / (1 + decay_constant * epoch);
	} else if (_lr_scheduler == "Epoch") {
		return learning_rate * (decay_constant / std::sqrt(epoch));
	} else if (_lr_scheduler == "Step") {
		return learning_rate * std::pow(decay_constant, int((1 + epoch) / drop_rate)); // Utilizing an explicit int conversion implicitly takes the floor.
	} else if (_lr_scheduler == "Exponential") {
		return learning_rate * std::exp(-decay_constant * epoch);
	}
	return learning_rate;
}

void MLPPANN::add_layer(int n_hidden, std::string activation, std::string weightInit, std::string reg, real_t lambda, real_t alpha) {
	if (_network.empty()) {
		_network.push_back(MLPPOldHiddenLayer(n_hidden, activation, _input_set, weightInit, reg, lambda, alpha));
		_network[0].forwardPass();
	} else {
		_network.push_back(MLPPOldHiddenLayer(n_hidden, activation, _network[_network.size() - 1].a, weightInit, reg, lambda, alpha));
		_network[_network.size() - 1].forwardPass();
	}
}

void MLPPANN::add_output_layer(std::string activation, std::string loss, std::string weightInit, std::string reg, real_t lambda, real_t alpha) {
	if (!_network.empty()) {
		_output_layer = new MLPPOldOutputLayer(_network[_network.size() - 1].n_hidden, activation, loss, _network[_network.size() - 1].a, weightInit, reg, lambda, alpha);
	} else {
		_output_layer = new MLPPOldOutputLayer(_k, activation, loss, _input_set, weightInit, reg, lambda, alpha);
	}
}

MLPPANN::MLPPANN(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set) {
	_input_set = p_input_set;
	_output_set = p_output_set;

	_n = _input_set.size();
	_k = _input_set[0].size();
	_lr_scheduler = "None";
	_decay_constant = 0;
	_drop_rate = 0;
}

MLPPANN::MLPPANN() {
}

MLPPANN::~MLPPANN() {
	delete _output_layer;
}

real_t MLPPANN::cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;
	real_t totalRegTerm = 0;

	auto cost_function = _output_layer->cost_map[_output_layer->cost];

	if (!_network.empty()) {
		for (uint32_t i = 0; i < _network.size() - 1; i++) {
			totalRegTerm += regularization.regTerm(_network[i].weights, _network[i].lambda, _network[i].alpha, _network[i].reg);
		}
	}

	return (mlpp_cost.*cost_function)(y_hat, y) + totalRegTerm + regularization.regTerm(_output_layer->weights, _output_layer->lambda, _output_layer->alpha, _output_layer->reg);
}

void MLPPANN::forward_pass() {
	if (!_network.empty()) {
		_network[0].input = _input_set;
		_network[0].forwardPass();

		for (uint32_t i = 1; i < _network.size(); i++) {
			_network[i].input = _network[i - 1].a;
			_network[i].forwardPass();
		}
		_output_layer->input = _network[_network.size() - 1].a;
	} else {
		_output_layer->input = _input_set;
	}

	_output_layer->forwardPass();
	_y_hat = _output_layer->a;
}

void MLPPANN::update_parameters(std::vector<std::vector<std::vector<real_t>>> hidden_layer_updations, std::vector<real_t> output_layer_updation, real_t learning_rate) {
	MLPPLinAlg alg;

	_output_layer->weights = alg.subtraction(_output_layer->weights, output_layer_updation);
	_output_layer->bias -= learning_rate * alg.sum_elements(_output_layer->delta) / _n;

	if (!_network.empty()) {
		_network[_network.size() - 1].weights = alg.subtraction(_network[_network.size() - 1].weights, hidden_layer_updations[0]);
		_network[_network.size() - 1].bias = alg.subtractMatrixRows(_network[_network.size() - 1].bias, alg.scalarMultiply(learning_rate / _n, _network[_network.size() - 1].delta));

		for (int i = _network.size() - 2; i >= 0; i--) {
			_network[i].weights = alg.subtraction(_network[i].weights, hidden_layer_updations[(_network.size() - 2) - i + 1]);
			_network[i].bias = alg.subtractMatrixRows(_network[i].bias, alg.scalarMultiply(learning_rate / _n, _network[i].delta));
		}
	}
}

std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<real_t>> MLPPANN::compute_gradients(std::vector<real_t> y_hat, std::vector<real_t> _output_set) {
	// std::cout << "BEGIN" << std::endl;
	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	std::vector<std::vector<std::vector<real_t>>> cumulative_hidden_layer_w_grad; // Tensor containing ALL hidden grads.

	auto costDeriv = _output_layer->costDeriv_map[_output_layer->cost];
	auto outputAvn = _output_layer->activation_map[_output_layer->activation];
	_output_layer->delta = alg.hadamard_product((mlpp_cost.*costDeriv)(y_hat, _output_set), (avn.*outputAvn)(_output_layer->z, 1));
	std::vector<real_t> output_w_grad = alg.mat_vec_mult(alg.transpose(_output_layer->input), _output_layer->delta);
	output_w_grad = alg.addition(output_w_grad, regularization.regDerivTerm(_output_layer->weights, _output_layer->lambda, _output_layer->alpha, _output_layer->reg));

	if (!_network.empty()) {
		auto hiddenLayerAvn = _network[_network.size() - 1].activation_map[_network[_network.size() - 1].activation];
		_network[_network.size() - 1].delta = alg.hadamard_product(alg.outerProduct(_output_layer->delta, _output_layer->weights), (avn.*hiddenLayerAvn)(_network[_network.size() - 1].z, 1));
		std::vector<std::vector<real_t>> hiddenLayerWGrad = alg.matmult(alg.transpose(_network[_network.size() - 1].input), _network[_network.size() - 1].delta);

		cumulative_hidden_layer_w_grad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(_network[_network.size() - 1].weights, _network[_network.size() - 1].lambda, _network[_network.size() - 1].alpha, _network[_network.size() - 1].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

		for (int i = _network.size() - 2; i >= 0; i--) {
			hiddenLayerAvn = _network[i].activation_map[_network[i].activation];
			_network[i].delta = alg.hadamard_product(alg.matmult(_network[i + 1].delta, alg.transpose(_network[i + 1].weights)), (avn.*hiddenLayerAvn)(_network[i].z, 1));
			hiddenLayerWGrad = alg.matmult(alg.transpose(_network[i].input), _network[i].delta);
			cumulative_hidden_layer_w_grad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(_network[i].weights, _network[i].lambda, _network[i].alpha, _network[i].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}
	return { cumulative_hidden_layer_w_grad, output_w_grad };
}

void MLPPANN::print_ui(int epoch, real_t cost_prev, std::vector<real_t> y_hat, std::vector<real_t> p_output_set) {
	MLPPUtilities::CostInfo(epoch, cost_prev, cost(y_hat, p_output_set));
	std::cout << "Layer " << _network.size() + 1 << ": " << std::endl;
	MLPPUtilities::UI(_output_layer->weights, _output_layer->bias);
	if (!_network.empty()) {
		for (int i = _network.size() - 1; i >= 0; i--) {
			std::cout << "Layer " << i + 1 << ": " << std::endl;
			MLPPUtilities::UI(_network[i].weights, _network[i].bias);
		}
	}
}

void MLPPANN::_bind_methods() {
}
