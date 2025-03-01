/*************************************************************************/
/*  mann.cpp                                                             */
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

#include "mann.h"

#ifdef USING_SFW
#include "sfw.h"
#else
#include "core/log/logger.h"
#endif

#include "../core/activation.h"
#include "../core/cost.h"
#include "../core/reg.h"
#include "../core/utilities.h"

/*
Ref<MLPPMatrix> MLPPMANN::get_input_set() {
	return input_set;
}
void MLPPMANN::set_input_set(const Ref<MLPPMatrix> &val) {
	input_set = val;

	_initialized = false;
}

Ref<MLPPMatrix> MLPPMANN::get_output_set() {
	return output_set;
}
void MLPPMANN::set_output_set(const Ref<MLPPMatrix> &val) {
	output_set = val;

	_initialized = false;
}
*/

Ref<MLPPMatrix> MLPPMANN::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPMatrix>());

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

Ref<MLPPVector> MLPPMANN::model_test(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPVector>());

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

void MLPPMANN::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		if (_output_layer->get_activation() == MLPPActivation::ACTIVATION_FUNCTION_SOFTMAX) {
			_output_layer->set_delta(_y_hat->subn(_output_set));
		} else {
			Ref<MLPPMatrix> r1 = mlpp_cost.run_cost_deriv_matrix(_output_layer->get_cost(), _y_hat, _output_set);
			Ref<MLPPMatrix> r2 = avn.run_activation_deriv_matrix(_output_layer->get_activation(), _output_layer->get_z());

			_output_layer->set_delta(r1->hadamard_productn(r2));
		}

		Ref<MLPPMatrix> output_w_grad = _output_layer->get_input()->transposen()->multn(_output_layer->get_delta());

		_output_layer->set_weights(_output_layer->get_weights()->subn(output_w_grad->scalar_multiplyn(learning_rate / _n)));
		_output_layer->set_weights(regularization.reg_weightsm(_output_layer->get_weights(), _output_layer->get_lambda(), _output_layer->get_alpha(),
				_output_layer->get_reg()));

		_output_layer->set_bias(_output_layer->get_bias()->subtract_matrix_rowsn(_output_layer->get_delta()->scalar_multiplyn(learning_rate / _n)));

		if (!_network.empty()) {
			Ref<MLPPHiddenLayer> layer = _network[_network.size() - 1];

			layer->set_delta(_output_layer->get_delta()->multn(_output_layer->get_weights()->transposen())->hadamard_productn(avn.run_activation_deriv_matrix(layer->get_activation(), layer->get_z())));

			Ref<MLPPMatrix> hidden_layer_w_grad = layer->get_input()->transposen()->multn(layer->get_delta());

			layer->set_weights(layer->get_weights()->subn(hidden_layer_w_grad->scalar_multiplyn(learning_rate / _n)));
			layer->set_weights(regularization.reg_weightsm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()));

			layer->set_bias(layer->get_bias()->subtract_matrix_rowsn(layer->get_delta()->scalar_multiplyn(learning_rate / _n)));

			for (int i = _network.size() - 2; i >= 0; i--) {
				layer = _network[i];
				Ref<MLPPHiddenLayer> next_layer = _network[i + 1];

				layer->set_delta(next_layer->get_delta()->multn(next_layer->get_weights())->hadamard_productn(avn.run_activation_deriv_matrix(layer->get_activation(), layer->get_z())));

				hidden_layer_w_grad = layer->get_input()->transposen()->multn(layer->get_delta());

				layer->set_weights(layer->get_weights()->subn(hidden_layer_w_grad->scalar_multiplyn(learning_rate / _n)));
				layer->set_weights(regularization.reg_weightsm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()));
				layer->set_bias(layer->get_bias()->subtract_matrix_rowsn(layer->get_delta()->scalar_multiplyn(learning_rate / _n)));
			}
		}

		forward_pass();

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_y_hat, _output_set));
			PLOG_MSG("Layer " + itos(_network.size() + 1) + ": ");
			MLPPUtilities::print_ui_mb(_output_layer->get_weights(), _output_layer->get_bias());

			if (!_network.empty()) {
				for (int i = _network.size() - 1; i >= 0; i--) {
					PLOG_MSG("Layer " + itos(i + 1) + ": ");

					Ref<MLPPHiddenLayer> layer = _network[i];

					MLPPUtilities::print_ui_mb(layer->get_weights(), layer->get_bias());
				}
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

real_t MLPPMANN::score() {
	ERR_FAIL_COND_V(!_initialized, 0);

	MLPPUtilities util;

	forward_pass();

	return util.performance_mat(_y_hat, _output_set);
}

void MLPPMANN::save(const String &file_name) {
	ERR_FAIL_COND(!_initialized);

	/*
	MLPPUtilities util;
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

void MLPPMANN::add_layer(int n_hidden, MLPPActivation::ActivationFunction activation, MLPPUtilities::WeightDistributionType weight_init, MLPPReg::RegularizationType reg, real_t lambda, real_t alpha) {
	if (_network.empty()) {
		_network.push_back(Ref<MLPPHiddenLayer>(memnew(MLPPHiddenLayer(n_hidden, activation, _input_set, weight_init, reg, lambda, alpha))));
		_network.write[0]->forward_pass();
	} else {
		_network.push_back(Ref<MLPPHiddenLayer>(memnew(MLPPHiddenLayer(n_hidden, activation, _network.write[_network.size() - 1]->get_a(), weight_init, reg, lambda, alpha))));
		_network.write[_network.size() - 1]->forward_pass();
	}
}

void MLPPMANN::add_output_layer(MLPPActivation::ActivationFunction activation, MLPPCost::CostTypes loss, MLPPUtilities::WeightDistributionType weight_init, MLPPReg::RegularizationType reg, real_t lambda, real_t alpha) {
	if (!_network.empty()) {
		_output_layer = Ref<MLPPMultiOutputLayer>(memnew(MLPPMultiOutputLayer(_n_output, _network.write[_network.size() - 1]->get_n_hidden(), activation, loss, _network.write[_network.size() - 1]->get_a(), weight_init, reg, lambda, alpha)));
	} else {
		_output_layer = Ref<MLPPMultiOutputLayer>(memnew(MLPPMultiOutputLayer(_n_output, _k, activation, loss, _input_set, weight_init, reg, lambda, alpha)));
	}
}

bool MLPPMANN::is_initialized() {
	return _initialized;
}

void MLPPMANN::initialize() {
	if (_initialized) {
		return;
	}

	//ERR_FAIL_COND(!input_set.is_valid() || !output_set.is_valid() || n_hidden == 0);

	_initialized = true;
}

MLPPMANN::MLPPMANN(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPMatrix> &p_output_set) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = _input_set->size().y;
	_k = _input_set->size().x;
	_n_output = _output_set->size().x;

	_initialized = true;
}

MLPPMANN::MLPPMANN() {
	_initialized = false;
}

MLPPMANN::~MLPPMANN() {
}

real_t MLPPMANN::cost(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;

	real_t total_reg_term = 0;

	if (!_network.empty()) {
		for (int i = 0; i < _network.size() - 1; i++) {
			Ref<MLPPHiddenLayer> layer = _network[i];

			total_reg_term += regularization.reg_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg());
		}
	}

	return mlpp_cost.run_cost_norm_matrix(_output_layer->get_cost(), y_hat, y) + total_reg_term + regularization.reg_termm(_output_layer->get_weights(), _output_layer->get_lambda(), _output_layer->get_alpha(), _output_layer->get_reg());
}

void MLPPMANN::forward_pass() {
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

void MLPPMANN::_bind_methods() {
	/*
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPMANN::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPMANN::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPMANN::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPMANN::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_output_set", "get_output_set");
	*/
}
