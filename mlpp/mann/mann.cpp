//
//  MANN.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "mann.h"
#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <iostream>

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

std::vector<std::vector<real_t>> MLPPMANN::model_set_test(std::vector<std::vector<real_t>> X) {
	ERR_FAIL_COND_V(!_initialized, std::vector<std::vector<real_t>>());

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

std::vector<real_t> MLPPMANN::model_test(std::vector<real_t> x) {
	ERR_FAIL_COND_V(!_initialized, std::vector<real_t>());

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

void MLPPMANN::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		if (_output_layer->activation == "Softmax") {
			_output_layer->delta = alg.subtraction(_y_hat, _output_set);
		} else {
			auto costDeriv = _output_layer->costDeriv_map[_output_layer->cost];
			auto outputAvn = _output_layer->activation_map[_output_layer->activation];
			_output_layer->delta = alg.hadamard_product((mlpp_cost.*costDeriv)(_y_hat, _output_set), (avn.*outputAvn)(_output_layer->z, 1));
		}

		std::vector<std::vector<real_t>> outputWGrad = alg.matmult(alg.transpose(_output_layer->input), _output_layer->delta);

		_output_layer->weights = alg.subtraction(_output_layer->weights, alg.scalarMultiply(learning_rate / _n, outputWGrad));
		_output_layer->weights = regularization.regWeights(_output_layer->weights, _output_layer->lambda, _output_layer->alpha, _output_layer->reg);
		_output_layer->bias = alg.subtractMatrixRows(_output_layer->bias, alg.scalarMultiply(learning_rate / _n, _output_layer->delta));

		if (!_network.empty()) {
			auto hiddenLayerAvn = _network[_network.size() - 1].activation_map[_network[_network.size() - 1].activation];
			_network[_network.size() - 1].delta = alg.hadamard_product(alg.matmult(_output_layer->delta, alg.transpose(_output_layer->weights)), (avn.*hiddenLayerAvn)(_network[_network.size() - 1].z, true));
			std::vector<std::vector<real_t>> hiddenLayerWGrad = alg.matmult(alg.transpose(_network[_network.size() - 1].input), _network[_network.size() - 1].delta);

			_network[_network.size() - 1].weights = alg.subtraction(_network[_network.size() - 1].weights, alg.scalarMultiply(learning_rate / _n, hiddenLayerWGrad));
			_network[_network.size() - 1].weights = regularization.regWeights(_network[_network.size() - 1].weights, _network[_network.size() - 1].lambda, _network[_network.size() - 1].alpha, _network[_network.size() - 1].reg);
			_network[_network.size() - 1].bias = alg.subtractMatrixRows(_network[_network.size() - 1].bias, alg.scalarMultiply(learning_rate / _n, _network[_network.size() - 1].delta));

			for (int i = _network.size() - 2; i >= 0; i--) {
				hiddenLayerAvn = _network[i].activation_map[_network[i].activation];
				_network[i].delta = alg.hadamard_product(alg.matmult(_network[i + 1].delta, _network[i + 1].weights), (avn.*hiddenLayerAvn)(_network[i].z, true));
				hiddenLayerWGrad = alg.matmult(alg.transpose(_network[i].input), _network[i].delta);
				_network[i].weights = alg.subtraction(_network[i].weights, alg.scalarMultiply(learning_rate / _n, hiddenLayerWGrad));
				_network[i].weights = regularization.regWeights(_network[i].weights, _network[i].lambda, _network[i].alpha, _network[i].reg);
				_network[i].bias = alg.subtractMatrixRows(_network[i].bias, alg.scalarMultiply(learning_rate / _n, _network[i].delta));
			}
		}

		forward_pass();

		if (ui) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost(_y_hat, _output_set));
			std::cout << "Layer " << _network.size() + 1 << ": " << std::endl;
			MLPPUtilities::UI(_output_layer->weights, _output_layer->bias);
			if (!_network.empty()) {
				std::cout << "Layer " << _network.size() << ": " << std::endl;
				for (int i = _network.size() - 1; i >= 0; i--) {
					std::cout << "Layer " << i + 1 << ": " << std::endl;
					MLPPUtilities::UI(_network[i].weights, _network[i].bias);
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

	return util.performance(_y_hat, _output_set);
}

void MLPPMANN::save(std::string fileName) {
	ERR_FAIL_COND(!_initialized);

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

void MLPPMANN::add_layer(int n_hidden, std::string activation, std::string weightInit, std::string reg, real_t lambda, real_t alpha) {
	if (_network.empty()) {
		_network.push_back(MLPPOldHiddenLayer(n_hidden, activation, _input_set, weightInit, reg, lambda, alpha));
		_network[0].forwardPass();
	} else {
		_network.push_back(MLPPOldHiddenLayer(n_hidden, activation, _network[_network.size() - 1].a, weightInit, reg, lambda, alpha));
		_network[_network.size() - 1].forwardPass();
	}
}

void MLPPMANN::add_output_layer(std::string activation, std::string loss, std::string weightInit, std::string reg, real_t lambda, real_t alpha) {
	if (!_network.empty()) {
		_output_layer = new MLPPOldMultiOutputLayer(_n_output, _network[0].n_hidden, activation, loss, _network[_network.size() - 1].a, weightInit, reg, lambda, alpha);
	} else {
		_output_layer = new MLPPOldMultiOutputLayer(_n_output, _k, activation, loss, _input_set, weightInit, reg, lambda, alpha);
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

MLPPMANN::MLPPMANN(std::vector<std::vector<real_t>> p_input_set, std::vector<std::vector<real_t>> p_output_set) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = _input_set.size();
	_k = _input_set[0].size();
	_n_output = _output_set[0].size();

	_initialized = true;
}

MLPPMANN::MLPPMANN() {
	_initialized = false;
}

MLPPMANN::~MLPPMANN() {
	delete _output_layer;
}

real_t MLPPMANN::cost(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPReg regularization;
	class MLPPCost cost;
	real_t totalRegTerm = 0;

	auto cost_function = _output_layer->cost_map[_output_layer->cost];
	if (!_network.empty()) {
		for (uint32_t i = 0; i < _network.size() - 1; i++) {
			totalRegTerm += regularization.regTerm(_network[i].weights, _network[i].lambda, _network[i].alpha, _network[i].reg);
		}
	}
	return (cost.*cost_function)(y_hat, y) + totalRegTerm + regularization.regTerm(_output_layer->weights, _output_layer->lambda, _output_layer->alpha, _output_layer->reg);
}

void MLPPMANN::forward_pass() {
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
