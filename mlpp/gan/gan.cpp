//
//  GAN.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "gan.h"
#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <cmath>
#include <iostream>

/*
Ref<MLPPMatrix> MLPPGAN::get_input_set() {
	return _input_set;
}
void MLPPGAN::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;
}

Ref<MLPPVector> MLPPGAN::get_output_set() {
	return _output_set;
}
void MLPPGAN::set_output_set(const Ref<MLPPVector> &val) {
	_output_set = val;
}

int MLPPGAN::get_k() {
	return _k;
}
void MLPPGAN::set_k(const int val) {
	_k = val;
}
*/

std::vector<std::vector<real_t>> MLPPGAN::generate_example(int n) {
	MLPPLinAlg alg;

	return model_set_test_generator(alg.gaussianNoise(n, _k));
}

void MLPPGAN::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	MLPPCost mlpp_cost;
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, alg.onevec(_n));

		// Training of the discriminator.

		std::vector<std::vector<real_t>> generator_input_set = alg.gaussianNoise(_n, _k);
		std::vector<std::vector<real_t>> discriminator_input_set = model_set_test_generator(generator_input_set);
		discriminator_input_set.insert(discriminator_input_set.end(), _output_set.begin(), _output_set.end()); // Fake + real inputs.

		std::vector<real_t> y_hat = model_set_test_discriminator(discriminator_input_set);
		std::vector<real_t> _output_set = alg.zerovec(_n);
		std::vector<real_t> _output_setReal = alg.onevec(_n);
		_output_set.insert(_output_set.end(), _output_setReal.begin(), _output_setReal.end()); // Fake + real output scores.

		auto dgrads = compute_discriminator_gradients(y_hat, _output_set);
		auto cumulative_discriminator_hidden_layer_w_grad = std::get<0>(dgrads);
		auto outputDiscriminatorWGrad = std::get<1>(dgrads);

		cumulative_discriminator_hidden_layer_w_grad = alg.scalarMultiply(learning_rate / _n, cumulative_discriminator_hidden_layer_w_grad);
		outputDiscriminatorWGrad = alg.scalarMultiply(learning_rate / _n, outputDiscriminatorWGrad);
		update_discriminator_parameters(cumulative_discriminator_hidden_layer_w_grad, outputDiscriminatorWGrad, learning_rate);

		// Training of the generator.
		generator_input_set = alg.gaussianNoise(_n, _k);
		discriminator_input_set = model_set_test_generator(generator_input_set);
		y_hat = model_set_test_discriminator(discriminator_input_set);
		_output_set = alg.onevec(_n);

		std::vector<std::vector<std::vector<real_t>>> cumulative_generator_hidden_layer_w_grad = compute_generator_gradients(y_hat, _output_set);
		cumulative_generator_hidden_layer_w_grad = alg.scalarMultiply(learning_rate / _n, cumulative_generator_hidden_layer_w_grad);
		update_generator_parameters(cumulative_generator_hidden_layer_w_grad, learning_rate);

		forward_pass();

		if (ui) {
			print_ui(epoch, cost_prev, _y_hat, alg.onevec(_n));
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

real_t MLPPGAN::score() {
	MLPPLinAlg alg;
	MLPPUtilities util;

	forward_pass();

	return util.performance(_y_hat, alg.onevec(_n));
}

void MLPPGAN::save(std::string fileName) {
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

void MLPPGAN::add_layer(int n_hidden, std::string activation, std::string weight_init, std::string reg, real_t lambda, real_t alpha) {
	MLPPLinAlg alg;
	if (_network.empty()) {
		_network.push_back(MLPPOldHiddenLayer(n_hidden, activation, alg.gaussianNoise(_n, _k), weight_init, reg, lambda, alpha));
		_network[0].forwardPass();
	} else {
		_network.push_back(MLPPOldHiddenLayer(n_hidden, activation, _network[_network.size() - 1].a, weight_init, reg, lambda, alpha));
		_network[_network.size() - 1].forwardPass();
	}
}

void MLPPGAN::add_output_layer(std::string weight_init, std::string reg, real_t lambda, real_t alpha) {
	MLPPLinAlg alg;
	if (!_network.empty()) {
		_output_layer = new MLPPOldOutputLayer(_network[_network.size() - 1].n_hidden, "Sigmoid", "LogLoss", _network[_network.size() - 1].a, weight_init, reg, lambda, alpha);
	} else {
		_output_layer = new MLPPOldOutputLayer(_k, "Sigmoid", "LogLoss", alg.gaussianNoise(_n, _k), weight_init, reg, lambda, alpha);
	}
}

MLPPGAN::MLPPGAN(real_t k, std::vector<std::vector<real_t>> output_set) {
	_output_set = output_set;
	_n = _output_set.size();
	_k = k;
}

MLPPGAN::MLPPGAN() {
}

MLPPGAN::~MLPPGAN() {
	delete _output_layer;
}

std::vector<std::vector<real_t>> MLPPGAN::model_set_test_generator(std::vector<std::vector<real_t>> X) {
	if (!_network.empty()) {
		_network[0].input = X;
		_network[0].forwardPass();

		for (uint32_t i = 1; i <= _network.size() / 2; i++) {
			_network[i].input = _network[i - 1].a;
			_network[i].forwardPass();
		}
	}
	return _network[_network.size() / 2].a;
}

std::vector<real_t> MLPPGAN::model_set_test_discriminator(std::vector<std::vector<real_t>> X) {
	if (!_network.empty()) {
		for (uint32_t i = _network.size() / 2 + 1; i < _network.size(); i++) {
			if (i == _network.size() / 2 + 1) {
				_network[i].input = X;
			} else {
				_network[i].input = _network[i - 1].a;
			}

			_network[i].forwardPass();
		}

		_output_layer->input = _network[_network.size() - 1].a;
	}

	_output_layer->forwardPass();

	return _output_layer->a;
}

real_t MLPPGAN::cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
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

void MLPPGAN::forward_pass() {
	MLPPLinAlg alg;

	if (!_network.empty()) {
		_network[0].input = alg.gaussianNoise(_n, _k);
		_network[0].forwardPass();

		for (uint32_t i = 1; i < _network.size(); i++) {
			_network[i].input = _network[i - 1].a;
			_network[i].forwardPass();
		}
		_output_layer->input = _network[_network.size() - 1].a;
	} else { // Should never happen, though.
		_output_layer->input = alg.gaussianNoise(_n, _k);
	}

	_output_layer->forwardPass();
	_y_hat = _output_layer->a;
}

void MLPPGAN::update_discriminator_parameters(std::vector<std::vector<std::vector<real_t>>> hidden_layer_updations, std::vector<real_t> output_layer_updation, real_t learning_rate) {
	MLPPLinAlg alg;

	_output_layer->weights = alg.subtraction(_output_layer->weights, output_layer_updation);
	_output_layer->bias -= learning_rate * alg.sum_elements(_output_layer->delta) / _n;

	if (!_network.empty()) {
		_network[_network.size() - 1].weights = alg.subtraction(_network[_network.size() - 1].weights, hidden_layer_updations[0]);
		_network[_network.size() - 1].bias = alg.subtractMatrixRows(_network[_network.size() - 1].bias, alg.scalarMultiply(learning_rate / _n, _network[_network.size() - 1].delta));

		for (int i = static_cast<int>(_network.size()) - 2; i > static_cast<int>(_network.size()) / 2; i--) {
			_network[i].weights = alg.subtraction(_network[i].weights, hidden_layer_updations[(_network.size() - 2) - i + 1]);
			_network[i].bias = alg.subtractMatrixRows(_network[i].bias, alg.scalarMultiply(learning_rate / _n, _network[i].delta));
		}
	}
}

void MLPPGAN::update_generator_parameters(std::vector<std::vector<std::vector<real_t>>> hidden_layer_updations, real_t learning_rate) {
	MLPPLinAlg alg;

	if (!_network.empty()) {
		for (int i = _network.size() / 2; i >= 0; i--) {
			//std::cout << network[i].weights.size() << "x" << network[i].weights[0].size() << std::endl;
			//std::cout << hidden_layer_updations[(network.size() - 2) - i + 1].size() << "x" << hidden_layer_updations[(network.size() - 2) - i + 1][0].size() << std::endl;
			_network[i].weights = alg.subtraction(_network[i].weights, hidden_layer_updations[(_network.size() - 2) - i + 1]);
			_network[i].bias = alg.subtractMatrixRows(_network[i].bias, alg.scalarMultiply(learning_rate / _n, _network[i].delta));
		}
	}
}

std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<real_t>> MLPPGAN::compute_discriminator_gradients(std::vector<real_t> y_hat, std::vector<real_t> _output_set) {
	class MLPPCost cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	std::vector<std::vector<std::vector<real_t>>> cumulativeHiddenLayerWGrad; // Tensor containing ALL hidden grads.

	auto costDeriv = _output_layer->costDeriv_map[_output_layer->cost];
	auto outputAvn = _output_layer->activation_map[_output_layer->activation];
	_output_layer->delta = alg.hadamard_product((cost.*costDeriv)(y_hat, _output_set), (avn.*outputAvn)(_output_layer->z, 1));
	std::vector<real_t> outputWGrad = alg.mat_vec_mult(alg.transpose(_output_layer->input), _output_layer->delta);
	outputWGrad = alg.addition(outputWGrad, regularization.regDerivTerm(_output_layer->weights, _output_layer->lambda, _output_layer->alpha, _output_layer->reg));

	if (!_network.empty()) {
		auto hiddenLayerAvn = _network[_network.size() - 1].activation_map[_network[_network.size() - 1].activation];

		_network[_network.size() - 1].delta = alg.hadamard_product(alg.outerProduct(_output_layer->delta, _output_layer->weights), (avn.*hiddenLayerAvn)(_network[_network.size() - 1].z, 1));
		std::vector<std::vector<real_t>> hiddenLayerWGrad = alg.matmult(alg.transpose(_network[_network.size() - 1].input), _network[_network.size() - 1].delta);

		cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(_network[_network.size() - 1].weights, _network[_network.size() - 1].lambda, _network[_network.size() - 1].alpha, _network[_network.size() - 1].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

		//std::cout << "HIDDENLAYER FIRST:" << hiddenLayerWGrad.size() << "x" << hiddenLayerWGrad[0].size() << std::endl;
		//std::cout << "WEIGHTS SECOND:" << network[network.size() - 1].weights.size() << "x" << network[network.size() - 1].weights[0].size() << std::endl;

		for (int i = static_cast<int>(_network.size()) - 2; i > static_cast<int>(_network.size()) / 2; i--) {
			hiddenLayerAvn = _network[i].activation_map[_network[i].activation];
			_network[i].delta = alg.hadamard_product(alg.matmult(_network[i + 1].delta, alg.transpose(_network[i + 1].weights)), (avn.*hiddenLayerAvn)(_network[i].z, 1));
			hiddenLayerWGrad = alg.matmult(alg.transpose(_network[i].input), _network[i].delta);

			cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(_network[i].weights, _network[i].lambda, _network[i].alpha, _network[i].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}
	return { cumulativeHiddenLayerWGrad, outputWGrad };
}

std::vector<std::vector<std::vector<real_t>>> MLPPGAN::compute_generator_gradients(std::vector<real_t> y_hat, std::vector<real_t> _output_set) {
	class MLPPCost cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	std::vector<std::vector<std::vector<real_t>>> cumulativeHiddenLayerWGrad; // Tensor containing ALL hidden grads.

	auto costDeriv = _output_layer->costDeriv_map[_output_layer->cost];
	auto outputAvn = _output_layer->activation_map[_output_layer->activation];
	_output_layer->delta = alg.hadamard_product((cost.*costDeriv)(y_hat, _output_set), (avn.*outputAvn)(_output_layer->z, true));
	std::vector<real_t> outputWGrad = alg.mat_vec_mult(alg.transpose(_output_layer->input), _output_layer->delta);
	outputWGrad = alg.addition(outputWGrad, regularization.regDerivTerm(_output_layer->weights, _output_layer->lambda, _output_layer->alpha, _output_layer->reg));

	if (!_network.empty()) {
		auto hiddenLayerAvn = _network[_network.size() - 1].activation_map[_network[_network.size() - 1].activation];
		_network[_network.size() - 1].delta = alg.hadamard_product(alg.outerProduct(_output_layer->delta, _output_layer->weights), (avn.*hiddenLayerAvn)(_network[_network.size() - 1].z, 1));
		std::vector<std::vector<real_t>> hiddenLayerWGrad = alg.matmult(alg.transpose(_network[_network.size() - 1].input), _network[_network.size() - 1].delta);
		cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(_network[_network.size() - 1].weights, _network[_network.size() - 1].lambda, _network[_network.size() - 1].alpha, _network[_network.size() - 1].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

		for (int i = _network.size() - 2; i >= 0; i--) {
			hiddenLayerAvn = _network[i].activation_map[_network[i].activation];
			_network[i].delta = alg.hadamard_product(alg.matmult(_network[i + 1].delta, alg.transpose(_network[i + 1].weights)), (avn.*hiddenLayerAvn)(_network[i].z, true));
			hiddenLayerWGrad = alg.matmult(alg.transpose(_network[i].input), _network[i].delta);
			cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(_network[i].weights, _network[i].lambda, _network[i].alpha, _network[i].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}

	return cumulativeHiddenLayerWGrad;
}

void MLPPGAN::print_ui(int epoch, real_t cost_prev, std::vector<real_t> y_hat, std::vector<real_t> _output_set) {
	MLPPUtilities::CostInfo(epoch, cost_prev, cost(y_hat, _output_set));
	std::cout << "Layer " << _network.size() + 1 << ": " << std::endl;
	MLPPUtilities::UI(_output_layer->weights, _output_layer->bias);
	if (!_network.empty()) {
		for (int i = _network.size() - 1; i >= 0; i--) {
			std::cout << "Layer " << i + 1 << ": " << std::endl;
			MLPPUtilities::UI(_network[i].weights, _network[i].bias);
		}
	}
}

void MLPPGAN::_bind_methods() {
	/*
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPGAN::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "value"), &MLPPGAN::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPGAN::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "value"), &MLPPGAN::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_k"), &MLPPGAN::get_k);
	ClassDB::bind_method(D_METHOD("set_k", "value"), &MLPPGAN::set_k);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "k"), "set_k", "get_k");

	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPGAN::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPGAN::model_test);
	ClassDB::bind_method(D_METHOD("score"), &MLPPGAN::score);
	*/
}
