//
//  AutoEncoder.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "auto_encoder.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

#include <random>

//UDPATE
Ref<MLPPMatrix> MLPPAutoEncoder::get_input_set() {
	return Ref<MLPPMatrix>();
	//return _input_set;
}
void MLPPAutoEncoder::set_input_set(const Ref<MLPPMatrix> &val) {
	//_input_set = val;

	_initialized = false;
}

int MLPPAutoEncoder::get_n_hidden() {
	return _n_hidden;
}
void MLPPAutoEncoder::set_n_hidden(const int val) {
	_n_hidden = val;

	_initialized = false;
}

std::vector<std::vector<real_t>> MLPPAutoEncoder::model_set_test(std::vector<std::vector<real_t>> X) {
	ERR_FAIL_COND_V(!_initialized, std::vector<std::vector<real_t>>());

	return evaluatem(X);
}

std::vector<real_t> MLPPAutoEncoder::model_test(std::vector<real_t> x) {
	ERR_FAIL_COND_V(!_initialized, std::vector<real_t>());

	return evaluatev(x);
}

void MLPPAutoEncoder::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _input_set);

		// Calculating the errors
		std::vector<std::vector<real_t>> error = alg.subtraction(_y_hat, _input_set);

		// Calculating the weight/bias gradients for layer 2
		std::vector<std::vector<real_t>> D2_1 = alg.matmult(alg.transpose(_a2), error);

		// weights and bias updation for layer 2
		_weights2 = alg.subtraction(_weights2, alg.scalarMultiply(learning_rate / _n, D2_1));

		// Calculating the bias gradients for layer 2
		_bias2 = alg.subtractMatrixRows(_bias2, alg.scalarMultiply(learning_rate, error));

		//Calculating the weight/bias for layer 1

		std::vector<std::vector<real_t>> D1_1 = alg.matmult(error, alg.transpose(_weights2));

		std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(_z2, 1));

		std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(_input_set), D1_2);

		// weight an bias updation for layer 1
		_weights1 = alg.subtraction(_weights1, alg.scalarMultiply(learning_rate / _n, D1_3));

		_bias1 = alg.subtractMatrixRows(_bias1, alg.scalarMultiply(learning_rate / _n, D1_2));

		forward_pass();

		// UI PORTION
		if (ui) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost(_y_hat, _input_set));
			std::cout << "Layer 1:" << std::endl;
			MLPPUtilities::UI(_weights1, _bias1);
			std::cout << "Layer 2:" << std::endl;
			MLPPUtilities::UI(_weights2, _bias2);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPAutoEncoder::sgd(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;

	while (true) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(_n - 1));
		int outputIndex = distribution(generator);

		std::vector<real_t> y_hat = evaluatev(_input_set[outputIndex]);
		auto prop_res = propagatev(_input_set[outputIndex]);
		auto z2 = std::get<0>(prop_res);
		auto a2 = std::get<1>(prop_res);

		cost_prev = cost({ y_hat }, { _input_set[outputIndex] });
		std::vector<real_t> error = alg.subtraction(y_hat, _input_set[outputIndex]);

		// Weight updation for layer 2
		std::vector<std::vector<real_t>> D2_1 = alg.outerProduct(error, a2);
		_weights2 = alg.subtraction(_weights2, alg.scalarMultiply(learning_rate, alg.transpose(D2_1)));

		// Bias updation for layer 2
		_bias2 = alg.subtraction(_bias2, alg.scalarMultiply(learning_rate, error));

		// Weight updation for layer 1
		std::vector<real_t> D1_1 = alg.mat_vec_mult(_weights2, error);
		std::vector<real_t> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));
		std::vector<std::vector<real_t>> D1_3 = alg.outerProduct(_input_set[outputIndex], D1_2);

		_weights1 = alg.subtraction(_weights1, alg.scalarMultiply(learning_rate, D1_3));
		// Bias updation for layer 1

		_bias1 = alg.subtraction(_bias1, alg.scalarMultiply(learning_rate, D1_2));

		y_hat = evaluatev(_input_set[outputIndex]);

		if (ui) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost({ y_hat }, { _input_set[outputIndex] }));
			std::cout << "Layer 1:" << std::endl;
			MLPPUtilities::UI(_weights1, _bias1);
			std::cout << "Layer 2:" << std::endl;
			MLPPUtilities::UI(_weights2, _bias2);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPAutoEncoder::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	std::vector<std::vector<std::vector<real_t>>> inputMiniBatches = MLPPUtilities::createMiniBatches(_input_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<std::vector<real_t>> y_hat = evaluatem(inputMiniBatches[i]);

			auto prop_res = propagatem(inputMiniBatches[i]);
			auto z2 = std::get<0>(prop_res);
			auto a2 = std::get<1>(prop_res);

			cost_prev = cost(y_hat, inputMiniBatches[i]);

			// Calculating the errors
			std::vector<std::vector<real_t>> error = alg.subtraction(y_hat, inputMiniBatches[i]);

			// Calculating the weight/bias gradients for layer 2

			std::vector<std::vector<real_t>> D2_1 = alg.matmult(alg.transpose(a2), error);

			// weights and bias updation for layer 2
			_weights2 = alg.subtraction(_weights2, alg.scalarMultiply(learning_rate / inputMiniBatches[i].size(), D2_1));

			// Bias Updation for layer 2
			_bias2 = alg.subtractMatrixRows(_bias2, alg.scalarMultiply(learning_rate, error));

			//Calculating the weight/bias for layer 1

			std::vector<std::vector<real_t>> D1_1 = alg.matmult(error, alg.transpose(_weights2));

			std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, true));

			std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(inputMiniBatches[i]), D1_2);

			// weight an bias updation for layer 1
			_weights1 = alg.subtraction(_weights1, alg.scalarMultiply(learning_rate / inputMiniBatches[i].size(), D1_3));

			_bias1 = alg.subtractMatrixRows(_bias1, alg.scalarMultiply(learning_rate / inputMiniBatches[i].size(), D1_2));

			y_hat = evaluatem(inputMiniBatches[i]);

			if (ui) {
				MLPPUtilities::CostInfo(epoch, cost_prev, cost(y_hat, inputMiniBatches[i]));
				std::cout << "Layer 1:" << std::endl;
				MLPPUtilities::UI(_weights1, _bias1);
				std::cout << "Layer 2:" << std::endl;
				MLPPUtilities::UI(_weights2, _bias2);
			}
		}

		epoch++;
		
		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

real_t MLPPAutoEncoder::score() {
	ERR_FAIL_COND_V(!_initialized, 0);

	MLPPUtilities util;
	return util.performance(_y_hat, _input_set);
}

void MLPPAutoEncoder::save(std::string fileName) {
	ERR_FAIL_COND(!_initialized);

	MLPPUtilities util;
	util.saveParameters(fileName, _weights1, _bias1, false, 1);
	util.saveParameters(fileName, _weights2, _bias2, true, 2);
}

MLPPAutoEncoder::MLPPAutoEncoder(std::vector<std::vector<real_t>> p_input_set, int pn_hidden) {
	_input_set = p_input_set;
	_n_hidden = pn_hidden;
	_n = _input_set.size();
	_k = _input_set[0].size();

	MLPPActivation avn;
	_y_hat.resize(_input_set.size());

	_weights1 = MLPPUtilities::weightInitialization(_k, _n_hidden);
	_weights2 = MLPPUtilities::weightInitialization(_n_hidden, _k);
	_bias1 = MLPPUtilities::biasInitialization(_n_hidden);
	_bias2 = MLPPUtilities::biasInitialization(_k);

	_initialized = true;
}

MLPPAutoEncoder::MLPPAutoEncoder() {
	_initialized = false;
}
MLPPAutoEncoder::~MLPPAutoEncoder() {
}

real_t MLPPAutoEncoder::cost(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	class MLPPCost cost;

	return cost.MSE(y_hat, _input_set);
}

std::vector<real_t> MLPPAutoEncoder::evaluatev(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(_weights1), x), _bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);

	return alg.addition(alg.mat_vec_mult(alg.transpose(_weights2), a2), _bias2);
}

std::tuple<std::vector<real_t>, std::vector<real_t>> MLPPAutoEncoder::propagatev(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(_weights1), x), _bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);

	return { z2, a2 };
}

std::vector<std::vector<real_t>> MLPPAutoEncoder::evaluatem(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, _weights1), _bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);

	return alg.mat_vec_add(alg.matmult(a2, _weights2), _bias2);
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPAutoEncoder::propagatem(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, _weights1), _bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);

	return { z2, a2 };
}

void MLPPAutoEncoder::forward_pass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	_z2 = alg.mat_vec_add(alg.matmult(_input_set, _weights1), _bias1);
	_a2 = avn.sigmoid(_z2);
	_y_hat = alg.mat_vec_add(alg.matmult(_a2, _weights2), _bias2);
}

void MLPPAutoEncoder::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPAutoEncoder::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPAutoEncoder::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_n_hidden"), &MLPPAutoEncoder::get_n_hidden);
	ClassDB::bind_method(D_METHOD("set_n_hidden", "val"), &MLPPAutoEncoder::set_n_hidden);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "n_hidden"), "set_n_hidden", "get_n_hidden");

	/*
	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPAutoEncoder::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPAutoEncoder::model_test);

	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPAutoEncoder::gradient_descent, false);
	ClassDB::bind_method(D_METHOD("sgd", "learning_rate", "max_epoch", "ui"), &MLPPAutoEncoder::sgd, false);
	ClassDB::bind_method(D_METHOD("mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPAutoEncoder::mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPAutoEncoder::score);

	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPAutoEncoder::save);

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPAutoEncoder::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPAutoEncoder::initialize);
	*/
}
