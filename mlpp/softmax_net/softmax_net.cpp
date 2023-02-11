//
//  SoftmaxNet.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "softmax_net.h"
#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../data/data.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

/*
Ref<MLPPMatrix> MLPPSoftmaxNet::get_input_set() {
	return _input_set;
}
void MLPPSoftmaxNet::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;

	_initialized = false;
}

Ref<MLPPMatrix> MLPPSoftmaxNet::get_output_set() {
	return _output_set;
}
void MLPPSoftmaxNet::set_output_set(const Ref<MLPPMatrix> &val) {
	_output_set = val;

	_initialized = false;
}

MLPPReg::RegularizationType MLPPSoftmaxNet::get_reg() {
	return _reg;
}
void MLPPSoftmaxNet::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;

	_initialized = false;
}

real_t MLPPSoftmaxNet::get_lambda() {
	return _lambda;
}
void MLPPSoftmaxNet::set_lambda(const real_t val) {
	_lambda = val;

	_initialized = false;
}

real_t MLPPSoftmaxNet::get_alpha() {
	return _alpha;
}
void MLPPSoftmaxNet::set_alpha(const real_t val) {
	_alpha = val;

	_initialized = false;
}
*/

std::vector<real_t> MLPPSoftmaxNet::model_test(std::vector<real_t> x) {
	return evaluatev(x);
}

std::vector<std::vector<real_t>> MLPPSoftmaxNet::model_set_test(std::vector<std::vector<real_t>> X) {
	return evaluatem(X);
}

void MLPPSoftmaxNet::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		// Calculating the errors
		std::vector<std::vector<real_t>> error = alg.subtraction(_y_hat, _output_set);

		// Calculating the weight/bias gradients for layer 2

		std::vector<std::vector<real_t>> D2_1 = alg.matmult(alg.transpose(_a2), error);

		// weights and bias updation for layer 2
		_weights2 = alg.subtraction(_weights2, alg.scalarMultiply(learning_rate, D2_1));
		//_reg
		_weights2 = regularization.regWeights(_weights2, _lambda, _alpha, "None");

		_bias2 = alg.subtractMatrixRows(_bias2, alg.scalarMultiply(learning_rate, error));

		//Calculating the weight/bias for layer 1

		std::vector<std::vector<real_t>> D1_1 = alg.matmult(error, alg.transpose(_weights2));

		std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(_z2, true));

		std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(_input_set), D1_2);

		// weight an bias updation for layer 1
		_weights1 = alg.subtraction(_weights1, alg.scalarMultiply(learning_rate, D1_3));
		//_reg
		_weights1 = regularization.regWeights(_weights1, _lambda, _alpha, "None");

		_bias1 = alg.subtractMatrixRows(_bias1, alg.scalarMultiply(learning_rate, D1_2));

		forward_pass();

		// UI PORTION
		if (ui) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost(_y_hat, _output_set));
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

void MLPPSoftmaxNet::sgd(real_t learning_rate, int max_epoch, bool ui) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(_n - 1));

	while (true) {
		int outputIndex = distribution(generator);

		std::vector<real_t> y_hat = evaluatev(_input_set[outputIndex]);

		auto prop_res = propagatev(_input_set[outputIndex]);
		auto z2 = std::get<0>(prop_res);
		auto a2 = std::get<1>(prop_res);

		cost_prev = cost({ y_hat }, { _output_set[outputIndex] });
		std::vector<real_t> error = alg.subtraction(y_hat, _output_set[outputIndex]);

		// Weight updation for layer 2
		std::vector<std::vector<real_t>> D2_1 = alg.outerProduct(error, a2);
		_weights2 = alg.subtraction(_weights2, alg.scalarMultiply(learning_rate, alg.transpose(D2_1)));
		//_reg
		_weights2 = regularization.regWeights(_weights2, _lambda, _alpha, "None");

		// Bias updation for layer 2
		_bias2 = alg.subtraction(_bias2, alg.scalarMultiply(learning_rate, error));

		// Weight updation for layer 1
		std::vector<real_t> D1_1 = alg.mat_vec_mult(_weights2, error);
		std::vector<real_t> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, true));
		std::vector<std::vector<real_t>> D1_3 = alg.outerProduct(_input_set[outputIndex], D1_2);

		_weights1 = alg.subtraction(_weights1, alg.scalarMultiply(learning_rate, D1_3));
		//_reg
		_weights1 = regularization.regWeights(_weights1, _lambda, _alpha, "None");
		// Bias updation for layer 1

		_bias1 = alg.subtraction(_bias1, alg.scalarMultiply(learning_rate, D1_2));

		y_hat = evaluatev(_input_set[outputIndex]);

		if (ui) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost({ y_hat }, { _output_set[outputIndex] }));
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

void MLPPSoftmaxNet::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;

	auto batches = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);
	auto inputMiniBatches = std::get<0>(batches);
	auto outputMiniBatches = std::get<1>(batches);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<std::vector<real_t>> y_hat = evaluatem(inputMiniBatches[i]);

			auto propagate_res = propagatem(inputMiniBatches[i]);
			auto z2 = std::get<0>(propagate_res);
			auto a2 = std::get<1>(propagate_res);

			cost_prev = cost(y_hat, outputMiniBatches[i]);

			// Calculating the errors
			std::vector<std::vector<real_t>> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight/bias gradients for layer 2

			std::vector<std::vector<real_t>> D2_1 = alg.matmult(alg.transpose(a2), error);

			// weights and bias updation for layser 2
			_weights2 = alg.subtraction(_weights2, alg.scalarMultiply(learning_rate, D2_1));
			//_reg
			_weights2 = regularization.regWeights(_weights2, _lambda, _alpha, "None");

			// Bias Updation for layer 2
			_bias2 = alg.subtractMatrixRows(_bias2, alg.scalarMultiply(learning_rate, error));

			//Calculating the weight/bias for layer 1

			std::vector<std::vector<real_t>> D1_1 = alg.matmult(error, alg.transpose(_weights2));

			std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, true));

			std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(inputMiniBatches[i]), D1_2);

			// weight an bias updation for layer 1
			_weights1 = alg.subtraction(_weights1, alg.scalarMultiply(learning_rate, D1_3));
			//_reg
			_weights1 = regularization.regWeights(_weights1, _lambda, _alpha, "None");

			_bias1 = alg.subtractMatrixRows(_bias1, alg.scalarMultiply(learning_rate, D1_2));

			y_hat = evaluatem(inputMiniBatches[i]);

			if (ui) {
				MLPPUtilities::CostInfo(epoch, cost_prev, cost(y_hat, outputMiniBatches[i]));
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

real_t MLPPSoftmaxNet::score() {
	MLPPUtilities util;

	return util.performance(_y_hat, _output_set);
}

void MLPPSoftmaxNet::save(std::string fileName) {
	MLPPUtilities util;

	util.saveParameters(fileName, _weights1, _bias1, false, 1);
	util.saveParameters(fileName, _weights2, _bias2, true, 2);
}

std::vector<std::vector<real_t>> MLPPSoftmaxNet::get_embeddings() {
	return _weights1;
}

bool MLPPSoftmaxNet::is_initialized() {
	return _initialized;
}
void MLPPSoftmaxNet::initialize() {
	if (_initialized) {
		return;
	}

	//ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	_initialized = true;
}

MLPPSoftmaxNet::MLPPSoftmaxNet(std::vector<std::vector<real_t>> p_input_set, std::vector<std::vector<real_t>> p_output_set, int p_n_hidden, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = p_input_set.size();
	_k = p_input_set[0].size();
	_n_hidden = p_n_hidden;
	_n_class = p_output_set[0].size();
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.resize(_n);

	_weights1 = MLPPUtilities::weightInitialization(_k, _n_hidden);
	_weights2 = MLPPUtilities::weightInitialization(_n_hidden, _n_class);
	_bias1 = MLPPUtilities::biasInitialization(_n_hidden);
	_bias2 = MLPPUtilities::biasInitialization(_n_class);

	_initialized = true;
}

MLPPSoftmaxNet::MLPPSoftmaxNet() {
	_initialized = false;
}
MLPPSoftmaxNet::~MLPPSoftmaxNet() {
}

real_t MLPPSoftmaxNet::cost(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPReg regularization;
	MLPPData data;
	class MLPPCost cost;

	//_reg
	return cost.CrossEntropy(y_hat, y) + regularization.regTerm(_weights1, _lambda, _alpha, "None") + regularization.regTerm(_weights2, _lambda, _alpha, "None");
}

std::vector<real_t> MLPPSoftmaxNet::evaluatev(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(_weights1), x), _bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);

	return avn.adjSoftmax(alg.addition(alg.mat_vec_mult(alg.transpose(_weights2), a2), _bias2));
}

std::tuple<std::vector<real_t>, std::vector<real_t>> MLPPSoftmaxNet::propagatev(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(_weights1), x), _bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);

	return { z2, a2 };
}

std::vector<std::vector<real_t>> MLPPSoftmaxNet::evaluatem(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, _weights1), _bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);

	return avn.adjSoftmax(alg.mat_vec_add(alg.matmult(a2, _weights2), _bias2));
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPSoftmaxNet::propagatem(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, _weights1), _bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);

	return { z2, a2 };
}

void MLPPSoftmaxNet::forward_pass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	_z2 = alg.mat_vec_add(alg.matmult(_input_set, _weights1), _bias1);
	_a2 = avn.sigmoid(_z2);
	_y_hat = avn.adjSoftmax(alg.mat_vec_add(alg.matmult(_a2, _weights2), _bias2));
}

void MLPPSoftmaxNet::_bind_methods() {
	/*
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPSoftmaxNet::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPSoftmaxNet::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPSoftmaxNet::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPSoftmaxNet::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_reg"), &MLPPSoftmaxNet::get_reg);
	ClassDB::bind_method(D_METHOD("set_reg", "val"), &MLPPSoftmaxNet::set_reg);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reg"), "set_reg", "get_reg");

	ClassDB::bind_method(D_METHOD("get_lambda"), &MLPPSoftmaxNet::get_lambda);
	ClassDB::bind_method(D_METHOD("set_lambda", "val"), &MLPPSoftmaxNet::set_lambda);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lambda"), "set_lambda", "get_lambda");

	ClassDB::bind_method(D_METHOD("get_alpha"), &MLPPSoftmaxNet::get_alpha);
	ClassDB::bind_method(D_METHOD("set_alpha", "val"), &MLPPSoftmaxNet::set_alpha);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alpha"), "set_alpha", "get_alpha");

	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPSoftmaxNet::model_test);
	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPSoftmaxNet::model_set_test);

	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPSoftmaxNet::gradient_descent, false);
	ClassDB::bind_method(D_METHOD("sgd", "learning_rate", "max_epoch", "ui"), &MLPPSoftmaxNet::sgd, false);
	ClassDB::bind_method(D_METHOD("mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPSoftmaxNet::mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPSoftmaxNet::score);

	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPSoftmaxNet::save);

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPSoftmaxNet::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPSoftmaxNet::initialize);
	*/
}
