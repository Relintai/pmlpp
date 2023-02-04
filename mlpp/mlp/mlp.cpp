//
//  MLP.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "mlp.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

std::vector<real_t> MLPPMLP::model_set_test(std::vector<std::vector<real_t>> X) {
	return evaluate(X);
}

real_t MLPPMLP::model_test(std::vector<real_t> x) {
	return evaluate(x);
}

void MLPPMLP::gradient_descent(real_t learning_rate, int max_epoch, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(y_hat, outputSet);

		// Calculating the errors
		std::vector<real_t> error = alg.subtraction(y_hat, outputSet);

		// Calculating the weight/bias gradients for layer 2

		std::vector<real_t> D2_1 = alg.mat_vec_mult(alg.transpose(a2), error);

		// weights and bias updation for layer 2
		weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate / n, D2_1));
		weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

		bias2 -= learning_rate * alg.sum_elements(error) / n;

		// Calculating the weight/bias for layer 1

		std::vector<std::vector<real_t>> D1_1;
		D1_1.resize(n);

		D1_1 = alg.outerProduct(error, weights2);

		std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));

		std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(inputSet), D1_2);

		// weight an bias updation for layer 1
		weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate / n, D1_3));
		weights1 = regularization.regWeights(weights1, lambda, alpha, reg);

		bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate / n, D1_2));

		forward_pass();

		// UI PORTION
		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost(y_hat, outputSet));
			std::cout << "Layer 1:" << std::endl;
			MLPPUtilities::UI(weights1, bias1);
			std::cout << "Layer 2:" << std::endl;
			MLPPUtilities::UI(weights2, bias2);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPMLP::sgd(real_t learning_rate, int max_epoch, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	while (true) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(n - 1));
		int outputIndex = distribution(generator);

		real_t y_hat = evaluate(inputSet[outputIndex]);
		auto [z2, a2] = propagate(inputSet[outputIndex]);
		cost_prev = cost({ y_hat }, { outputSet[outputIndex] });
		real_t error = y_hat - outputSet[outputIndex];

		// Weight updation for layer 2
		std::vector<real_t> D2_1 = alg.scalarMultiply(error, a2);
		weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate, D2_1));
		weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

		// Bias updation for layer 2
		bias2 -= learning_rate * error;

		// Weight updation for layer 1
		std::vector<real_t> D1_1 = alg.scalarMultiply(error, weights2);
		std::vector<real_t> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));
		std::vector<std::vector<real_t>> D1_3 = alg.outerProduct(inputSet[outputIndex], D1_2);

		weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate, D1_3));
		weights1 = regularization.regWeights(weights1, lambda, alpha, reg);
		// Bias updation for layer 1

		bias1 = alg.subtraction(bias1, alg.scalarMultiply(learning_rate, D1_2));

		y_hat = evaluate(inputSet[outputIndex]);
		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost({ y_hat }, { outputSet[outputIndex] }));
			std::cout << "Layer 1:" << std::endl;
			MLPPUtilities::UI(weights1, bias1);
			std::cout << "Layer 2:" << std::endl;
			MLPPUtilities::UI(weights2, bias2);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPMLP::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto [inputMiniBatches, outputMiniBatches] = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = evaluate(inputMiniBatches[i]);
			auto [z2, a2] = propagate(inputMiniBatches[i]);
			cost_prev = cost(y_hat, outputMiniBatches[i]);

			// Calculating the errors
			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight/bias gradients for layer 2

			std::vector<real_t> D2_1 = alg.mat_vec_mult(alg.transpose(a2), error);

			// weights and bias updation for layser 2
			weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate / outputMiniBatches[i].size(), D2_1));
			weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

			// Calculating the bias gradients for layer 2
			real_t b_gradient = alg.sum_elements(error);

			// Bias Updation for layer 2
			bias2 -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size();

			//Calculating the weight/bias for layer 1

			std::vector<std::vector<real_t>> D1_1 = alg.outerProduct(error, weights2);

			std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));

			std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(inputMiniBatches[i]), D1_2);

			// weight an bias updation for layer 1
			weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate / outputMiniBatches[i].size(), D1_3));
			weights1 = regularization.regWeights(weights1, lambda, alpha, reg);

			bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate / outputMiniBatches[i].size(), D1_2));

			y_hat = evaluate(inputMiniBatches[i]);

			if (UI) {
				MLPPUtilities::CostInfo(epoch, cost_prev, cost(y_hat, outputMiniBatches[i]));
				std::cout << "Layer 1:" << std::endl;
				MLPPUtilities::UI(weights1, bias1);
				std::cout << "Layer 2:" << std::endl;
				MLPPUtilities::UI(weights2, bias2);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

real_t MLPPMLP::score() {
	MLPPUtilities util;
	return util.performance(y_hat, outputSet);
}

void MLPPMLP::save(std::string fileName) {
	MLPPUtilities util;
	util.saveParameters(fileName, weights1, bias1, 0, 1);
	util.saveParameters(fileName, weights2, bias2, 1, 2);
}

real_t MLPPMLP::cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPReg regularization;
	class MLPPCost cost;
	return cost.LogLoss(y_hat, y) + regularization.regTerm(weights2, lambda, alpha, reg) + regularization.regTerm(weights1, lambda, alpha, reg);
}

std::vector<real_t> MLPPMLP::evaluate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);
	return avn.sigmoid(alg.scalarAdd(bias2, alg.mat_vec_mult(a2, weights2)));
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPMLP::propagate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);
	return { z2, a2 };
}

real_t MLPPMLP::evaluate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);
	return avn.sigmoid(alg.dot(weights2, a2) + bias2);
}

std::tuple<std::vector<real_t>, std::vector<real_t>> MLPPMLP::propagate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);
	return { z2, a2 };
}

void MLPPMLP::forward_pass() {
	MLPPLinAlg alg;
	MLPPActivation avn;
	z2 = alg.mat_vec_add(alg.matmult(inputSet, weights1), bias1);
	a2 = avn.sigmoid(z2);
	y_hat = avn.sigmoid(alg.scalarAdd(bias2, alg.mat_vec_mult(a2, weights2)));
}

MLPPMLP::MLPPMLP(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, int n_hidden, std::string reg, real_t lambda, real_t alpha) :
		inputSet(inputSet), outputSet(outputSet), n_hidden(n_hidden), n(inputSet.size()), k(inputSet[0].size()), reg(reg), lambda(lambda), alpha(alpha) {
	MLPPActivation avn;
	y_hat.resize(n);

	weights1 = MLPPUtilities::weightInitialization(k, n_hidden);
	weights2 = MLPPUtilities::weightInitialization(n_hidden);
	bias1 = MLPPUtilities::biasInitialization(n_hidden);
	bias2 = MLPPUtilities::biasInitialization();
}

MLPPMLP::MLPPMLP() {
}
MLPPMLP::~MLPPMLP() {
}

// =======    OLD    =======

MLPPMLPOld::MLPPMLPOld(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, int n_hidden, std::string reg, real_t lambda, real_t alpha) :
		inputSet(inputSet), outputSet(outputSet), n_hidden(n_hidden), n(inputSet.size()), k(inputSet[0].size()), reg(reg), lambda(lambda), alpha(alpha) {
	MLPPActivation avn;
	y_hat.resize(n);

	weights1 = MLPPUtilities::weightInitialization(k, n_hidden);
	weights2 = MLPPUtilities::weightInitialization(n_hidden);
	bias1 = MLPPUtilities::biasInitialization(n_hidden);
	bias2 = MLPPUtilities::biasInitialization();
}

std::vector<real_t> MLPPMLPOld::modelSetTest(std::vector<std::vector<real_t>> X) {
	return Evaluate(X);
}

real_t MLPPMLPOld::modelTest(std::vector<real_t> x) {
	return Evaluate(x);
}

void MLPPMLPOld::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, outputSet);

		// Calculating the errors
		std::vector<real_t> error = alg.subtraction(y_hat, outputSet);

		// Calculating the weight/bias gradients for layer 2

		std::vector<real_t> D2_1 = alg.mat_vec_mult(alg.transpose(a2), error);

		// weights and bias updation for layer 2
		weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate / n, D2_1));
		weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

		bias2 -= learning_rate * alg.sum_elements(error) / n;

		// Calculating the weight/bias for layer 1

		std::vector<std::vector<real_t>> D1_1;
		D1_1.resize(n);

		D1_1 = alg.outerProduct(error, weights2);

		std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));

		std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(inputSet), D1_2);

		// weight an bias updation for layer 1
		weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate / n, D1_3));
		weights1 = regularization.regWeights(weights1, lambda, alpha, reg);

		bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate / n, D1_2));

		forwardPass();

		// UI PORTION
		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
			std::cout << "Layer 1:" << std::endl;
			MLPPUtilities::UI(weights1, bias1);
			std::cout << "Layer 2:" << std::endl;
			MLPPUtilities::UI(weights2, bias2);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPMLPOld::SGD(real_t learning_rate, int max_epoch, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	while (true) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(n - 1));
		int outputIndex = distribution(generator);

		real_t y_hat = Evaluate(inputSet[outputIndex]);
		auto [z2, a2] = propagate(inputSet[outputIndex]);
		cost_prev = Cost({ y_hat }, { outputSet[outputIndex] });
		real_t error = y_hat - outputSet[outputIndex];

		// Weight updation for layer 2
		std::vector<real_t> D2_1 = alg.scalarMultiply(error, a2);
		weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate, D2_1));
		weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

		// Bias updation for layer 2
		bias2 -= learning_rate * error;

		// Weight updation for layer 1
		std::vector<real_t> D1_1 = alg.scalarMultiply(error, weights2);
		std::vector<real_t> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));
		std::vector<std::vector<real_t>> D1_3 = alg.outerProduct(inputSet[outputIndex], D1_2);

		weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate, D1_3));
		weights1 = regularization.regWeights(weights1, lambda, alpha, reg);
		// Bias updation for layer 1

		bias1 = alg.subtraction(bias1, alg.scalarMultiply(learning_rate, D1_2));

		y_hat = Evaluate(inputSet[outputIndex]);
		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, Cost({ y_hat }, { outputSet[outputIndex] }));
			std::cout << "Layer 1:" << std::endl;
			MLPPUtilities::UI(weights1, bias1);
			std::cout << "Layer 2:" << std::endl;
			MLPPUtilities::UI(weights2, bias2);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPMLPOld::MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto [inputMiniBatches, outputMiniBatches] = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = Evaluate(inputMiniBatches[i]);
			auto [z2, a2] = propagate(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			// Calculating the errors
			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight/bias gradients for layer 2

			std::vector<real_t> D2_1 = alg.mat_vec_mult(alg.transpose(a2), error);

			// weights and bias updation for layser 2
			weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate / outputMiniBatches[i].size(), D2_1));
			weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

			// Calculating the bias gradients for layer 2
			real_t b_gradient = alg.sum_elements(error);

			// Bias Updation for layer 2
			bias2 -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size();

			//Calculating the weight/bias for layer 1

			std::vector<std::vector<real_t>> D1_1 = alg.outerProduct(error, weights2);

			std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));

			std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(inputMiniBatches[i]), D1_2);

			// weight an bias updation for layer 1
			weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate / outputMiniBatches[i].size(), D1_3));
			weights1 = regularization.regWeights(weights1, lambda, alpha, reg);

			bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate / outputMiniBatches[i].size(), D1_2));

			y_hat = Evaluate(inputMiniBatches[i]);

			if (UI) {
				MLPPUtilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputMiniBatches[i]));
				std::cout << "Layer 1:" << std::endl;
				MLPPUtilities::UI(weights1, bias1);
				std::cout << "Layer 2:" << std::endl;
				MLPPUtilities::UI(weights2, bias2);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

real_t MLPPMLPOld::score() {
	MLPPUtilities util;
	return util.performance(y_hat, outputSet);
}

void MLPPMLPOld::save(std::string fileName) {
	MLPPUtilities util;
	util.saveParameters(fileName, weights1, bias1, 0, 1);
	util.saveParameters(fileName, weights2, bias2, 1, 2);
}

real_t MLPPMLPOld::Cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPReg regularization;
	class MLPPCost cost;
	return cost.LogLoss(y_hat, y) + regularization.regTerm(weights2, lambda, alpha, reg) + regularization.regTerm(weights1, lambda, alpha, reg);
}

std::vector<real_t> MLPPMLPOld::Evaluate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);
	return avn.sigmoid(alg.scalarAdd(bias2, alg.mat_vec_mult(a2, weights2)));
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPMLPOld::propagate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);
	return { z2, a2 };
}

real_t MLPPMLPOld::Evaluate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);
	return avn.sigmoid(alg.dot(weights2, a2) + bias2);
}

std::tuple<std::vector<real_t>, std::vector<real_t>> MLPPMLPOld::propagate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);
	return { z2, a2 };
}

void MLPPMLPOld::forwardPass() {
	MLPPLinAlg alg;
	MLPPActivation avn;
	z2 = alg.mat_vec_add(alg.matmult(inputSet, weights1), bias1);
	a2 = avn.sigmoid(z2);
	y_hat = avn.sigmoid(alg.scalarAdd(bias2, alg.mat_vec_mult(a2, weights2)));
}
