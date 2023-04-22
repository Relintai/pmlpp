//
//  SoftmaxNet.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "softmax_net_old.h"

#include "../activation/activation_old.h"
#include "../cost/cost_old.h"
#include "../data/data.h"
#include "../lin_alg/lin_alg_old.h"
#include "../regularization/reg_old.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

MLPPSoftmaxNetOld::MLPPSoftmaxNetOld(std::vector<std::vector<real_t>> pinputSet, std::vector<std::vector<real_t>> poutputSet, int pn_hidden, std::string preg, real_t plambda, real_t palpha) {
	inputSet = pinputSet;
	outputSet = poutputSet;
	n = pinputSet.size();
	k = pinputSet[0].size();
	n_hidden = pn_hidden;
	n_class = poutputSet[0].size();
	reg = preg;
	lambda = plambda;
	alpha = palpha;

	y_hat.resize(n);

	weights1 = MLPPUtilities::weightInitialization(k, n_hidden);
	weights2 = MLPPUtilities::weightInitialization(n_hidden, n_class);
	bias1 = MLPPUtilities::biasInitialization(n_hidden);
	bias2 = MLPPUtilities::biasInitialization(n_class);
}

std::vector<real_t> MLPPSoftmaxNetOld::modelTest(std::vector<real_t> x) {
	return Evaluate(x);
}

std::vector<std::vector<real_t>> MLPPSoftmaxNetOld::modelSetTest(std::vector<std::vector<real_t>> X) {
	return Evaluate(X);
}

void MLPPSoftmaxNetOld::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	MLPPActivationOld avn;
	MLPPLinAlgOld alg;
	MLPPRegOld regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, outputSet);

		// Calculating the errors
		std::vector<std::vector<real_t>> error = alg.subtraction(y_hat, outputSet);

		// Calculating the weight/bias gradients for layer 2

		std::vector<std::vector<real_t>> D2_1 = alg.matmult(alg.transpose(a2), error);

		// weights and bias updation for layer 2
		weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate, D2_1));
		weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

		bias2 = alg.subtractMatrixRows(bias2, alg.scalarMultiply(learning_rate, error));

		//Calculating the weight/bias for layer 1

		std::vector<std::vector<real_t>> D1_1 = alg.matmult(error, alg.transpose(weights2));

		std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));

		std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(inputSet), D1_2);

		// weight an bias updation for layer 1
		weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate, D1_3));
		weights1 = regularization.regWeights(weights1, lambda, alpha, reg);

		bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate, D1_2));

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

void MLPPSoftmaxNetOld::SGD(real_t learning_rate, int max_epoch, bool UI) {
	MLPPActivationOld avn;
	MLPPLinAlgOld alg;
	MLPPRegOld regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	while (true) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(n - 1));
		int outputIndex = distribution(generator);

		std::vector<real_t> y_hat = Evaluate(inputSet[outputIndex]);

		auto prop_res = propagate(inputSet[outputIndex]);
		auto z2 = std::get<0>(prop_res);
		auto a2 = std::get<1>(prop_res);

		cost_prev = Cost({ y_hat }, { outputSet[outputIndex] });
		std::vector<real_t> error = alg.subtraction(y_hat, outputSet[outputIndex]);

		// Weight updation for layer 2
		std::vector<std::vector<real_t>> D2_1 = alg.outerProduct(error, a2);
		weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate, alg.transpose(D2_1)));
		weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

		// Bias updation for layer 2
		bias2 = alg.subtraction(bias2, alg.scalarMultiply(learning_rate, error));

		// Weight updation for layer 1
		std::vector<real_t> D1_1 = alg.mat_vec_mult(weights2, error);
		std::vector<real_t> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, true));
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

void MLPPSoftmaxNetOld::MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	MLPPActivationOld avn;
	MLPPLinAlgOld alg;
	MLPPRegOld regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;

	auto batches = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
	auto inputMiniBatches = std::get<0>(batches);
	auto outputMiniBatches = std::get<1>(batches);

	// Creating the mini-batches
	for (int i = 0; i < n_mini_batch; i++) {
		std::vector<std::vector<real_t>> currentInputSet;
		std::vector<std::vector<real_t>> currentOutputSet;
		for (int j = 0; j < n / n_mini_batch; j++) {
			currentInputSet.push_back(inputSet[n / n_mini_batch * i + j]);
			currentOutputSet.push_back(outputSet[n / n_mini_batch * i + j]);
		}
		inputMiniBatches.push_back(currentInputSet);
		outputMiniBatches.push_back(currentOutputSet);
	}

	if (real_t(n) / real_t(n_mini_batch) - int(n / n_mini_batch) != 0) {
		for (int i = 0; i < n - n / n_mini_batch * n_mini_batch; i++) {
			inputMiniBatches[n_mini_batch - 1].push_back(inputSet[n / n_mini_batch * n_mini_batch + i]);
			outputMiniBatches[n_mini_batch - 1].push_back(outputSet[n / n_mini_batch * n_mini_batch + i]);
		}
	}

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<std::vector<real_t>> y_hat = Evaluate(inputMiniBatches[i]);

			auto propagate_res = propagate(inputMiniBatches[i]);
			auto z2 = std::get<0>(propagate_res);
			auto a2 = std::get<1>(propagate_res);

			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			// Calculating the errors
			std::vector<std::vector<real_t>> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight/bias gradients for layer 2

			std::vector<std::vector<real_t>> D2_1 = alg.matmult(alg.transpose(a2), error);

			// weights and bias updation for layser 2
			weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate, D2_1));
			weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

			// Bias Updation for layer 2
			bias2 = alg.subtractMatrixRows(bias2, alg.scalarMultiply(learning_rate, error));

			//Calculating the weight/bias for layer 1

			std::vector<std::vector<real_t>> D1_1 = alg.matmult(error, alg.transpose(weights2));

			std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));

			std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(inputMiniBatches[i]), D1_2);

			// weight an bias updation for layer 1
			weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate, D1_3));
			weights1 = regularization.regWeights(weights1, lambda, alpha, reg);

			bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate, D1_2));

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

real_t MLPPSoftmaxNetOld::score() {
	MLPPUtilities util;
	return util.performance(y_hat, outputSet);
}

void MLPPSoftmaxNetOld::save(std::string fileName) {
	MLPPUtilities util;
	util.saveParameters(fileName, weights1, bias1, 0, 1);
	util.saveParameters(fileName, weights2, bias2, 1, 2);
}

std::vector<std::vector<real_t>> MLPPSoftmaxNetOld::getEmbeddings() {
	return weights1;
}

real_t MLPPSoftmaxNetOld::Cost(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPRegOld regularization;
	MLPPData data;
	class MLPPCostOld cost;
	return cost.CrossEntropy(y_hat, y) + regularization.regTerm(weights1, lambda, alpha, reg) + regularization.regTerm(weights2, lambda, alpha, reg);
}

std::vector<std::vector<real_t>> MLPPSoftmaxNetOld::Evaluate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlgOld alg;
	MLPPActivationOld avn;
	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);
	return avn.adjSoftmax(alg.mat_vec_add(alg.matmult(a2, weights2), bias2));
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPSoftmaxNetOld::propagate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlgOld alg;
	MLPPActivationOld avn;
	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);
	return { z2, a2 };
}

std::vector<real_t> MLPPSoftmaxNetOld::Evaluate(std::vector<real_t> x) {
	MLPPLinAlgOld alg;
	MLPPActivationOld avn;
	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);
	return avn.adjSoftmax(alg.addition(alg.mat_vec_mult(alg.transpose(weights2), a2), bias2));
}

std::tuple<std::vector<real_t>, std::vector<real_t>> MLPPSoftmaxNetOld::propagate(std::vector<real_t> x) {
	MLPPLinAlgOld alg;
	MLPPActivationOld avn;
	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);
	return { z2, a2 };
}

void MLPPSoftmaxNetOld::forwardPass() {
	MLPPLinAlgOld alg;
	MLPPActivationOld avn;
	z2 = alg.mat_vec_add(alg.matmult(inputSet, weights1), bias1);
	a2 = avn.sigmoid(z2);
	y_hat = avn.adjSoftmax(alg.mat_vec_add(alg.matmult(a2, weights2), bias2));
}
