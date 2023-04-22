//
//  AutoEncoder.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "auto_encoder_old.h"

#include "../activation/activation_old.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

std::vector<std::vector<real_t>> MLPPAutoEncoderOld::modelSetTest(std::vector<std::vector<real_t>> X) {
	return Evaluate(X);
}

std::vector<real_t> MLPPAutoEncoderOld::modelTest(std::vector<real_t> x) {
	return Evaluate(x);
}

void MLPPAutoEncoderOld::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	MLPPActivationOld avn;
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, inputSet);

		// Calculating the errors
		std::vector<std::vector<real_t>> error = alg.subtraction(y_hat, inputSet);

		// Calculating the weight/bias gradients for layer 2
		std::vector<std::vector<real_t>> D2_1 = alg.matmult(alg.transpose(a2), error);

		// weights and bias updation for layer 2
		weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate / n, D2_1));

		// Calculating the bias gradients for layer 2
		bias2 = alg.subtractMatrixRows(bias2, alg.scalarMultiply(learning_rate, error));

		//Calculating the weight/bias for layer 1

		std::vector<std::vector<real_t>> D1_1 = alg.matmult(error, alg.transpose(weights2));

		std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));

		std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(inputSet), D1_2);

		// weight an bias updation for layer 1
		weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate / n, D1_3));

		bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate / n, D1_2));

		forwardPass();

		// UI PORTION
		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, Cost(y_hat, inputSet));
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

void MLPPAutoEncoderOld::SGD(real_t learning_rate, int max_epoch, bool UI) {
	MLPPActivationOld avn;
	MLPPLinAlg alg;
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

		cost_prev = Cost({ y_hat }, { inputSet[outputIndex] });
		std::vector<real_t> error = alg.subtraction(y_hat, inputSet[outputIndex]);

		// Weight updation for layer 2
		std::vector<std::vector<real_t>> D2_1 = alg.outerProduct(error, a2);
		weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate, alg.transpose(D2_1)));

		// Bias updation for layer 2
		bias2 = alg.subtraction(bias2, alg.scalarMultiply(learning_rate, error));

		// Weight updation for layer 1
		std::vector<real_t> D1_1 = alg.mat_vec_mult(weights2, error);
		std::vector<real_t> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));
		std::vector<std::vector<real_t>> D1_3 = alg.outerProduct(inputSet[outputIndex], D1_2);

		weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate, D1_3));
		// Bias updation for layer 1

		bias1 = alg.subtraction(bias1, alg.scalarMultiply(learning_rate, D1_2));

		y_hat = Evaluate(inputSet[outputIndex]);
		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, Cost({ y_hat }, { inputSet[outputIndex] }));
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

void MLPPAutoEncoderOld::MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	MLPPActivationOld avn;
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	std::vector<std::vector<std::vector<real_t>>> inputMiniBatches = MLPPUtilities::createMiniBatches(inputSet, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<std::vector<real_t>> y_hat = Evaluate(inputMiniBatches[i]);

			auto prop_res = propagate(inputMiniBatches[i]);
			auto z2 = std::get<0>(prop_res);
			auto a2 = std::get<1>(prop_res);

			cost_prev = Cost(y_hat, inputMiniBatches[i]);

			// Calculating the errors
			std::vector<std::vector<real_t>> error = alg.subtraction(y_hat, inputMiniBatches[i]);

			// Calculating the weight/bias gradients for layer 2

			std::vector<std::vector<real_t>> D2_1 = alg.matmult(alg.transpose(a2), error);

			// weights and bias updation for layer 2
			weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate / inputMiniBatches[i].size(), D2_1));

			// Bias Updation for layer 2
			bias2 = alg.subtractMatrixRows(bias2, alg.scalarMultiply(learning_rate, error));

			//Calculating the weight/bias for layer 1

			std::vector<std::vector<real_t>> D1_1 = alg.matmult(error, alg.transpose(weights2));

			std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));

			std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(inputMiniBatches[i]), D1_2);

			// weight an bias updation for layer 1
			weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate / inputMiniBatches[i].size(), D1_3));

			bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate / inputMiniBatches[i].size(), D1_2));

			y_hat = Evaluate(inputMiniBatches[i]);

			if (UI) {
				MLPPUtilities::CostInfo(epoch, cost_prev, Cost(y_hat, inputMiniBatches[i]));
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

real_t MLPPAutoEncoderOld::score() {
	MLPPUtilities util;
	return util.performance(y_hat, inputSet);
}

void MLPPAutoEncoderOld::save(std::string fileName) {
	MLPPUtilities util;
	util.saveParameters(fileName, weights1, bias1, 0, 1);
	util.saveParameters(fileName, weights2, bias2, 1, 2);
}

MLPPAutoEncoderOld::MLPPAutoEncoderOld(std::vector<std::vector<real_t>> pinputSet, int pn_hidden) {
	inputSet = pinputSet;
	n_hidden = pn_hidden;
	n = inputSet.size();
	k = inputSet[0].size();

	MLPPActivationOld avn;
	y_hat.resize(inputSet.size());

	weights1 = MLPPUtilities::weightInitialization(k, n_hidden);
	weights2 = MLPPUtilities::weightInitialization(n_hidden, k);
	bias1 = MLPPUtilities::biasInitialization(n_hidden);
	bias2 = MLPPUtilities::biasInitialization(k);
}

real_t MLPPAutoEncoderOld::Cost(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	class MLPPCost cost;
	return cost.MSE(y_hat, inputSet);
}

std::vector<std::vector<real_t>> MLPPAutoEncoderOld::Evaluate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivationOld avn;
	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);
	return alg.mat_vec_add(alg.matmult(a2, weights2), bias2);
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPAutoEncoderOld::propagate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivationOld avn;
	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);
	return { z2, a2 };
}

std::vector<real_t> MLPPAutoEncoderOld::Evaluate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivationOld avn;
	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);
	return alg.addition(alg.mat_vec_mult(alg.transpose(weights2), a2), bias2);
}

std::tuple<std::vector<real_t>, std::vector<real_t>> MLPPAutoEncoderOld::propagate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivationOld avn;
	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);
	return { z2, a2 };
}

void MLPPAutoEncoderOld::forwardPass() {
	MLPPLinAlg alg;
	MLPPActivationOld avn;
	z2 = alg.mat_vec_add(alg.matmult(inputSet, weights1), bias1);
	a2 = avn.sigmoid(z2);
	y_hat = alg.mat_vec_add(alg.matmult(a2, weights2), bias2);
}
