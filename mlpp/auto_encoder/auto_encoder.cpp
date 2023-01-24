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

#include <iostream>
#include <random>

MLPPAutoEncoder::MLPPAutoEncoder(std::vector<std::vector<double>> inputSet, int n_hidden) :
		inputSet(inputSet), n_hidden(n_hidden), n(inputSet.size()), k(inputSet[0].size()) {
	MLPPActivation avn;
	y_hat.resize(inputSet.size());

	weights1 = Utilities::weightInitialization(k, n_hidden);
	weights2 = Utilities::weightInitialization(n_hidden, k);
	bias1 = Utilities::biasInitialization(n_hidden);
	bias2 = Utilities::biasInitialization(k);
}

std::vector<std::vector<double>> MLPPAutoEncoder::modelSetTest(std::vector<std::vector<double>> X) {
	return Evaluate(X);
}

std::vector<double> MLPPAutoEncoder::modelTest(std::vector<double> x) {
	return Evaluate(x);
}

void MLPPAutoEncoder::gradientDescent(double learning_rate, int max_epoch, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	double cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, inputSet);

		// Calculating the errors
		std::vector<std::vector<double>> error = alg.subtraction(y_hat, inputSet);

		// Calculating the weight/bias gradients for layer 2
		std::vector<std::vector<double>> D2_1 = alg.matmult(alg.transpose(a2), error);

		// weights and bias updation for layer 2
		weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate / n, D2_1));

		// Calculating the bias gradients for layer 2
		bias2 = alg.subtractMatrixRows(bias2, alg.scalarMultiply(learning_rate, error));

		//Calculating the weight/bias for layer 1

		std::vector<std::vector<double>> D1_1 = alg.matmult(error, alg.transpose(weights2));

		std::vector<std::vector<double>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));

		std::vector<std::vector<double>> D1_3 = alg.matmult(alg.transpose(inputSet), D1_2);

		// weight an bias updation for layer 1
		weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate / n, D1_3));

		bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate / n, D1_2));

		forwardPass();

		// UI PORTION
		if (UI) {
			Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, inputSet));
			std::cout << "Layer 1:" << std::endl;
			Utilities::UI(weights1, bias1);
			std::cout << "Layer 2:" << std::endl;
			Utilities::UI(weights2, bias2);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPAutoEncoder::SGD(double learning_rate, int max_epoch, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	double cost_prev = 0;
	int epoch = 1;

	while (true) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(n - 1));
		int outputIndex = distribution(generator);

		std::vector<double> y_hat = Evaluate(inputSet[outputIndex]);
		auto [z2, a2] = propagate(inputSet[outputIndex]);
		cost_prev = Cost({ y_hat }, { inputSet[outputIndex] });
		std::vector<double> error = alg.subtraction(y_hat, inputSet[outputIndex]);

		// Weight updation for layer 2
		std::vector<std::vector<double>> D2_1 = alg.outerProduct(error, a2);
		weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate, alg.transpose(D2_1)));

		// Bias updation for layer 2
		bias2 = alg.subtraction(bias2, alg.scalarMultiply(learning_rate, error));

		// Weight updation for layer 1
		std::vector<double> D1_1 = alg.mat_vec_mult(weights2, error);
		std::vector<double> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));
		std::vector<std::vector<double>> D1_3 = alg.outerProduct(inputSet[outputIndex], D1_2);

		weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate, D1_3));
		// Bias updation for layer 1

		bias1 = alg.subtraction(bias1, alg.scalarMultiply(learning_rate, D1_2));

		y_hat = Evaluate(inputSet[outputIndex]);
		if (UI) {
			Utilities::CostInfo(epoch, cost_prev, Cost({ y_hat }, { inputSet[outputIndex] }));
			std::cout << "Layer 1:" << std::endl;
			Utilities::UI(weights1, bias1);
			std::cout << "Layer 2:" << std::endl;
			Utilities::UI(weights2, bias2);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPAutoEncoder::MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	double cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	std::vector<std::vector<std::vector<double>>> inputMiniBatches = Utilities::createMiniBatches(inputSet, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<std::vector<double>> y_hat = Evaluate(inputMiniBatches[i]);
			auto [z2, a2] = propagate(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, inputMiniBatches[i]);

			// Calculating the errors
			std::vector<std::vector<double>> error = alg.subtraction(y_hat, inputMiniBatches[i]);

			// Calculating the weight/bias gradients for layer 2

			std::vector<std::vector<double>> D2_1 = alg.matmult(alg.transpose(a2), error);

			// weights and bias updation for layer 2
			weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate / inputMiniBatches[i].size(), D2_1));

			// Bias Updation for layer 2
			bias2 = alg.subtractMatrixRows(bias2, alg.scalarMultiply(learning_rate, error));

			//Calculating the weight/bias for layer 1

			std::vector<std::vector<double>> D1_1 = alg.matmult(error, alg.transpose(weights2));

			std::vector<std::vector<double>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, 1));

			std::vector<std::vector<double>> D1_3 = alg.matmult(alg.transpose(inputMiniBatches[i]), D1_2);

			// weight an bias updation for layer 1
			weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate / inputMiniBatches[i].size(), D1_3));

			bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate / inputMiniBatches[i].size(), D1_2));

			y_hat = Evaluate(inputMiniBatches[i]);

			if (UI) {
				Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, inputMiniBatches[i]));
				std::cout << "Layer 1:" << std::endl;
				Utilities::UI(weights1, bias1);
				std::cout << "Layer 2:" << std::endl;
				Utilities::UI(weights2, bias2);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

double MLPPAutoEncoder::score() {
	Utilities util;
	return util.performance(y_hat, inputSet);
}

void MLPPAutoEncoder::save(std::string fileName) {
	Utilities util;
	util.saveParameters(fileName, weights1, bias1, 0, 1);
	util.saveParameters(fileName, weights2, bias2, 1, 2);
}

double MLPPAutoEncoder::Cost(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	class MLPPCost cost;
	return cost.MSE(y_hat, inputSet);
}

std::vector<std::vector<double>> MLPPAutoEncoder::Evaluate(std::vector<std::vector<double>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<std::vector<double>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
	std::vector<std::vector<double>> a2 = avn.sigmoid(z2);
	return alg.mat_vec_add(alg.matmult(a2, weights2), bias2);
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> MLPPAutoEncoder::propagate(std::vector<std::vector<double>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<std::vector<double>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
	std::vector<std::vector<double>> a2 = avn.sigmoid(z2);
	return { z2, a2 };
}

std::vector<double> MLPPAutoEncoder::Evaluate(std::vector<double> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<double> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1);
	std::vector<double> a2 = avn.sigmoid(z2);
	return alg.addition(alg.mat_vec_mult(alg.transpose(weights2), a2), bias2);
}

std::tuple<std::vector<double>, std::vector<double>> MLPPAutoEncoder::propagate(std::vector<double> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<double> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1);
	std::vector<double> a2 = avn.sigmoid(z2);
	return { z2, a2 };
}

void MLPPAutoEncoder::forwardPass() {
	MLPPLinAlg alg;
	MLPPActivation avn;
	z2 = alg.mat_vec_add(alg.matmult(inputSet, weights1), bias1);
	a2 = avn.sigmoid(z2);
	y_hat = alg.mat_vec_add(alg.matmult(a2, weights2), bias2);
}
