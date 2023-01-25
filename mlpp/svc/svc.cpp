//
//  SVC.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "svc.h"
#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>


MLPPSVC::MLPPSVC(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, double C) :
		inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size()), C(C) {
	y_hat.resize(n);
	weights = MLPPUtilities::weightInitialization(k);
	bias = MLPPUtilities::biasInitialization();
}

std::vector<double> MLPPSVC::modelSetTest(std::vector<std::vector<double>> X) {
	return Evaluate(X);
}

double MLPPSVC::modelTest(std::vector<double> x) {
	return Evaluate(x);
}

void MLPPSVC::gradientDescent(double learning_rate, int max_epoch, bool UI) {
	class MLPPCost cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	double cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, outputSet, weights, C);

		weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate / n, alg.mat_vec_mult(alg.transpose(inputSet), cost.HingeLossDeriv(z, outputSet, C))));
		weights = regularization.regWeights(weights, learning_rate / n, 0, "Ridge");

		// Calculating the bias gradients
		bias += learning_rate * alg.sum_elements(cost.HingeLossDeriv(y_hat, outputSet, C)) / n;

		forwardPass();

		// UI PORTION
		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet, weights, C));
			MLPPUtilities::UI(weights, bias);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPSVC::SGD(double learning_rate, int max_epoch, bool UI) {
	class MLPPCost cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	double cost_prev = 0;
	int epoch = 1;

	while (true) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(n - 1));
		int outputIndex = distribution(generator);

		double y_hat = Evaluate(inputSet[outputIndex]);
		double z = propagate(inputSet[outputIndex]);
		cost_prev = Cost({ z }, { outputSet[outputIndex] }, weights, C);

		double costDeriv = cost.HingeLossDeriv(std::vector<double>({ z }), std::vector<double>({ outputSet[outputIndex] }), C)[0]; // Explicit conversion to avoid ambiguity with overloaded function. Error occured on Ubuntu.

		// Weight Updation
		weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate * costDeriv, inputSet[outputIndex]));
		weights = regularization.regWeights(weights, learning_rate, 0, "Ridge");

		// Bias updation
		bias -= learning_rate * costDeriv;

		y_hat = Evaluate({ inputSet[outputIndex] });

		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, Cost({ z }, { outputSet[outputIndex] }, weights, C));
			MLPPUtilities::UI(weights, bias);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPSVC::MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	class MLPPCost cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	double cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto [inputMiniBatches, outputMiniBatches] = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<double> y_hat = Evaluate(inputMiniBatches[i]);
			std::vector<double> z = propagate(inputMiniBatches[i]);
			cost_prev = Cost(z, outputMiniBatches[i], weights, C);

			// Calculating the weight gradients
			weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate / n, alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), cost.HingeLossDeriv(z, outputMiniBatches[i], C))));
			weights = regularization.regWeights(weights, learning_rate / n, 0, "Ridge");

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elements(cost.HingeLossDeriv(y_hat, outputMiniBatches[i], C)) / n;

			forwardPass();

			y_hat = Evaluate(inputMiniBatches[i]);

			if (UI) {
				MLPPUtilities::CostInfo(epoch, cost_prev, Cost(z, outputMiniBatches[i], weights, C));
				MLPPUtilities::UI(weights, bias);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

double MLPPSVC::score() {
	MLPPUtilities   util;
	return util.performance(y_hat, outputSet);
}

void MLPPSVC::save(std::string fileName) {
	MLPPUtilities   util;
	util.saveParameters(fileName, weights, bias);
}

double MLPPSVC::Cost(std::vector<double> z, std::vector<double> y, std::vector<double> weights, double C) {
	class MLPPCost cost;
	return cost.HingeLoss(z, y, weights, C);
}

std::vector<double> MLPPSVC::Evaluate(std::vector<std::vector<double>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.sign(alg.scalarAdd(bias, alg.mat_vec_mult(X, weights)));
}

std::vector<double> MLPPSVC::propagate(std::vector<std::vector<double>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return alg.scalarAdd(bias, alg.mat_vec_mult(X, weights));
}

double MLPPSVC::Evaluate(std::vector<double> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.sign(alg.dot(weights, x) + bias);
}

double MLPPSVC::propagate(std::vector<double> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return alg.dot(weights, x) + bias;
}

// sign ( wTx + b )
void MLPPSVC::forwardPass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	z = propagate(inputSet);
	y_hat = avn.sign(z);
}
