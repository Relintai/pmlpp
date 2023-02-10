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

std::vector<real_t> MLPPSVC::modelSetTest(std::vector<std::vector<real_t>> X) {
	return Evaluate(X);
}

real_t MLPPSVC::modelTest(std::vector<real_t> x) {
	return Evaluate(x);
}

void MLPPSVC::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	class MLPPCost cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
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

void MLPPSVC::SGD(real_t learning_rate, int max_epoch, bool UI) {
	class MLPPCost cost;
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

		//real_t y_hat = Evaluate(inputSet[outputIndex]);
		real_t z = propagate(inputSet[outputIndex]);
		cost_prev = Cost({ z }, { outputSet[outputIndex] }, weights, C);

		real_t costDeriv = cost.HingeLossDeriv(std::vector<real_t>({ z }), std::vector<real_t>({ outputSet[outputIndex] }), C)[0]; // Explicit conversion to avoid ambiguity with overloaded function. Error occured on Ubuntu.

		// Weight Updation
		weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate * costDeriv, inputSet[outputIndex]));
		weights = regularization.regWeights(weights, learning_rate, 0, "Ridge");

		// Bias updation
		bias -= learning_rate * costDeriv;

		//y_hat = Evaluate({ inputSet[outputIndex] });

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

void MLPPSVC::MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	class MLPPCost cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto batches = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
	auto inputMiniBatches = std::get<0>(batches);
	auto outputMiniBatches = std::get<1>(batches);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = Evaluate(inputMiniBatches[i]);
			std::vector<real_t> z = propagate(inputMiniBatches[i]);
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

real_t MLPPSVC::score() {
	MLPPUtilities util;
	return util.performance(y_hat, outputSet);
}

void MLPPSVC::save(std::string fileName) {
	MLPPUtilities util;
	util.saveParameters(fileName, weights, bias);
}

MLPPSVC::MLPPSVC(std::vector<std::vector<real_t>> p_inputSet, std::vector<real_t> p_outputSet, real_t p_C) {
	inputSet = p_inputSet;
	outputSet = p_outputSet;
	n = inputSet.size();
	k = inputSet[0].size();
	C = p_C;

	y_hat.resize(n);
	weights = MLPPUtilities::weightInitialization(k);
	bias = MLPPUtilities::biasInitialization();
}

real_t MLPPSVC::Cost(std::vector<real_t> z, std::vector<real_t> y, std::vector<real_t> weights, real_t C) {
	class MLPPCost cost;
	return cost.HingeLoss(z, y, weights, C);
}

std::vector<real_t> MLPPSVC::Evaluate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.sign(alg.scalarAdd(bias, alg.mat_vec_mult(X, weights)));
}

std::vector<real_t> MLPPSVC::propagate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return alg.scalarAdd(bias, alg.mat_vec_mult(X, weights));
}

real_t MLPPSVC::Evaluate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.sign(alg.dot(weights, x) + bias);
}

real_t MLPPSVC::propagate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return alg.dot(weights, x) + bias;
}

// sign ( wTx + b )
void MLPPSVC::forwardPass() {
	MLPPActivation avn;

	z = propagate(inputSet);
	y_hat = avn.sign(z);
}
