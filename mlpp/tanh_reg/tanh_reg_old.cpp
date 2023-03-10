//
//  TanhReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "tanh_reg_old.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

MLPPTanhRegOld::MLPPTanhRegOld(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, std::string reg, real_t lambda, real_t alpha) :
		inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size()), reg(reg), lambda(lambda), alpha(alpha) {
	y_hat.resize(n);
	weights = MLPPUtilities::weightInitialization(k);
	bias = MLPPUtilities::biasInitialization();
}

std::vector<real_t> MLPPTanhRegOld::modelSetTest(std::vector<std::vector<real_t>> X) {
	return Evaluate(X);
}

real_t MLPPTanhRegOld::modelTest(std::vector<real_t> x) {
	return Evaluate(x);
}

void MLPPTanhRegOld::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, outputSet);

		std::vector<real_t> error = alg.subtraction(y_hat, outputSet);

		weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate / n, alg.mat_vec_mult(alg.transpose(inputSet), alg.hadamard_product(error, avn.tanh(z, 1)))));
		weights = regularization.regWeights(weights, lambda, alpha, reg);

		// Calculating the bias gradients
		bias -= learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.tanh(z, 1))) / n;

		forwardPass();

		// UI PORTION
		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
			MLPPUtilities::UI(weights, bias);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPTanhRegOld::SGD(real_t learning_rate, int max_epoch, bool UI) {
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
		cost_prev = Cost({ y_hat }, { outputSet[outputIndex] });

		real_t error = y_hat - outputSet[outputIndex];

		// Weight Updation
		weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate * error * (1 - y_hat * y_hat), inputSet[outputIndex]));
		weights = regularization.regWeights(weights, lambda, alpha, reg);

		// Bias updation
		bias -= learning_rate * error * (1 - y_hat * y_hat);

		y_hat = Evaluate({ inputSet[outputIndex] });

		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, Cost({ y_hat }, { outputSet[outputIndex] }));
			MLPPUtilities::UI(weights, bias);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPTanhRegOld::MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
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
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate / n, alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), alg.hadamard_product(error, avn.tanh(z, 1)))));
			weights = regularization.regWeights(weights, lambda, alpha, reg);

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.tanh(z, 1))) / n;

			forwardPass();

			y_hat = Evaluate(inputMiniBatches[i]);

			if (UI) {
				MLPPUtilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputMiniBatches[i]));
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

real_t MLPPTanhRegOld::score() {
	MLPPUtilities util;
	return util.performance(y_hat, outputSet);
}

void MLPPTanhRegOld::save(std::string fileName) {
	MLPPUtilities util;
	util.saveParameters(fileName, weights, bias);
}

real_t MLPPTanhRegOld::Cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPReg regularization;
	class MLPPCost cost;
	return cost.MSE(y_hat, y) + regularization.regTerm(weights, lambda, alpha, reg);
}

std::vector<real_t> MLPPTanhRegOld::Evaluate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.tanh(alg.scalarAdd(bias, alg.mat_vec_mult(X, weights)));
}

std::vector<real_t> MLPPTanhRegOld::propagate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	return alg.scalarAdd(bias, alg.mat_vec_mult(X, weights));
}

real_t MLPPTanhRegOld::Evaluate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.tanh(alg.dot(weights, x) + bias);
}

real_t MLPPTanhRegOld::propagate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	return alg.dot(weights, x) + bias;
}

// Tanh ( wTx + b )
void MLPPTanhRegOld::forwardPass() {
	MLPPActivation avn;

	z = propagate(inputSet);
	y_hat = avn.tanh(z);
}
