//
//  SoftmaxReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "softmax_reg_old.h"
#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

MLPPSoftmaxRegOld::MLPPSoftmaxRegOld(std::vector<std::vector<real_t>> inputSet, std::vector<std::vector<real_t>> outputSet, std::string reg, real_t lambda, real_t alpha) :
		inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size()), n_class(outputSet[0].size()), reg(reg), lambda(lambda), alpha(alpha) {
	y_hat.resize(n);
	weights = MLPPUtilities::weightInitialization(k, n_class);
	bias = MLPPUtilities::biasInitialization(n_class);
}

std::vector<real_t> MLPPSoftmaxRegOld::modelTest(std::vector<real_t> x) {
	return Evaluate(x);
}

std::vector<std::vector<real_t>> MLPPSoftmaxRegOld::modelSetTest(std::vector<std::vector<real_t>> X) {
	return Evaluate(X);
}

void MLPPSoftmaxRegOld::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, outputSet);
		std::vector<std::vector<real_t>> error = alg.subtraction(y_hat, outputSet);

		//Calculating the weight gradients
		std::vector<std::vector<real_t>> w_gradient = alg.matmult(alg.transpose(inputSet), error);

		//Weight updation
		weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate, w_gradient));
		weights = regularization.regWeights(weights, lambda, alpha, reg);

		// Calculating the bias gradients
		//real_t b_gradient = alg.sum_elements(error);

		// Bias Updation
		bias = alg.subtractMatrixRows(bias, alg.scalarMultiply(learning_rate, error));

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

void MLPPSoftmaxRegOld::SGD(real_t learning_rate, int max_epoch, bool UI) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	while (true) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(n - 1));
		real_t outputIndex = distribution(generator);

		std::vector<real_t> y_hat = Evaluate(inputSet[outputIndex]);
		cost_prev = Cost({ y_hat }, { outputSet[outputIndex] });

		// Calculating the weight gradients
		std::vector<std::vector<real_t>> w_gradient = alg.outerProduct(inputSet[outputIndex], alg.subtraction(y_hat, outputSet[outputIndex]));

		// Weight Updation
		weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate, w_gradient));
		weights = regularization.regWeights(weights, lambda, alpha, reg);

		// Calculating the bias gradients
		std::vector<real_t> b_gradient = alg.subtraction(y_hat, outputSet[outputIndex]);

		// Bias updation
		bias = alg.subtraction(bias, alg.scalarMultiply(learning_rate, b_gradient));

		//y_hat = Evaluate({ inputSet[outputIndex] });
		y_hat = Evaluate(inputSet[outputIndex]);

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

void MLPPSoftmaxRegOld::MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
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
			std::vector<std::vector<real_t>> y_hat = Evaluate(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			std::vector<std::vector<real_t>> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			std::vector<std::vector<real_t>> w_gradient = alg.matmult(alg.transpose(inputMiniBatches[i]), error);

			//Weight updation
			weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate, w_gradient));
			weights = regularization.regWeights(weights, lambda, alpha, reg);

			// Calculating the bias gradients
			bias = alg.subtractMatrixRows(bias, alg.scalarMultiply(learning_rate, error));
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

real_t MLPPSoftmaxRegOld::score() {
	MLPPUtilities util;
	return util.performance(y_hat, outputSet);
}

void MLPPSoftmaxRegOld::save(std::string fileName) {
	MLPPUtilities util;
	util.saveParameters(fileName, weights, bias);
}

real_t MLPPSoftmaxRegOld::Cost(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPReg regularization;
	class MLPPCost cost;
	return cost.CrossEntropy(y_hat, y) + regularization.regTerm(weights, lambda, alpha, reg);
}

std::vector<real_t> MLPPSoftmaxRegOld::Evaluate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.softmax(alg.addition(bias, alg.mat_vec_mult(alg.transpose(weights), x)));
}

std::vector<std::vector<real_t>> MLPPSoftmaxRegOld::Evaluate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	return avn.softmax(alg.mat_vec_add(alg.matmult(X, weights), bias));
}

// softmax ( wTx + b )
void MLPPSoftmaxRegOld::forwardPass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	y_hat = avn.softmax(alg.mat_vec_add(alg.matmult(inputSet, weights), bias));
}
