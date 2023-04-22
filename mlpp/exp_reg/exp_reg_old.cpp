//
//  ExpReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "exp_reg_old.h"

#include "../cost/cost_old.h"
#include "../lin_alg/lin_alg_old.h"
#include "../regularization/reg_old.h"
#include "../stat/stat_old.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

MLPPExpRegOld::MLPPExpRegOld(std::vector<std::vector<real_t>> p_inputSet, std::vector<real_t> p_outputSet, std::string p_reg, real_t p_lambda, real_t p_alpha) {
	inputSet = p_inputSet;
	outputSet = p_outputSet;
	n = p_inputSet.size();
	k = p_inputSet[0].size();
	reg = p_reg;
	lambda = p_lambda;
	alpha = p_alpha;

	y_hat.resize(n);
	weights = MLPPUtilities::weightInitialization(k);
	initial = MLPPUtilities::weightInitialization(k);
	bias = MLPPUtilities::biasInitialization();
}

std::vector<real_t> MLPPExpRegOld::modelSetTest(std::vector<std::vector<real_t>> X) {
	return Evaluate(X);
}

real_t MLPPExpRegOld::modelTest(std::vector<real_t> x) {
	return Evaluate(x);
}

void MLPPExpRegOld::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	MLPPLinAlgOld alg;
	MLPPRegOld regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, outputSet);

		std::vector<real_t> error = alg.subtraction(y_hat, outputSet);

		for (int i = 0; i < k; i++) {
			// Calculating the weight gradient
			real_t sum = 0;
			for (int j = 0; j < n; j++) {
				sum += error[j] * inputSet[j][i] * std::pow(weights[i], inputSet[j][i] - 1);
			}
			real_t w_gradient = sum / n;

			// Calculating the initial gradient
			real_t sum2 = 0;
			for (int j = 0; j < n; j++) {
				sum2 += error[j] * std::pow(weights[i], inputSet[j][i]);
			}

			real_t i_gradient = sum2 / n;

			// Weight/initial updation
			weights[i] -= learning_rate * w_gradient;
			initial[i] -= learning_rate * i_gradient;
		}
		weights = regularization.regWeights(weights, lambda, alpha, reg);

		// Calculating the bias gradient
		real_t sum = 0;
		for (int j = 0; j < n; j++) {
			sum += (y_hat[j] - outputSet[j]);
		}
		real_t b_gradient = sum / n;

		// bias updation
		bias -= learning_rate * b_gradient;
		forwardPass();

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

void MLPPExpRegOld::SGD(real_t learning_rate, int max_epoch, bool UI) {
	MLPPRegOld regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	while (true) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(n - 1));
		int outputIndex = distribution(generator);

		real_t y_hat = Evaluate(inputSet[outputIndex]);
		cost_prev = Cost({ y_hat }, { outputSet[outputIndex] });

		for (int i = 0; i < k; i++) {
			// Calculating the weight gradients

			real_t w_gradient = (y_hat - outputSet[outputIndex]) * inputSet[outputIndex][i] * std::pow(weights[i], inputSet[outputIndex][i] - 1);
			real_t i_gradient = (y_hat - outputSet[outputIndex]) * std::pow(weights[i], inputSet[outputIndex][i]);

			// Weight/initial updation
			weights[i] -= learning_rate * w_gradient;
			initial[i] -= learning_rate * i_gradient;
		}
		weights = regularization.regWeights(weights, lambda, alpha, reg);

		// Calculating the bias gradients
		real_t b_gradient = (y_hat - outputSet[outputIndex]);

		// Bias updation
		bias -= learning_rate * b_gradient;
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

void MLPPExpRegOld::MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	MLPPLinAlgOld alg;
	MLPPRegOld regularization;
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
			cost_prev = Cost(y_hat, outputMiniBatches[i]);
			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			for (int j = 0; j < k; j++) {
				// Calculating the weight gradient
				real_t sum = 0;
				for (uint32_t k = 0; k < outputMiniBatches[i].size(); k++) {
					sum += error[k] * inputMiniBatches[i][k][j] * std::pow(weights[j], inputMiniBatches[i][k][j] - 1);
				}
				real_t w_gradient = sum / outputMiniBatches[i].size();

				// Calculating the initial gradient
				real_t sum2 = 0;
				for (uint32_t k = 0; k < outputMiniBatches[i].size(); k++) {
					sum2 += error[k] * std::pow(weights[j], inputMiniBatches[i][k][j]);
				}

				real_t i_gradient = sum2 / outputMiniBatches[i].size();

				// Weight/initial updation
				weights[j] -= learning_rate * w_gradient;
				initial[j] -= learning_rate * i_gradient;
			}
			weights = regularization.regWeights(weights, lambda, alpha, reg);

			// Calculating the bias gradient
			//real_t sum = 0;
			//for (uint32_t j = 0; j < outputMiniBatches[i].size(); j++) {
			//	sum += (y_hat[j] - outputMiniBatches[i][j]);
			//}

			//real_t b_gradient = sum / outputMiniBatches[i].size();
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

real_t MLPPExpRegOld::score() {
	MLPPUtilities util;
	return util.performance(y_hat, outputSet);
}

void MLPPExpRegOld::save(std::string fileName) {
	MLPPUtilities util;
	util.saveParameters(fileName, weights, initial, bias);
}

real_t MLPPExpRegOld::Cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPRegOld regularization;
	class MLPPCostOld cost;
	return cost.MSE(y_hat, y) + regularization.regTerm(weights, lambda, alpha, reg);
}

std::vector<real_t> MLPPExpRegOld::Evaluate(std::vector<std::vector<real_t>> X) {
	std::vector<real_t> y_hat;
	y_hat.resize(X.size());
	for (uint32_t i = 0; i < X.size(); i++) {
		y_hat[i] = 0;
		for (uint32_t j = 0; j < X[i].size(); j++) {
			y_hat[i] += initial[j] * std::pow(weights[j], X[i][j]);
		}
		y_hat[i] += bias;
	}
	return y_hat;
}

real_t MLPPExpRegOld::Evaluate(std::vector<real_t> x) {
	real_t y_hat = 0;
	for (uint32_t i = 0; i < x.size(); i++) {
		y_hat += initial[i] * std::pow(weights[i], x[i]);
	}

	return y_hat + bias;
}

// a * w^x + b
void MLPPExpRegOld::forwardPass() {
	y_hat = Evaluate(inputSet);
}
