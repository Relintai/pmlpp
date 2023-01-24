//
//  CLogLogReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "c_log_log_reg.h"
#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

MLPPCLogLogReg::MLPPCLogLogReg(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, std::string reg, double lambda, double alpha) :
		inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size()), reg(reg), lambda(lambda), alpha(alpha) {
	y_hat.resize(n);
	weights = Utilities::weightInitialization(k);
	bias = Utilities::biasInitialization();
}

std::vector<double> MLPPCLogLogReg::modelSetTest(std::vector<std::vector<double>> X) {
	return Evaluate(X);
}

double MLPPCLogLogReg::modelTest(std::vector<double> x) {
	return Evaluate(x);
}

void MLPPCLogLogReg::gradientDescent(double learning_rate, int max_epoch, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	double cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, outputSet);

		std::vector<double> error = alg.subtraction(y_hat, outputSet);

		// Calculating the weight gradients
		weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate / n, alg.mat_vec_mult(alg.transpose(inputSet), alg.hadamard_product(error, avn.cloglog(z, 1)))));
		weights = regularization.regWeights(weights, lambda, alpha, reg);

		// Calculating the bias gradients
		bias -= learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.cloglog(z, 1))) / n;

		forwardPass();

		if (UI) {
			Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
			Utilities::UI(weights, bias);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPCLogLogReg::MLE(double learning_rate, int max_epoch, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	double cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, outputSet);

		std::vector<double> error = alg.subtraction(y_hat, outputSet);

		weights = alg.addition(weights, alg.scalarMultiply(learning_rate / n, alg.mat_vec_mult(alg.transpose(inputSet), alg.hadamard_product(error, avn.cloglog(z, 1)))));
		weights = regularization.regWeights(weights, lambda, alpha, reg);

		// Calculating the bias gradients
		bias += learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.cloglog(z, 1))) / n;
		forwardPass();

		if (UI) {
			Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
			Utilities::UI(weights, bias);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPCLogLogReg::SGD(double learning_rate, int max_epoch, bool UI) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	double cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(n - 1));
		int outputIndex = distribution(generator);

		double y_hat = Evaluate(inputSet[outputIndex]);
		double z = propagate(inputSet[outputIndex]);
		cost_prev = Cost({ y_hat }, { outputSet[outputIndex] });

		double error = y_hat - outputSet[outputIndex];

		// Weight Updation
		weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate * error * exp(z - exp(z)), inputSet[outputIndex]));
		weights = regularization.regWeights(weights, lambda, alpha, reg);

		// Bias updation
		bias -= learning_rate * error * exp(z - exp(z));

		y_hat = Evaluate({ inputSet[outputIndex] });

		if (UI) {
			Utilities::CostInfo(epoch, cost_prev, Cost({ y_hat }, { outputSet[outputIndex] }));
			Utilities::UI(weights, bias);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPCLogLogReg::MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	double cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto [inputMiniBatches, outputMiniBatches] = Utilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<double> y_hat = Evaluate(inputMiniBatches[i]);
			std::vector<double> z = propagate(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			std::vector<double> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate / n, alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), alg.hadamard_product(error, avn.cloglog(z, 1)))));
			weights = regularization.regWeights(weights, lambda, alpha, reg);

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elements(alg.hadamard_product(error, avn.cloglog(z, 1))) / n;

			forwardPass();

			y_hat = Evaluate(inputMiniBatches[i]);

			if (UI) {
				Utilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputMiniBatches[i]));
				Utilities::UI(weights, bias);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

double MLPPCLogLogReg::score() {
	Utilities util;
	return util.performance(y_hat, outputSet);
}

double MLPPCLogLogReg::Cost(std::vector<double> y_hat, std::vector<double> y) {
	MLPPReg regularization;
	class MLPPCost cost;
	return cost.MSE(y_hat, y) + regularization.regTerm(weights, lambda, alpha, reg);
}

std::vector<double> MLPPCLogLogReg::Evaluate(std::vector<std::vector<double>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.cloglog(alg.scalarAdd(bias, alg.mat_vec_mult(X, weights)));
}

std::vector<double> MLPPCLogLogReg::propagate(std::vector<std::vector<double>> X) {
	MLPPLinAlg alg;
	return alg.scalarAdd(bias, alg.mat_vec_mult(X, weights));
}

double MLPPCLogLogReg::Evaluate(std::vector<double> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.cloglog(alg.dot(weights, x) + bias);
}

double MLPPCLogLogReg::propagate(std::vector<double> x) {
	MLPPLinAlg alg;
	return alg.dot(weights, x) + bias;
}

// cloglog ( wTx + b )
void MLPPCLogLogReg::forwardPass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	z = propagate(inputSet);
	y_hat = avn.cloglog(z);
}
