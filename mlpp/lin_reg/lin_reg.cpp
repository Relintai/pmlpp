//
//  LinReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "lin_reg.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../stat/stat.h"
#include "../utilities/utilities.h"

#include <cmath>
#include <iostream>
#include <random>

MLPPLinReg::MLPPLinReg(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, std::string reg, real_t lambda, real_t alpha) :
		inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size()), reg(reg), lambda(lambda), alpha(alpha) {
	y_hat.resize(n);

	weights = MLPPUtilities::weightInitialization(k);
	bias = MLPPUtilities::biasInitialization();
}

std::vector<real_t> MLPPLinReg::modelSetTest(std::vector<std::vector<real_t>> X) {
	return Evaluate(X);
}

real_t MLPPLinReg::modelTest(std::vector<real_t> x) {
	return Evaluate(x);
}

void MLPPLinReg::NewtonRaphson(real_t learning_rate, int max_epoch, bool UI) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();
	while (true) {
		cost_prev = Cost(y_hat, outputSet);

		std::vector<real_t> error = alg.subtraction(y_hat, outputSet);

		// Calculating the weight gradients (2nd derivative)
		std::vector<real_t> first_derivative = alg.mat_vec_mult(alg.transpose(inputSet), error);
		std::vector<std::vector<real_t>> second_derivative = alg.matmult(alg.transpose(inputSet), inputSet);
		weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate / n, alg.mat_vec_mult(alg.transpose(alg.inverse(second_derivative)), first_derivative)));
		weights = regularization.regWeights(weights, lambda, alpha, reg);

		// Calculating the bias gradients (2nd derivative)
		bias -= learning_rate * alg.sum_elements(error) / n; // We keep this the same. The 2nd derivative is just [1].
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

void MLPPLinReg::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, outputSet);

		std::vector<real_t> error = alg.subtraction(y_hat, outputSet);

		// Calculating the weight gradients
		weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate / n, alg.mat_vec_mult(alg.transpose(inputSet), error)));
		weights = regularization.regWeights(weights, lambda, alpha, reg);

		// Calculating the bias gradients
		bias -= learning_rate * alg.sum_elements(error) / n;
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

void MLPPLinReg::SGD(real_t learning_rate, int max_epoch, bool UI) {
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

		// Weight updation
		weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate * error, inputSet[outputIndex]));
		weights = regularization.regWeights(weights, lambda, alpha, reg);

		// Bias updation
		bias -= learning_rate * error;

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

void MLPPLinReg::MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
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
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate / outputMiniBatches[i].size(), alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), error)));
			weights = regularization.regWeights(weights, lambda, alpha, reg);

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size();
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

void MLPPLinReg::Momentum(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool UI) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto [inputMiniBatches, outputMiniBatches] = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

	// Initializing necessary components for Momentum.
	std::vector<real_t> v = alg.zerovec(weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = Evaluate(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / outputMiniBatches[i].size(), alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), error));
			std::vector<real_t> RegDerivTerm = regularization.regDerivTerm(weights, lambda, alpha, reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, RegDerivTerm); // Weight_grad_final

			v = alg.addition(alg.scalarMultiply(gamma, v), alg.scalarMultiply(learning_rate, weight_grad));

			weights = alg.subtraction(weights, v);

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size(); // As normal
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

void MLPPLinReg::NAG(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool UI) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto [inputMiniBatches, outputMiniBatches] = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

	// Initializing necessary components for Momentum.
	std::vector<real_t> v = alg.zerovec(weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			weights = alg.subtraction(weights, alg.scalarMultiply(gamma, v)); // "Aposterori" calculation

			std::vector<real_t> y_hat = Evaluate(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / outputMiniBatches[i].size(), alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), error));
			std::vector<real_t> RegDerivTerm = regularization.regDerivTerm(weights, lambda, alpha, reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, RegDerivTerm); // Weight_grad_final

			v = alg.addition(alg.scalarMultiply(gamma, v), alg.scalarMultiply(learning_rate, weight_grad));

			weights = alg.subtraction(weights, v);

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size(); // As normal
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

void MLPPLinReg::Adagrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t e, bool UI) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto [inputMiniBatches, outputMiniBatches] = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

	// Initializing necessary components for Adagrad.
	std::vector<real_t> v = alg.zerovec(weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = Evaluate(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / outputMiniBatches[i].size(), alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), error));
			std::vector<real_t> RegDerivTerm = regularization.regDerivTerm(weights, lambda, alpha, reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, RegDerivTerm); // Weight_grad_final

			v = alg.hadamard_product(weight_grad, weight_grad);

			weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate, alg.elementWiseDivision(weight_grad, alg.sqrt(alg.scalarAdd(e, v)))));

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size(); // As normal
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

void MLPPLinReg::Adadelta(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t e, bool UI) {
	// Adagrad upgrade. Momentum is applied.
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto [inputMiniBatches, outputMiniBatches] = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

	// Initializing necessary components for Adagrad.
	std::vector<real_t> v = alg.zerovec(weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = Evaluate(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / outputMiniBatches[i].size(), alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), error));
			std::vector<real_t> RegDerivTerm = regularization.regDerivTerm(weights, lambda, alpha, reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, RegDerivTerm); // Weight_grad_final

			v = alg.addition(alg.scalarMultiply(b1, v), alg.scalarMultiply(1 - b1, alg.hadamard_product(weight_grad, weight_grad)));

			weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate, alg.elementWiseDivision(weight_grad, alg.sqrt(alg.scalarAdd(e, v)))));

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size(); // As normal
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

void MLPPLinReg::Adam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto [inputMiniBatches, outputMiniBatches] = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

	// Initializing necessary components for Adam.
	std::vector<real_t> m = alg.zerovec(weights.size());

	std::vector<real_t> v = alg.zerovec(weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = Evaluate(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / outputMiniBatches[i].size(), alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), error));
			std::vector<real_t> RegDerivTerm = regularization.regDerivTerm(weights, lambda, alpha, reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, RegDerivTerm); // Weight_grad_final

			m = alg.addition(alg.scalarMultiply(b1, m), alg.scalarMultiply(1 - b1, weight_grad));
			v = alg.addition(alg.scalarMultiply(b2, v), alg.scalarMultiply(1 - b2, alg.exponentiate(weight_grad, 2)));

			std::vector<real_t> m_hat = alg.scalarMultiply(1 / (1 - pow(b1, epoch)), m);
			std::vector<real_t> v_hat = alg.scalarMultiply(1 / (1 - pow(b2, epoch)), v);

			weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate, alg.elementWiseDivision(m_hat, alg.scalarAdd(e, alg.sqrt(v_hat)))));

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size(); // As normal
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

void MLPPLinReg::Adamax(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto [inputMiniBatches, outputMiniBatches] = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

	std::vector<real_t> m = alg.zerovec(weights.size());

	std::vector<real_t> u = alg.zerovec(weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = Evaluate(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / outputMiniBatches[i].size(), alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), error));
			std::vector<real_t> RegDerivTerm = regularization.regDerivTerm(weights, lambda, alpha, reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, RegDerivTerm); // Weight_grad_final

			m = alg.addition(alg.scalarMultiply(b1, m), alg.scalarMultiply(1 - b1, weight_grad));
			u = alg.max(alg.scalarMultiply(b2, u), alg.abs(weight_grad));

			std::vector<real_t> m_hat = alg.scalarMultiply(1 / (1 - pow(b1, epoch)), m);

			weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate, alg.elementWiseDivision(m_hat, u)));

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size(); // As normal
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

void MLPPLinReg::Nadam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto [inputMiniBatches, outputMiniBatches] = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

	// Initializing necessary components for Adam.
	std::vector<real_t> m = alg.zerovec(weights.size());
	std::vector<real_t> v = alg.zerovec(weights.size());
	std::vector<real_t> m_final = alg.zerovec(weights.size());
	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = Evaluate(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight gradients
			std::vector<real_t> gradient = alg.scalarMultiply(1 / outputMiniBatches[i].size(), alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), error));
			std::vector<real_t> RegDerivTerm = regularization.regDerivTerm(weights, lambda, alpha, reg);
			std::vector<real_t> weight_grad = alg.addition(gradient, RegDerivTerm); // Weight_grad_final

			m = alg.addition(alg.scalarMultiply(b1, m), alg.scalarMultiply(1 - b1, weight_grad));
			v = alg.addition(alg.scalarMultiply(b2, v), alg.scalarMultiply(1 - b2, alg.exponentiate(weight_grad, 2)));
			m_final = alg.addition(alg.scalarMultiply(b1, m), alg.scalarMultiply((1 - b1) / (1 - pow(b1, epoch)), weight_grad));

			std::vector<real_t> m_hat = alg.scalarMultiply(1 / (1 - pow(b1, epoch)), m);
			std::vector<real_t> v_hat = alg.scalarMultiply(1 / (1 - pow(b2, epoch)), v);

			weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate, alg.elementWiseDivision(m_final, alg.scalarAdd(e, alg.sqrt(v_hat)))));

			// Calculating the bias gradients
			bias -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size(); // As normal
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

void MLPPLinReg::normalEquation() {
	MLPPLinAlg alg;
	MLPPStat stat;
	std::vector<real_t> x_means;
	std::vector<std::vector<real_t>> inputSetT = alg.transpose(inputSet);

	x_means.resize(inputSetT.size());
	for (int i = 0; i < inputSetT.size(); i++) {
		x_means[i] = (stat.mean(inputSetT[i]));
	}

	//try {
	std::vector<real_t> temp;
	temp.resize(k);
	temp = alg.mat_vec_mult(alg.inverse(alg.matmult(alg.transpose(inputSet), inputSet)), alg.mat_vec_mult(alg.transpose(inputSet), outputSet));
	if (std::isnan(temp[0])) {
		//throw 99;
		//TODO ERR_FAIL_COND
		std::cout << "ERR: Resulting matrix was noninvertible/degenerate, and so the normal equation could not be performed. Try utilizing gradient descent." << std::endl;
		return;
	} else {
		if (reg == "Ridge") {
			weights = alg.mat_vec_mult(alg.inverse(alg.addition(alg.matmult(alg.transpose(inputSet), inputSet), alg.scalarMultiply(lambda, alg.identity(k)))), alg.mat_vec_mult(alg.transpose(inputSet), outputSet));
		} else {
			weights = alg.mat_vec_mult(alg.inverse(alg.matmult(alg.transpose(inputSet), inputSet)), alg.mat_vec_mult(alg.transpose(inputSet), outputSet));
		}

		bias = stat.mean(outputSet) - alg.dot(weights, x_means);

		forwardPass();
	}
	//} catch (int err_num) {
	//	std::cout << "ERR " << err_num << ": Resulting matrix was noninvertible/degenerate, and so the normal equation could not be performed. Try utilizing gradient descent." << std::endl;
	//}
}

real_t MLPPLinReg::score() {
	MLPPUtilities util;
	return util.performance(y_hat, outputSet);
}

void MLPPLinReg::save(std::string fileName) {
	MLPPUtilities util;
	util.saveParameters(fileName, weights, bias);
}

real_t MLPPLinReg::Cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPReg regularization;
	class MLPPCost cost;
	return cost.MSE(y_hat, y) + regularization.regTerm(weights, lambda, alpha, reg);
}

std::vector<real_t> MLPPLinReg::Evaluate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	return alg.scalarAdd(bias, alg.mat_vec_mult(X, weights));
}

real_t MLPPLinReg::Evaluate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	return alg.dot(weights, x) + bias;
}

// wTx + b
void MLPPLinReg::forwardPass() {
	y_hat = Evaluate(inputSet);
}
