//
//  DualSVC.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "dual_svc.h"
#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>


MLPPDualSVC::MLPPDualSVC(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, real_t C, std::string kernel) :
		inputSet(inputSet), outputSet(outputSet), n(inputSet.size()), k(inputSet[0].size()), C(C), kernel(kernel) {
	y_hat.resize(n);
	bias = MLPPUtilities::biasInitialization();
	alpha = MLPPUtilities::weightInitialization(n); // One alpha for all training examples, as per the lagrangian multipliers.
	K = kernelFunction(inputSet, inputSet, kernel); // For now this is unused. When non-linear kernels are added, the K will be manipulated.
}

std::vector<real_t> MLPPDualSVC::modelSetTest(std::vector<std::vector<real_t>> X) {
	return Evaluate(X);
}

real_t MLPPDualSVC::modelTest(std::vector<real_t> x) {
	return Evaluate(x);
}

void MLPPDualSVC::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	class MLPPCost cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(alpha, inputSet, outputSet);

		alpha = alg.subtraction(alpha, alg.scalarMultiply(learning_rate, cost.dualFormSVMDeriv(alpha, inputSet, outputSet)));

		alphaProjection();

		// Calculating the bias
		real_t biasGradient = 0;
		for (int i = 0; i < alpha.size(); i++) {
			real_t sum = 0;
			if (alpha[i] < C && alpha[i] > 0) {
				for (int j = 0; j < alpha.size(); j++) {
					if (alpha[j] > 0) {
						sum += alpha[j] * outputSet[j] * alg.dot(inputSet[j], inputSet[i]); // TO DO: DON'T forget to add non-linear kernelizations.
					}
				}
			}
			biasGradient = (1 - outputSet[i] * sum) / outputSet[i];
			break;
		}
		bias -= biasGradient * learning_rate;

		forwardPass();

		// UI PORTION
		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, Cost(alpha, inputSet, outputSet));
			MLPPUtilities::UI(alpha, bias);
			std::cout << score() << std::endl; // TO DO: DELETE THIS.
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

// void MLPPDualSVC::SGD(real_t learning_rate, int max_epoch, bool UI){
//     class MLPPCost cost;
//     MLPPActivation avn;
//     MLPPLinAlg alg;
//     MLPPReg regularization;

//     real_t cost_prev = 0;
//     int epoch = 1;

//     while(true){
//         std::random_device rd;
//         std::default_random_engine generator(rd());
//         std::uniform_int_distribution<int> distribution(0, int(n - 1));
//         int outputIndex = distribution(generator);

//         cost_prev = Cost(alpha, inputSet[outputIndex], outputSet[outputIndex]);

//         // Bias updation
//         bias -= learning_rate * costDeriv;

//         y_hat = Evaluate({inputSet[outputIndex]});

//         if(UI) {
//             MLPPUtilities::CostInfo(epoch, cost_prev, Cost(alpha));
//             MLPPUtilities::UI(weights, bias);
//         }
//         epoch++;

//         if(epoch > max_epoch) { break; }
//     }
//     forwardPass();
// }

// void MLPPDualSVC::MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI){
//     class MLPPCost cost;
//     MLPPActivation avn;
//     MLPPLinAlg alg;
//     MLPPReg regularization;
//     real_t cost_prev = 0;
//     int epoch = 1;

//     // Creating the mini-batches
//     int n_mini_batch = n/mini_batch_size;
//     auto [inputMiniBatches, outputMiniBatches] = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);

//     while(true){
//         for(int i = 0; i < n_mini_batch; i++){
//             std::vector<real_t> y_hat = Evaluate(inputMiniBatches[i]);
//             std::vector<real_t> z = propagate(inputMiniBatches[i]);
//             cost_prev = Cost(z, outputMiniBatches[i], weights, C);

//             // Calculating the weight gradients
//             weights = alg.subtraction(weights, alg.scalarMultiply(learning_rate/n, alg.mat_vec_mult(alg.transpose(inputMiniBatches[i]), cost.HingeLossDeriv(z, outputMiniBatches[i], C))));
//             weights = regularization.regWeights(weights, learning_rate/n, 0, "Ridge");

//             // Calculating the bias gradients
//             bias -= learning_rate * alg.sum_elements(cost.HingeLossDeriv(y_hat, outputMiniBatches[i], C)) / n;

//             forwardPass();

//             y_hat = Evaluate(inputMiniBatches[i]);

//             if(UI) {
//                 MLPPUtilities::CostInfo(epoch, cost_prev, Cost(z, outputMiniBatches[i], weights, C));
//                 MLPPUtilities::UI(weights, bias);
//             }
//         }
//         epoch++;
//         if(epoch > max_epoch) { break; }
//     }
//     forwardPass();
// }

real_t MLPPDualSVC::score() {
	MLPPUtilities   util;
	return util.performance(y_hat, outputSet);
}

void MLPPDualSVC::save(std::string fileName) {
	MLPPUtilities   util;
	util.saveParameters(fileName, alpha, bias);
}

real_t MLPPDualSVC::Cost(std::vector<real_t> alpha, std::vector<std::vector<real_t>> X, std::vector<real_t> y) {
	class MLPPCost cost;
	return cost.dualFormSVM(alpha, X, y);
}

std::vector<real_t> MLPPDualSVC::Evaluate(std::vector<std::vector<real_t>> X) {
	MLPPActivation avn;
	return avn.sign(propagate(X));
}

std::vector<real_t> MLPPDualSVC::propagate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	std::vector<real_t> z;
	for (int i = 0; i < X.size(); i++) {
		real_t sum = 0;
		for (int j = 0; j < alpha.size(); j++) {
			if (alpha[j] != 0) {
				sum += alpha[j] * outputSet[j] * alg.dot(inputSet[j], X[i]); // TO DO: DON'T forget to add non-linear kernelizations.
			}
		}
		sum += bias;
		z.push_back(sum);
	}
	return z;
}

real_t MLPPDualSVC::Evaluate(std::vector<real_t> x) {
	MLPPActivation avn;
	return avn.sign(propagate(x));
}

real_t MLPPDualSVC::propagate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	real_t z = 0;
	for (int j = 0; j < alpha.size(); j++) {
		if (alpha[j] != 0) {
			z += alpha[j] * outputSet[j] * alg.dot(inputSet[j], x); // TO DO: DON'T forget to add non-linear kernelizations.
		}
	}
	z += bias;
	return z;
}

void MLPPDualSVC::forwardPass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	z = propagate(inputSet);
	y_hat = avn.sign(z);
}

void MLPPDualSVC::alphaProjection() {
	for (int i = 0; i < alpha.size(); i++) {
		if (alpha[i] > C) {
			alpha[i] = C;
		} else if (alpha[i] < 0) {
			alpha[i] = 0;
		}
	}
}

real_t MLPPDualSVC::kernelFunction(std::vector<real_t> u, std::vector<real_t> v, std::string kernel) {
	MLPPLinAlg alg;
	if (kernel == "Linear") {
		return alg.dot(u, v);
	} // warning: non-void function does not return a value in all control paths [-Wreturn-type]
}

std::vector<std::vector<real_t>> MLPPDualSVC::kernelFunction(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B, std::string kernel) {
	MLPPLinAlg alg;
	if (kernel == "Linear") {
		return alg.matmult(inputSet, alg.transpose(inputSet));
	} // warning: non-void function does not return a value in all control paths [-Wreturn-type]
}
