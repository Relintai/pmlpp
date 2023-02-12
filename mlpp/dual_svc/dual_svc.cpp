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

std::vector<real_t> MLPPDualSVC::model_set_test(std::vector<std::vector<real_t>> X) {
	return evaluatem(X);
}

real_t MLPPDualSVC::model_test(std::vector<real_t> x) {
	return evaluatev(x);
}

void MLPPDualSVC::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_alpha, _input_set, _output_set);

		_alpha = alg.subtraction(_alpha, alg.scalarMultiply(learning_rate, mlpp_cost.dualFormSVMDeriv(_alpha, _input_set, _output_set)));

		alpha_projection();

		// Calculating the bias
		real_t biasGradient = 0;
		for (uint32_t i = 0; i < _alpha.size(); i++) {
			real_t sum = 0;
			if (_alpha[i] < _C && _alpha[i] > 0) {
				for (uint32_t j = 0; j < _alpha.size(); j++) {
					if (_alpha[j] > 0) {
						sum += _alpha[j] * _output_set[j] * alg.dot(_input_set[j], _input_set[i]); // TO DO: DON'T forget to add non-linear kernelizations.
					}
				}
			}
			biasGradient = (1 - _output_set[i] * sum) / _output_set[i];
			break;
		}

		_bias -= biasGradient * learning_rate;

		forward_pass();

		// UI PORTION
		if (ui) {
			MLPPUtilities::CostInfo(epoch, cost_prev, cost(_alpha, _input_set, _output_set));
			MLPPUtilities::UI(_alpha, _bias);
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

//         cost_prev = Cost(alpha, _input_set[outputIndex], _output_set[outputIndex]);

//         // Bias updation
//         bias -= learning_rate * costDeriv;

//         y_hat = Evaluate({_input_set[outputIndex]});

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
//     auto [inputMiniBatches, outputMiniBatches] = MLPPUtilities::createMiniBatches(_input_set, _output_set, n_mini_batch);

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
	MLPPUtilities util;
	return util.performance(_y_hat, _output_set);
}

MLPPDualSVC::MLPPDualSVC(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, real_t p_C, std::string p_kernel) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = p_input_set.size();
	_k = p_input_set[0].size();
	_C = p_C;
	_kernel = p_kernel;

	_y_hat.resize(_n);
	_bias = MLPPUtilities::biasInitialization();
	_alpha = MLPPUtilities::weightInitialization(_n); // One alpha for all training examples, as per the lagrangian multipliers.
	_K = kernel_functionm(_input_set, _input_set, _kernel); // For now this is unused. When non-linear kernels are added, the K will be manipulated.
}

MLPPDualSVC::MLPPDualSVC() {
}
MLPPDualSVC::~MLPPDualSVC() {
}

void MLPPDualSVC::save(std::string file_name) {
	MLPPUtilities util;

	util.saveParameters(file_name, _alpha, _bias);
}

real_t MLPPDualSVC::cost(std::vector<real_t> alpha, std::vector<std::vector<real_t>> X, std::vector<real_t> y) {
	class MLPPCost cost;
	return cost.dualFormSVM(alpha, X, y);
}

real_t MLPPDualSVC::evaluatev(std::vector<real_t> x) {
	MLPPActivation avn;
	return avn.sign(propagatev(x));
}

real_t MLPPDualSVC::propagatev(std::vector<real_t> x) {
	MLPPLinAlg alg;
	real_t z = 0;
	for (uint32_t j = 0; j < _alpha.size(); j++) {
		if (_alpha[j] != 0) {
			z += _alpha[j] * _output_set[j] * alg.dot(_input_set[j], x); // TO DO: DON'T forget to add non-linear kernelizations.
		}
	}
	z += _bias;
	return z;
}

std::vector<real_t> MLPPDualSVC::evaluatem(std::vector<std::vector<real_t>> X) {
	MLPPActivation avn;
	return avn.sign(propagatem(X));
}

std::vector<real_t> MLPPDualSVC::propagatem(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	std::vector<real_t> z;
	for (uint32_t i = 0; i < X.size(); i++) {
		real_t sum = 0;
		for (uint32_t j = 0; j < _alpha.size(); j++) {
			if (_alpha[j] != 0) {
				sum += _alpha[j] * _output_set[j] * alg.dot(_input_set[j], X[i]); // TO DO: DON'T forget to add non-linear kernelizations.
			}
		}
		sum += _bias;
		z.push_back(sum);
	}
	return z;
}

void MLPPDualSVC::forward_pass() {
	MLPPActivation avn;

	_z = propagatem(_input_set);
	_y_hat = avn.sign(_z);
}

void MLPPDualSVC::alpha_projection() {
	for (uint32_t i = 0; i < _alpha.size(); i++) {
		if (_alpha[i] > _C) {
			_alpha[i] = _C;
		} else if (_alpha[i] < 0) {
			_alpha[i] = 0;
		}
	}
}

real_t MLPPDualSVC::kernel_functionv(std::vector<real_t> u, std::vector<real_t> v, std::string kernel) {
	MLPPLinAlg alg;

	if (kernel == "Linear") {
		return alg.dot(u, v);
	}

	return 0;
}

std::vector<std::vector<real_t>> MLPPDualSVC::kernel_functionm(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B, std::string kernel) {
	MLPPLinAlg alg;
	if (kernel == "Linear") {
		return alg.matmult(_input_set, alg.transpose(_input_set));
	}

	return std::vector<std::vector<real_t>>();
}
