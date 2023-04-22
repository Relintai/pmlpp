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

#include <random>

Ref<MLPPVector> MLPPDualSVC::model_set_test(const Ref<MLPPMatrix> &X) {
	return evaluatem(X);
}

real_t MLPPDualSVC::model_test(const Ref<MLPPVector> &x) {
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

	Ref<MLPPVector> input_set_i_row_tmp;
	input_set_i_row_tmp.instance();
	input_set_i_row_tmp->resize(_input_set->size().x);

	Ref<MLPPVector> input_set_j_row_tmp;
	input_set_j_row_tmp.instance();
	input_set_j_row_tmp->resize(_input_set->size().x);

	while (true) {
		cost_prev = cost(_alpha, _input_set, _output_set);

		_alpha = alg.subtractionnv(_alpha, alg.scalar_multiplynv(learning_rate, mlpp_cost.dual_form_svm_deriv(_alpha, _input_set, _output_set)));

		alpha_projection();

		// Calculating the bias
		real_t biasGradient = 0;
		for (int i = 0; i < _alpha->size(); i++) {
			real_t sum = 0;
			if (_alpha->get_element(i) < _C && _alpha->get_element(i) > 0) {
				for (int j = 0; j < _alpha->size(); j++) {
					if (_alpha->get_element(j) > 0) {
						_input_set->get_row_into_mlpp_vector(i, input_set_i_row_tmp);
						_input_set->get_row_into_mlpp_vector(j, input_set_j_row_tmp);

						sum += _alpha->get_element(j) * _output_set->get_element(j) * alg.dotnv(input_set_j_row_tmp, input_set_i_row_tmp); // TO DO: DON'T forget to add non-linear kernelizations.
					}
				}
			}

			biasGradient = (1 - _output_set->get_element(i) * sum) / _output_set->get_element(i);

			break;
		}

		_bias -= biasGradient * learning_rate;

		forward_pass();

		// UI PORTION
		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_alpha, _input_set, _output_set));
			MLPPUtilities::print_ui_vb(_alpha, _bias);
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
	return util.performance_vec(_y_hat, _output_set);
}

void MLPPDualSVC::save(const String &file_name) {
	MLPPUtilities util;

	//util.saveParameters(file_name, _alpha, _bias);
}

MLPPDualSVC::MLPPDualSVC(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPMatrix> &p_output_set, real_t p_C, KernelMethod p_kernel) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = p_input_set->size().y;
	_k = p_input_set->size().x;
	_C = p_C;
	_kernel = p_kernel;

	_y_hat.instance();
	_y_hat->resize(_n);

	MLPPUtilities utils;

	_bias = utils.bias_initializationr();

	_alpha.instance();
	_alpha->resize(_n);

	utils.weight_initializationv(_alpha); // One alpha for all training examples, as per the lagrangian multipliers.
	_K = kernel_functionm(_input_set, _input_set, _kernel); // For now this is unused. When non-linear kernels are added, the K will be manipulated.
}

MLPPDualSVC::MLPPDualSVC() {
}
MLPPDualSVC::~MLPPDualSVC() {
}

real_t MLPPDualSVC::cost(const Ref<MLPPVector> &alpha, const Ref<MLPPMatrix> &X, const Ref<MLPPVector> &y) {
	class MLPPCost cost;

	return cost.dual_form_svm(alpha, X, y);
}

real_t MLPPDualSVC::evaluatev(const Ref<MLPPVector> &x) {
	MLPPActivation avn;
	return avn.sign_normr(propagatev(x));
}

real_t MLPPDualSVC::propagatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	real_t z = 0;

	Ref<MLPPVector> input_set_row_tmp;
	input_set_row_tmp.instance();
	input_set_row_tmp->resize(_input_set->size().x);

	for (int j = 0; j < _alpha->size(); j++) {
		if (_alpha->get_element(j) != 0) {
			_input_set->get_row_into_mlpp_vector(j, input_set_row_tmp);
			z += _alpha->get_element(j) * _output_set->get_element(j) * alg.dotnv(input_set_row_tmp, x); // TO DO: DON'T forget to add non-linear kernelizations.
		}
	}
	z += _bias;
	return z;
}

Ref<MLPPVector> MLPPDualSVC::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPActivation avn;

	return avn.sign_normv(propagatem(X));
}

Ref<MLPPVector> MLPPDualSVC::propagatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;
	Ref<MLPPVector> z;
	z.instance();
	z->resize(X->size().y);

	Ref<MLPPVector> input_set_row_tmp;
	input_set_row_tmp.instance();
	input_set_row_tmp->resize(_input_set->size().x);

	Ref<MLPPVector> x_row_tmp;
	x_row_tmp.instance();
	x_row_tmp->resize(X->size().x);

	for (int i = 0; i < X->size().y; i++) {
		real_t sum = 0;

		for (int j = 0; j < _alpha->size(); j++) {
			if (_alpha->get_element(j) != 0) {
				_input_set->get_row_into_mlpp_vector(j, input_set_row_tmp);
				X->get_row_into_mlpp_vector(i, x_row_tmp);

				sum += _alpha->get_element(j) * _output_set->get_element(j) * alg.dotnv(input_set_row_tmp, x_row_tmp); // TO DO: DON'T forget to add non-linear kernelizations.
			}
		}

		sum += _bias;

		z->set_element(i, sum);
	}
	return z;
}

void MLPPDualSVC::forward_pass() {
	MLPPActivation avn;

	_z = propagatem(_input_set);
	_y_hat = avn.sign_normv(_z);
}

void MLPPDualSVC::alpha_projection() {
	for (int i = 0; i < _alpha->size(); i++) {
		if (_alpha->get_element(i) > _C) {
			_alpha->set_element(i, _C);
		} else if (_alpha->get_element(i) < 0) {
			_alpha->set_element(i, 0);
		}
	}
}

real_t MLPPDualSVC::kernel_functionv(const Ref<MLPPVector> &v, const Ref<MLPPVector> &u, KernelMethod kernel) {
	MLPPLinAlg alg;

	if (kernel == KERNEL_METHOD_LINEAR) {
		return alg.dotnv(u, v);
	}

	return 0;
}

Ref<MLPPMatrix> MLPPDualSVC::kernel_functionm(const Ref<MLPPMatrix> &U, const Ref<MLPPMatrix> &V, KernelMethod kernel) {
	MLPPLinAlg alg;

	if (kernel == KERNEL_METHOD_LINEAR) {
		return alg.matmultnm(_input_set, alg.transposenm(_input_set));
	}

	Ref<MLPPMatrix> m;
	m.instance();

	return m;
}

void MLPPDualSVC::_bind_methods() {
}
