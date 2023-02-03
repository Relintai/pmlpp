//
//  HiddenLayer.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "hidden_layer.h"
#include "../activation/activation.h"
#include "../lin_alg/lin_alg.h"

#include <iostream>
#include <random>

void MLPPHiddenLayer::forward_pass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	z = alg.mat_vec_addv(alg.matmultm(input, weights), bias);
	a = avn.run_activation_norm_matrix(activation, z);
}

void MLPPHiddenLayer::test(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	z_test = alg.additionm(alg.mat_vec_multv(alg.transposem(weights), x), bias);
	a_test = avn.run_activation_norm_matrix(activation, z_test);
}

MLPPHiddenLayer::MLPPHiddenLayer(int p_n_hidden, MLPPActivation::ActivationFunction p_activation, Ref<MLPPMatrix> p_input, MLPPUtilities::WeightDistributionType p_weight_init, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	n_hidden = p_n_hidden;
	activation = p_activation;

	input = p_input;

	// Regularization Params
	reg = p_reg;
	lambda = p_lambda; /* Regularization Parameter */
	alpha = p_alpha; /* This is the controlling param for Elastic Net*/

	weight_init = p_weight_init;

	z.instance();
	a.instance();

	z_test.instance();
	a_test.instance();

	delta.instance();

	weights.instance();
	bias.instance();

	weights->resize(Size2i(input->size().x, n_hidden));
	bias->resize(n_hidden);

	MLPPUtilities::weight_initializationm(weights, weight_init);
	MLPPUtilities::bias_initializationv(bias);
}

MLPPHiddenLayer::MLPPHiddenLayer() {
	n_hidden = 0;
	activation = MLPPActivation::ACTIVATION_FUNCTION_LINEAR;

	// Regularization Params
	//reg = 0;
	lambda = 0; /* Regularization Parameter */
	alpha = 0; /* This is the controlling param for Elastic Net*/

	weight_init = MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT;

	z.instance();
	a.instance();

	z_test.instance();
	a_test.instance();

	delta.instance();

	weights.instance();
	bias.instance();
}
MLPPHiddenLayer::~MLPPHiddenLayer() {
}

MLPPOldHiddenLayer::MLPPOldHiddenLayer(int n_hidden, std::string activation, std::vector<std::vector<real_t>> input, std::string weightInit, std::string reg, real_t lambda, real_t alpha) :
		n_hidden(n_hidden), activation(activation), input(input), weightInit(weightInit), reg(reg), lambda(lambda), alpha(alpha) {
	weights = MLPPUtilities::weightInitialization(input[0].size(), n_hidden, weightInit);
	bias = MLPPUtilities::biasInitialization(n_hidden);

	activation_map["Linear"] = &MLPPActivation::linear;
	activationTest_map["Linear"] = &MLPPActivation::linear;

	activation_map["Sigmoid"] = &MLPPActivation::sigmoid;
	activationTest_map["Sigmoid"] = &MLPPActivation::sigmoid;

	activation_map["Swish"] = &MLPPActivation::swish;
	activationTest_map["Swish"] = &MLPPActivation::swish;

	activation_map["Mish"] = &MLPPActivation::mish;
	activationTest_map["Mish"] = &MLPPActivation::mish;

	activation_map["SinC"] = &MLPPActivation::sinc;
	activationTest_map["SinC"] = &MLPPActivation::sinc;

	activation_map["Softplus"] = &MLPPActivation::softplus;
	activationTest_map["Softplus"] = &MLPPActivation::softplus;

	activation_map["Softsign"] = &MLPPActivation::softsign;
	activationTest_map["Softsign"] = &MLPPActivation::softsign;

	activation_map["CLogLog"] = &MLPPActivation::cloglog;
	activationTest_map["CLogLog"] = &MLPPActivation::cloglog;

	activation_map["Logit"] = &MLPPActivation::logit;
	activationTest_map["Logit"] = &MLPPActivation::logit;

	activation_map["GaussianCDF"] = &MLPPActivation::gaussianCDF;
	activationTest_map["GaussianCDF"] = &MLPPActivation::gaussianCDF;

	activation_map["RELU"] = &MLPPActivation::RELU;
	activationTest_map["RELU"] = &MLPPActivation::RELU;

	activation_map["GELU"] = &MLPPActivation::GELU;
	activationTest_map["GELU"] = &MLPPActivation::GELU;

	activation_map["Sign"] = &MLPPActivation::sign;
	activationTest_map["Sign"] = &MLPPActivation::sign;

	activation_map["UnitStep"] = &MLPPActivation::unitStep;
	activationTest_map["UnitStep"] = &MLPPActivation::unitStep;

	activation_map["Sinh"] = &MLPPActivation::sinh;
	activationTest_map["Sinh"] = &MLPPActivation::sinh;

	activation_map["Cosh"] = &MLPPActivation::cosh;
	activationTest_map["Cosh"] = &MLPPActivation::cosh;

	activation_map["Tanh"] = &MLPPActivation::tanh;
	activationTest_map["Tanh"] = &MLPPActivation::tanh;

	activation_map["Csch"] = &MLPPActivation::csch;
	activationTest_map["Csch"] = &MLPPActivation::csch;

	activation_map["Sech"] = &MLPPActivation::sech;
	activationTest_map["Sech"] = &MLPPActivation::sech;

	activation_map["Coth"] = &MLPPActivation::coth;
	activationTest_map["Coth"] = &MLPPActivation::coth;

	activation_map["Arsinh"] = &MLPPActivation::arsinh;
	activationTest_map["Arsinh"] = &MLPPActivation::arsinh;

	activation_map["Arcosh"] = &MLPPActivation::arcosh;
	activationTest_map["Arcosh"] = &MLPPActivation::arcosh;

	activation_map["Artanh"] = &MLPPActivation::artanh;
	activationTest_map["Artanh"] = &MLPPActivation::artanh;

	activation_map["Arcsch"] = &MLPPActivation::arcsch;
	activationTest_map["Arcsch"] = &MLPPActivation::arcsch;

	activation_map["Arsech"] = &MLPPActivation::arsech;
	activationTest_map["Arsech"] = &MLPPActivation::arsech;

	activation_map["Arcoth"] = &MLPPActivation::arcoth;
	activationTest_map["Arcoth"] = &MLPPActivation::arcoth;
}

void MLPPOldHiddenLayer::forwardPass() {
	MLPPLinAlg alg;
	MLPPActivation avn;
	z = alg.mat_vec_add(alg.matmult(input, weights), bias);
	a = (avn.*activation_map[activation])(z, false);
}

void MLPPOldHiddenLayer::Test(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	z_test = alg.addition(alg.mat_vec_mult(alg.transpose(weights), x), bias);
	a_test = (avn.*activationTest_map[activation])(z_test, false);
}
