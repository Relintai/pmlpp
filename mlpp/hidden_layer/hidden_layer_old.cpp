//
//  HiddenLayer.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "hidden_layer_old.h"
#include "../activation/activation.h"
#include "../lin_alg/lin_alg.h"

#include <iostream>
#include <random>

MLPPOldHiddenLayer::MLPPOldHiddenLayer(int p_n_hidden, std::string p_activation, std::vector<std::vector<real_t>> p_input, std::string p_weightInit, std::string p_reg, real_t p_lambda, real_t p_alpha) {
	n_hidden = p_n_hidden;
	activation = p_activation;
	input = p_input;
	weightInit = p_weightInit;
	reg = p_reg;
	lambda = p_lambda;
	alpha = p_alpha;

	weights = MLPPUtilities::weightInitialization(input[0].size(), n_hidden, weightInit);
	bias = MLPPUtilities::biasInitialization(n_hidden);

	activation_map["Linear"] = &MLPPActivationOld::linear;
	activationTest_map["Linear"] = &MLPPActivationOld::linear;

	activation_map["Sigmoid"] = &MLPPActivationOld::sigmoid;
	activationTest_map["Sigmoid"] = &MLPPActivationOld::sigmoid;

	activation_map["Swish"] = &MLPPActivationOld::swish;
	activationTest_map["Swish"] = &MLPPActivationOld::swish;

	activation_map["Mish"] = &MLPPActivationOld::mish;
	activationTest_map["Mish"] = &MLPPActivationOld::mish;

	activation_map["SinC"] = &MLPPActivationOld::sinc;
	activationTest_map["SinC"] = &MLPPActivationOld::sinc;

	activation_map["Softplus"] = &MLPPActivationOld::softplus;
	activationTest_map["Softplus"] = &MLPPActivationOld::softplus;

	activation_map["Softsign"] = &MLPPActivationOld::softsign;
	activationTest_map["Softsign"] = &MLPPActivationOld::softsign;

	activation_map["CLogLog"] = &MLPPActivationOld::cloglog;
	activationTest_map["CLogLog"] = &MLPPActivationOld::cloglog;

	activation_map["Logit"] = &MLPPActivationOld::logit;
	activationTest_map["Logit"] = &MLPPActivationOld::logit;

	activation_map["GaussianCDF"] = &MLPPActivationOld::gaussianCDF;
	activationTest_map["GaussianCDF"] = &MLPPActivationOld::gaussianCDF;

	activation_map["RELU"] = &MLPPActivationOld::RELU;
	activationTest_map["RELU"] = &MLPPActivationOld::RELU;

	activation_map["GELU"] = &MLPPActivationOld::GELU;
	activationTest_map["GELU"] = &MLPPActivationOld::GELU;

	activation_map["Sign"] = &MLPPActivationOld::sign;
	activationTest_map["Sign"] = &MLPPActivationOld::sign;

	activation_map["UnitStep"] = &MLPPActivationOld::unitStep;
	activationTest_map["UnitStep"] = &MLPPActivationOld::unitStep;

	activation_map["Sinh"] = &MLPPActivationOld::sinh;
	activationTest_map["Sinh"] = &MLPPActivationOld::sinh;

	activation_map["Cosh"] = &MLPPActivationOld::cosh;
	activationTest_map["Cosh"] = &MLPPActivationOld::cosh;

	activation_map["Tanh"] = &MLPPActivationOld::tanh;
	activationTest_map["Tanh"] = &MLPPActivationOld::tanh;

	activation_map["Csch"] = &MLPPActivationOld::csch;
	activationTest_map["Csch"] = &MLPPActivationOld::csch;

	activation_map["Sech"] = &MLPPActivationOld::sech;
	activationTest_map["Sech"] = &MLPPActivationOld::sech;

	activation_map["Coth"] = &MLPPActivationOld::coth;
	activationTest_map["Coth"] = &MLPPActivationOld::coth;

	activation_map["Arsinh"] = &MLPPActivationOld::arsinh;
	activationTest_map["Arsinh"] = &MLPPActivationOld::arsinh;

	activation_map["Arcosh"] = &MLPPActivationOld::arcosh;
	activationTest_map["Arcosh"] = &MLPPActivationOld::arcosh;

	activation_map["Artanh"] = &MLPPActivationOld::artanh;
	activationTest_map["Artanh"] = &MLPPActivationOld::artanh;

	activation_map["Arcsch"] = &MLPPActivationOld::arcsch;
	activationTest_map["Arcsch"] = &MLPPActivationOld::arcsch;

	activation_map["Arsech"] = &MLPPActivationOld::arsech;
	activationTest_map["Arsech"] = &MLPPActivationOld::arsech;

	activation_map["Arcoth"] = &MLPPActivationOld::arcoth;
	activationTest_map["Arcoth"] = &MLPPActivationOld::arcoth;
}

void MLPPOldHiddenLayer::forwardPass() {
	MLPPLinAlg alg;
	MLPPActivationOld avn;

	z = alg.mat_vec_add(alg.matmult(input, weights), bias);
	a = (avn.*activation_map[activation])(z, false);
}

void MLPPOldHiddenLayer::Test(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivationOld avn;
	z_test = alg.addition(alg.mat_vec_mult(alg.transpose(weights), x), bias);
	a_test = (avn.*activationTest_map[activation])(z_test, false);
}
