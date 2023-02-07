//
//  OutputLayer.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "output_layer_old.h"
#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

MLPPOldOutputLayer::MLPPOldOutputLayer(int p_n_hidden, std::string p_activation, std::string p_cost, std::vector<std::vector<real_t>> p_input, std::string p_weightInit, std::string p_reg, real_t p_lambda, real_t p_alpha) {
	n_hidden = p_n_hidden;
	activation = p_activation;
	cost = p_cost;
	input = p_input;
	weightInit = p_weightInit;
	reg = p_reg;
	lambda = p_lambda;
	alpha = p_alpha;

	weights = MLPPUtilities::weightInitialization(n_hidden, weightInit);
	bias = MLPPUtilities::biasInitialization();

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

	costDeriv_map["MSE"] = &MLPPCost::MSEDeriv;
	cost_map["MSE"] = &MLPPCost::MSE;
	costDeriv_map["RMSE"] = &MLPPCost::RMSEDeriv;
	cost_map["RMSE"] = &MLPPCost::RMSE;
	costDeriv_map["MAE"] = &MLPPCost::MAEDeriv;
	cost_map["MAE"] = &MLPPCost::MAE;
	costDeriv_map["MBE"] = &MLPPCost::MBEDeriv;
	cost_map["MBE"] = &MLPPCost::MBE;
	costDeriv_map["LogLoss"] = &MLPPCost::LogLossDeriv;
	cost_map["LogLoss"] = &MLPPCost::LogLoss;
	costDeriv_map["CrossEntropy"] = &MLPPCost::CrossEntropyDeriv;
	cost_map["CrossEntropy"] = &MLPPCost::CrossEntropy;
	costDeriv_map["HingeLoss"] = &MLPPCost::HingeLossDeriv;
	cost_map["HingeLoss"] = &MLPPCost::HingeLoss;
	costDeriv_map["WassersteinLoss"] = &MLPPCost::HingeLossDeriv;
	cost_map["WassersteinLoss"] = &MLPPCost::HingeLoss;
}

void MLPPOldOutputLayer::forwardPass() {
	MLPPLinAlg alg;
	MLPPActivation avn;
	z = alg.scalarAdd(bias, alg.mat_vec_mult(input, weights));
	a = (avn.*activation_map[activation])(z, false);
}

void MLPPOldOutputLayer::Test(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	z_test = alg.dot(weights, x) + bias;
	a_test = (avn.*activationTest_map[activation])(z_test, false);
}
