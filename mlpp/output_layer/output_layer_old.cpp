//
//  OutputLayer.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "output_layer_old.h"
#include "../lin_alg/lin_alg_old.h"
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

	costDeriv_map["MSE"] = &MLPPCostOld::MSEDeriv;
	cost_map["MSE"] = &MLPPCostOld::MSE;
	costDeriv_map["RMSE"] = &MLPPCostOld::RMSEDeriv;
	cost_map["RMSE"] = &MLPPCostOld::RMSE;
	costDeriv_map["MAE"] = &MLPPCostOld::MAEDeriv;
	cost_map["MAE"] = &MLPPCostOld::MAE;
	costDeriv_map["MBE"] = &MLPPCostOld::MBEDeriv;
	cost_map["MBE"] = &MLPPCostOld::MBE;
	costDeriv_map["LogLoss"] = &MLPPCostOld::LogLossDeriv;
	cost_map["LogLoss"] = &MLPPCostOld::LogLoss;
	costDeriv_map["CrossEntropy"] = &MLPPCostOld::CrossEntropyDeriv;
	cost_map["CrossEntropy"] = &MLPPCostOld::CrossEntropy;
	costDeriv_map["HingeLoss"] = &MLPPCostOld::HingeLossDeriv;
	cost_map["HingeLoss"] = &MLPPCostOld::HingeLoss;
	costDeriv_map["WassersteinLoss"] = &MLPPCostOld::HingeLossDeriv;
	cost_map["WassersteinLoss"] = &MLPPCostOld::HingeLoss;
}

void MLPPOldOutputLayer::forwardPass() {
	MLPPLinAlgOld alg;
	MLPPActivationOld avn;
	z = alg.scalarAdd(bias, alg.mat_vec_mult(input, weights));
	a = (avn.*activation_map[activation])(z, false);
}

void MLPPOldOutputLayer::Test(std::vector<real_t> x) {
	MLPPLinAlgOld alg;
	MLPPActivationOld avn;
	z_test = alg.dot(weights, x) + bias;
	a_test = (avn.*activationTest_map[activation])(z_test, false);
}
