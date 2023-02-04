//
//  MultiOutputLayer.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "multi_output_layer.h"
#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

int MLPPMultiOutputLayer::get_n_output() {
	return n_output;
}
void MLPPMultiOutputLayer::set_n_output(const int val) {
	n_output = val;
}

int MLPPMultiOutputLayer::get_n_hidden() {
	return n_hidden;
}
void MLPPMultiOutputLayer::set_n_hidden(const int val) {
	n_hidden = val;
}

MLPPActivation::ActivationFunction MLPPMultiOutputLayer::get_activation() {
	return activation;
}
void MLPPMultiOutputLayer::set_activation(const MLPPActivation::ActivationFunction val) {
	activation = val;
}

MLPPCost::CostTypes MLPPMultiOutputLayer::get_cost() {
	return cost;
}
void MLPPMultiOutputLayer::set_cost(const MLPPCost::CostTypes val) {
	cost = val;
}

Ref<MLPPMatrix> MLPPMultiOutputLayer::get_input() {
	return input;
}
void MLPPMultiOutputLayer::set_input(const Ref<MLPPMatrix> &val) {
	input = val;
}

Ref<MLPPMatrix> MLPPMultiOutputLayer::get_weights() {
	return weights;
}
void MLPPMultiOutputLayer::set_weights(const Ref<MLPPMatrix> &val) {
	weights = val;
}

Ref<MLPPVector> MLPPMultiOutputLayer::get_bias() {
	return bias;
}
void MLPPMultiOutputLayer::set_bias(const Ref<MLPPVector> &val) {
	bias = val;
}

Ref<MLPPMatrix> MLPPMultiOutputLayer::get_z() {
	return z;
}
void MLPPMultiOutputLayer::set_z(const Ref<MLPPMatrix> &val) {
	z = val;
}

Ref<MLPPMatrix> MLPPMultiOutputLayer::get_a() {
	return a;
}
void MLPPMultiOutputLayer::set_a(const Ref<MLPPMatrix> &val) {
	a = val;
}

Ref<MLPPVector> MLPPMultiOutputLayer::get_z_test() {
	return z_test;
}
void MLPPMultiOutputLayer::set_z_test(const Ref<MLPPVector> &val) {
	z_test = val;
}

Ref<MLPPVector> MLPPMultiOutputLayer::get_a_test() {
	return a_test;
}
void MLPPMultiOutputLayer::set_a_test(const Ref<MLPPVector> &val) {
	a_test = val;
}

Ref<MLPPMatrix> MLPPMultiOutputLayer::get_delta() {
	return delta;
}
void MLPPMultiOutputLayer::set_delta(const Ref<MLPPMatrix> &val) {
	delta = val;
}

MLPPReg::RegularizationType MLPPMultiOutputLayer::get_reg() {
	return reg;
}
void MLPPMultiOutputLayer::set_reg(const MLPPReg::RegularizationType val) {
	reg = val;
}

real_t MLPPMultiOutputLayer::get_lambda() {
	return lambda;
}
void MLPPMultiOutputLayer::set_lambda(const real_t val) {
	lambda = val;
}

real_t MLPPMultiOutputLayer::get_alpha() {
	return alpha;
}
void MLPPMultiOutputLayer::set_alpha(const real_t val) {
	alpha = val;
}

MLPPUtilities::WeightDistributionType MLPPMultiOutputLayer::get_weight_init() {
	return weight_init;
}
void MLPPMultiOutputLayer::set_weight_init(const MLPPUtilities::WeightDistributionType val) {
	weight_init = val;
}

void MLPPMultiOutputLayer::forward_pass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	z = alg.mat_vec_addv(alg.matmultm(input, weights), bias);
	a = avn.run_activation_norm_matrix(activation, z);
}

void MLPPMultiOutputLayer::test(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	z_test = alg.additionm(alg.mat_vec_multv(alg.transposem(weights), x), bias);
	a_test = avn.run_activation_norm_vector(activation, z_test);
}

MLPPMultiOutputLayer::MLPPMultiOutputLayer(int p_n_hidden, MLPPActivation::ActivationFunction p_activation, Ref<MLPPMatrix> p_input, MLPPUtilities::WeightDistributionType p_weight_init, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
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

	weights->resize(Size2i(n_hidden, n_output));
	bias->resize(n_output);

	MLPPUtilities utils;

	utils.weight_initializationm(weights, weight_init);
	utils.bias_initializationv(bias);
}

MLPPMultiOutputLayer::MLPPMultiOutputLayer() {
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
MLPPMultiOutputLayer::~MLPPMultiOutputLayer() {
}

void MLPPMultiOutputLayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_n_output"), &MLPPMultiOutputLayer::get_n_output);
	ClassDB::bind_method(D_METHOD("set_n_output", "val"), &MLPPMultiOutputLayer::set_n_output);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "n_output"), "set_n_output", "get_n_output");

	ClassDB::bind_method(D_METHOD("get_n_hidden"), &MLPPMultiOutputLayer::get_n_hidden);
	ClassDB::bind_method(D_METHOD("set_n_hidden", "val"), &MLPPMultiOutputLayer::set_n_hidden);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "n_hidden"), "set_n_hidden", "get_n_hidden");

	ClassDB::bind_method(D_METHOD("get_activation"), &MLPPMultiOutputLayer::get_activation);
	ClassDB::bind_method(D_METHOD("set_activation", "val"), &MLPPMultiOutputLayer::set_activation);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "activation"), "set_activation", "get_activation");

	ClassDB::bind_method(D_METHOD("get_cost"), &MLPPMultiOutputLayer::get_cost);
	ClassDB::bind_method(D_METHOD("set_cost", "val"), &MLPPMultiOutputLayer::set_cost);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cost"), "set_cost", "get_cost");

	ClassDB::bind_method(D_METHOD("get_input"), &MLPPMultiOutputLayer::get_input);
	ClassDB::bind_method(D_METHOD("set_input", "val"), &MLPPMultiOutputLayer::set_input);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input", "get_input");

	ClassDB::bind_method(D_METHOD("get_weights"), &MLPPMultiOutputLayer::get_weights);
	ClassDB::bind_method(D_METHOD("set_weights", "val"), &MLPPMultiOutputLayer::set_weights);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "weights", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_weights", "get_weights");

	ClassDB::bind_method(D_METHOD("get_bias"), &MLPPMultiOutputLayer::get_bias);
	ClassDB::bind_method(D_METHOD("set_bias", "val"), &MLPPMultiOutputLayer::set_bias);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "bias", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_bias", "get_bias");

	ClassDB::bind_method(D_METHOD("get_z"), &MLPPMultiOutputLayer::get_z);
	ClassDB::bind_method(D_METHOD("set_z", "val"), &MLPPMultiOutputLayer::set_z);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "z", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_z", "get_z");

	ClassDB::bind_method(D_METHOD("get_a"), &MLPPMultiOutputLayer::get_a);
	ClassDB::bind_method(D_METHOD("set_a", "val"), &MLPPMultiOutputLayer::set_a);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "a", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_a", "get_a");

	ClassDB::bind_method(D_METHOD("get_z_test"), &MLPPMultiOutputLayer::get_z_test);
	ClassDB::bind_method(D_METHOD("set_z_test", "val"), &MLPPMultiOutputLayer::set_z_test);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "z_test", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_z_test", "get_z_test");

	ClassDB::bind_method(D_METHOD("get_a_test"), &MLPPMultiOutputLayer::get_a_test);
	ClassDB::bind_method(D_METHOD("set_a_test", "val"), &MLPPMultiOutputLayer::set_a_test);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "a_test", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_a_test", "get_a_test");

	ClassDB::bind_method(D_METHOD("get_delta"), &MLPPMultiOutputLayer::get_delta);
	ClassDB::bind_method(D_METHOD("set_delta", "val"), &MLPPMultiOutputLayer::set_delta);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "delta", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_delta", "get_delta");

	ClassDB::bind_method(D_METHOD("get_reg"), &MLPPMultiOutputLayer::get_reg);
	ClassDB::bind_method(D_METHOD("set_reg", "val"), &MLPPMultiOutputLayer::set_reg);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reg"), "set_reg", "get_reg");

	ClassDB::bind_method(D_METHOD("get_lambda"), &MLPPMultiOutputLayer::get_lambda);
	ClassDB::bind_method(D_METHOD("set_lambda", "val"), &MLPPMultiOutputLayer::set_lambda);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lambda"), "set_lambda", "get_lambda");

	ClassDB::bind_method(D_METHOD("get_alpha"), &MLPPMultiOutputLayer::get_alpha);
	ClassDB::bind_method(D_METHOD("set_alpha", "val"), &MLPPMultiOutputLayer::set_alpha);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alpha"), "set_alpha", "get_alpha");

	ClassDB::bind_method(D_METHOD("get_weight_init"), &MLPPMultiOutputLayer::get_weight_init);
	ClassDB::bind_method(D_METHOD("set_weight_init", "val"), &MLPPMultiOutputLayer::set_weight_init);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "set_weight_init"), "set_weight_init", "get_weight_init");

	ClassDB::bind_method(D_METHOD("forward_pass"), &MLPPMultiOutputLayer::forward_pass);
	ClassDB::bind_method(D_METHOD("test", "x"), &MLPPMultiOutputLayer::test);
}

MLPPOldMultiOutputLayer::MLPPOldMultiOutputLayer(int p_n_output, int p_n_hidden, std::string p_activation, std::string p_cost, std::vector<std::vector<real_t>> p_input, std::string p_weightInit, std::string p_reg, real_t p_lambda, real_t p_alpha) {
	n_output = p_n_output;
	n_hidden = p_n_hidden;
	activation = p_activation;
	cost = p_cost;
	input = p_input;
	weightInit = p_weightInit;
	reg = p_reg;
	lambda = p_lambda;
	alpha = p_alpha;

	weights = MLPPUtilities::weightInitialization(n_hidden, n_output, weightInit);
	bias = MLPPUtilities::biasInitialization(n_output);

	activation_map["Linear"] = &MLPPActivation::linear;
	activationTest_map["Linear"] = &MLPPActivation::linear;

	activation_map["Sigmoid"] = &MLPPActivation::sigmoid;
	activationTest_map["Sigmoid"] = &MLPPActivation::sigmoid;

	activation_map["Softmax"] = &MLPPActivation::softmax;
	activationTest_map["Softmax"] = &MLPPActivation::softmax;

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

void MLPPOldMultiOutputLayer::forwardPass() {
	MLPPLinAlg alg;
	MLPPActivation avn;
	z = alg.mat_vec_add(alg.matmult(input, weights), bias);
	a = (avn.*activation_map[activation])(z, false);
}

void MLPPOldMultiOutputLayer::Test(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	z_test = alg.addition(alg.mat_vec_mult(alg.transpose(weights), x), bias);
	a_test = (avn.*activationTest_map[activation])(z_test, false);
}
