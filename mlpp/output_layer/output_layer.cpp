//
//  OutputLayer.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "output_layer.h"
#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

int MLPPOutputLayer::get_n_hidden() {
	return n_hidden;
}
void MLPPOutputLayer::set_n_hidden(const int val) {
	n_hidden = val;
}

MLPPActivation::ActivationFunction MLPPOutputLayer::get_activation() {
	return activation;
}
void MLPPOutputLayer::set_activation(const MLPPActivation::ActivationFunction val) {
	activation = val;
}

Ref<MLPPMatrix> MLPPOutputLayer::get_input() {
	return input;
}
void MLPPOutputLayer::set_input(const Ref<MLPPMatrix> &val) {
	input = val;
}

Ref<MLPPVector> MLPPOutputLayer::get_weights() {
	return weights;
}
void MLPPOutputLayer::set_weights(const Ref<MLPPVector> &val) {
	weights = val;
}

real_t MLPPOutputLayer::MLPPOutputLayer::get_bias() {
	return bias;
}
void MLPPOutputLayer::set_bias(const real_t val) {
	bias = val;
}

Ref<MLPPVector> MLPPOutputLayer::get_z() {
	return z;
}
void MLPPOutputLayer::set_z(const Ref<MLPPVector> &val) {
	z = val;
}

Ref<MLPPVector> MLPPOutputLayer::get_a() {
	return a;
}
void MLPPOutputLayer::set_a(const Ref<MLPPVector> &val) {
	a = val;
}

Ref<MLPPVector> MLPPOutputLayer::get_z_test() {
	return z_test;
}
void MLPPOutputLayer::set_z_test(const Ref<MLPPVector> &val) {
	z_test = val;
}

Ref<MLPPVector> MLPPOutputLayer::get_a_test() {
	return a_test;
}
void MLPPOutputLayer::set_a_test(const Ref<MLPPVector> &val) {
	a_test = val;
}

Ref<MLPPVector> MLPPOutputLayer::get_delta() {
	return delta;
}
void MLPPOutputLayer::set_delta(const Ref<MLPPVector> &val) {
	delta = val;
}

MLPPReg::RegularizationType MLPPOutputLayer::get_reg() {
	return reg;
}
void MLPPOutputLayer::set_reg(const MLPPReg::RegularizationType val) {
	reg = val;
}

real_t MLPPOutputLayer::get_lambda() {
	return lambda;
}
void MLPPOutputLayer::set_lambda(const real_t val) {
	lambda = val;
}

real_t MLPPOutputLayer::get_alpha() {
	return alpha;
}
void MLPPOutputLayer::set_alpha(const real_t val) {
	alpha = val;
}

MLPPUtilities::WeightDistributionType MLPPOutputLayer::get_weight_init() {
	return weight_init;
}
void MLPPOutputLayer::set_weight_init(const MLPPUtilities::WeightDistributionType val) {
	weight_init = val;
}

void MLPPOutputLayer::forward_pass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	//z = alg.mat_vec_addv(alg.matmultm(input, weights), bias);
	//a = avn.run_activation_norm_matrix(activation, z);
}

void MLPPOutputLayer::test(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	//z_test = alg.additionm(alg.mat_vec_multv(alg.transposem(weights), x), bias);
	//a_test = avn.run_activation_norm_matrix(activation, z_test);
}

MLPPOutputLayer::MLPPOutputLayer(int p_n_hidden, MLPPActivation::ActivationFunction p_activation, Ref<MLPPMatrix> p_input, MLPPUtilities::WeightDistributionType p_weight_init, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
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
	bias = 0;

	//weights->resize(Size2i(input->size().x, n_hidden));
	//bias->resize(n_hidden);

	//MLPPUtilities utils;

	//utils.weight_initializationm(weights, weight_init);
	//utils.bias_initializationv(bias);
}

MLPPOutputLayer::MLPPOutputLayer() {
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
	bias = 0;
}
MLPPOutputLayer::~MLPPOutputLayer() {
}

void MLPPOutputLayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_n_hidden"), &MLPPOutputLayer::get_n_hidden);
	ClassDB::bind_method(D_METHOD("set_n_hidden", "val"), &MLPPOutputLayer::set_n_hidden);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "n_hidden"), "set_n_hidden", "get_n_hidden");

	ClassDB::bind_method(D_METHOD("get_activation"), &MLPPOutputLayer::get_activation);
	ClassDB::bind_method(D_METHOD("set_activation", "val"), &MLPPOutputLayer::set_activation);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "activation"), "set_activation", "get_activation");

	ClassDB::bind_method(D_METHOD("get_input"), &MLPPOutputLayer::get_input);
	ClassDB::bind_method(D_METHOD("set_input", "val"), &MLPPOutputLayer::set_input);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input", "get_input");

	ClassDB::bind_method(D_METHOD("get_weights"), &MLPPOutputLayer::get_weights);
	ClassDB::bind_method(D_METHOD("set_weights", "val"), &MLPPOutputLayer::set_weights);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "weights", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_weights", "get_weights");

	ClassDB::bind_method(D_METHOD("get_bias"), &MLPPOutputLayer::get_bias);
	ClassDB::bind_method(D_METHOD("set_bias", "val"), &MLPPOutputLayer::set_bias);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bias"), "set_bias", "get_bias");

	ClassDB::bind_method(D_METHOD("get_z"), &MLPPOutputLayer::get_z);
	ClassDB::bind_method(D_METHOD("set_z", "val"), &MLPPOutputLayer::set_z);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "z", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_z", "get_z");

	ClassDB::bind_method(D_METHOD("get_a"), &MLPPOutputLayer::get_a);
	ClassDB::bind_method(D_METHOD("set_a", "val"), &MLPPOutputLayer::set_a);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "a", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_a", "get_a");

	ClassDB::bind_method(D_METHOD("get_z_test"), &MLPPOutputLayer::get_z_test);
	ClassDB::bind_method(D_METHOD("set_z_test", "val"), &MLPPOutputLayer::set_z_test);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "z_test", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_z_test", "get_z_test");

	ClassDB::bind_method(D_METHOD("get_a_test"), &MLPPOutputLayer::get_a_test);
	ClassDB::bind_method(D_METHOD("set_a_test", "val"), &MLPPOutputLayer::set_a_test);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "a_test", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_a_test", "get_a_test");

	ClassDB::bind_method(D_METHOD("get_delta"), &MLPPOutputLayer::get_delta);
	ClassDB::bind_method(D_METHOD("set_delta", "val"), &MLPPOutputLayer::set_delta);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "delta", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_delta", "get_delta");

	ClassDB::bind_method(D_METHOD("get_reg"), &MLPPOutputLayer::get_reg);
	ClassDB::bind_method(D_METHOD("set_reg", "val"), &MLPPOutputLayer::set_reg);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reg"), "set_reg", "get_reg");

	ClassDB::bind_method(D_METHOD("get_lambda"), &MLPPOutputLayer::get_lambda);
	ClassDB::bind_method(D_METHOD("set_lambda", "val"), &MLPPOutputLayer::set_lambda);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lambda"), "set_lambda", "get_lambda");

	ClassDB::bind_method(D_METHOD("get_alpha"), &MLPPOutputLayer::get_alpha);
	ClassDB::bind_method(D_METHOD("set_alpha", "val"), &MLPPOutputLayer::set_alpha);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alpha"), "set_alpha", "get_alpha");

	ClassDB::bind_method(D_METHOD("get_weight_init"), &MLPPOutputLayer::get_weight_init);
	ClassDB::bind_method(D_METHOD("set_weight_init", "val"), &MLPPOutputLayer::set_weight_init);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "set_weight_init"), "set_weight_init", "get_weight_init");

	ClassDB::bind_method(D_METHOD("forward_pass"), &MLPPOutputLayer::forward_pass);
	ClassDB::bind_method(D_METHOD("test", "x"), &MLPPOutputLayer::test);
}

MLPPOldOutputLayer::MLPPOldOutputLayer(int n_hidden, std::string activation, std::string cost, std::vector<std::vector<real_t>> input, std::string weightInit, std::string reg, real_t lambda, real_t alpha) :
		n_hidden(n_hidden), activation(activation), cost(cost), input(input), weightInit(weightInit), reg(reg), lambda(lambda), alpha(alpha) {
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
	a = (avn.*activation_map[activation])(z, 0);
}

void MLPPOldOutputLayer::Test(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	z_test = alg.dot(weights, x) + bias;
	a_test = (avn.*activationTest_map[activation])(z_test, 0);
}
