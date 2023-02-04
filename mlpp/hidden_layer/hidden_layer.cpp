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

int MLPPHiddenLayer::get_n_hidden() {
	return n_hidden;
}
void MLPPHiddenLayer::set_n_hidden(const int val) {
	n_hidden = val;
}

MLPPActivation::ActivationFunction MLPPHiddenLayer::get_activation() {
	return activation;
}
void MLPPHiddenLayer::set_activation(const MLPPActivation::ActivationFunction val) {
	activation = val;
}

Ref<MLPPMatrix> MLPPHiddenLayer::get_input() {
	return input;
}
void MLPPHiddenLayer::set_input(const Ref<MLPPMatrix> &val) {
	input = val;
}

Ref<MLPPMatrix> MLPPHiddenLayer::get_weights() {
	return weights;
}
void MLPPHiddenLayer::set_weights(const Ref<MLPPMatrix> &val) {
	weights = val;
}

Ref<MLPPVector> MLPPHiddenLayer::MLPPHiddenLayer::get_bias() {
	return bias;
}
void MLPPHiddenLayer::set_bias(const Ref<MLPPVector> &val) {
	bias = val;
}

Ref<MLPPMatrix> MLPPHiddenLayer::get_z() {
	return z;
}
void MLPPHiddenLayer::set_z(const Ref<MLPPMatrix> &val) {
	z = val;
}

Ref<MLPPMatrix> MLPPHiddenLayer::get_a() {
	return a;
}
void MLPPHiddenLayer::set_a(const Ref<MLPPMatrix> &val) {
	a = val;
}

Ref<MLPPVector> MLPPHiddenLayer::get_z_test() {
	return z_test;
}
void MLPPHiddenLayer::set_z_test(const Ref<MLPPVector> &val) {
	z_test = val;
}

Ref<MLPPVector> MLPPHiddenLayer::get_a_test() {
	return a_test;
}
void MLPPHiddenLayer::set_a_test(const Ref<MLPPVector> &val) {
	a_test = val;
}

Ref<MLPPMatrix> MLPPHiddenLayer::get_delta() {
	return delta;
}
void MLPPHiddenLayer::set_delta(const Ref<MLPPMatrix> &val) {
	delta = val;
}

MLPPReg::RegularizationType MLPPHiddenLayer::get_reg() {
	return reg;
}
void MLPPHiddenLayer::set_reg(const MLPPReg::RegularizationType val) {
	reg = val;
}

real_t MLPPHiddenLayer::get_lambda() {
	return lambda;
}
void MLPPHiddenLayer::set_lambda(const real_t val) {
	lambda = val;
}

real_t MLPPHiddenLayer::get_alpha() {
	return alpha;
}
void MLPPHiddenLayer::set_alpha(const real_t val) {
	alpha = val;
}

MLPPUtilities::WeightDistributionType MLPPHiddenLayer::get_weight_init() {
	return weight_init;
}
void MLPPHiddenLayer::set_weight_init(const MLPPUtilities::WeightDistributionType val) {
	weight_init = val;
}

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

	MLPPUtilities utils;

	utils.weight_initializationm(weights, weight_init);
	utils.bias_initializationv(bias);
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

void MLPPHiddenLayer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_n_hidden"), &MLPPHiddenLayer::get_n_hidden);
	ClassDB::bind_method(D_METHOD("set_n_hidden", "val"), &MLPPHiddenLayer::set_n_hidden);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "n_hidden"), "set_n_hidden", "get_n_hidden");

	ClassDB::bind_method(D_METHOD("get_activation"), &MLPPHiddenLayer::get_activation);
	ClassDB::bind_method(D_METHOD("set_activation", "val"), &MLPPHiddenLayer::set_activation);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "activation"), "set_activation", "get_activation");

	ClassDB::bind_method(D_METHOD("get_input"), &MLPPHiddenLayer::get_input);
	ClassDB::bind_method(D_METHOD("set_input", "val"), &MLPPHiddenLayer::set_input);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input", "get_input");

	ClassDB::bind_method(D_METHOD("get_weights"), &MLPPHiddenLayer::get_weights);
	ClassDB::bind_method(D_METHOD("set_weights", "val"), &MLPPHiddenLayer::set_weights);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "weights", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_weights", "get_weights");

	ClassDB::bind_method(D_METHOD("get_bias"), &MLPPHiddenLayer::get_bias);
	ClassDB::bind_method(D_METHOD("set_bias", "val"), &MLPPHiddenLayer::set_bias);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "bias", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_bias", "get_bias");

	ClassDB::bind_method(D_METHOD("get_z"), &MLPPHiddenLayer::get_z);
	ClassDB::bind_method(D_METHOD("set_z", "val"), &MLPPHiddenLayer::set_z);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "z", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_z", "get_z");

	ClassDB::bind_method(D_METHOD("get_a"), &MLPPHiddenLayer::get_a);
	ClassDB::bind_method(D_METHOD("set_a", "val"), &MLPPHiddenLayer::set_a);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "a", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_a", "get_a");

	ClassDB::bind_method(D_METHOD("get_z_test"), &MLPPHiddenLayer::get_z_test);
	ClassDB::bind_method(D_METHOD("set_z_test", "val"), &MLPPHiddenLayer::set_z_test);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "z_test", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_z_test", "get_z_test");

	ClassDB::bind_method(D_METHOD("get_a_test"), &MLPPHiddenLayer::get_a_test);
	ClassDB::bind_method(D_METHOD("set_a_test", "val"), &MLPPHiddenLayer::set_a_test);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "a_test", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_a_test", "get_a_test");

	ClassDB::bind_method(D_METHOD("get_delta"), &MLPPHiddenLayer::get_delta);
	ClassDB::bind_method(D_METHOD("set_delta", "val"), &MLPPHiddenLayer::set_delta);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "delta", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_delta", "get_delta");

	ClassDB::bind_method(D_METHOD("get_reg"), &MLPPHiddenLayer::get_reg);
	ClassDB::bind_method(D_METHOD("set_reg", "val"), &MLPPHiddenLayer::set_reg);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reg"), "set_reg", "get_reg");

	ClassDB::bind_method(D_METHOD("get_lambda"), &MLPPHiddenLayer::get_lambda);
	ClassDB::bind_method(D_METHOD("set_lambda", "val"), &MLPPHiddenLayer::set_lambda);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lambda"), "set_lambda", "get_lambda");

	ClassDB::bind_method(D_METHOD("get_alpha"), &MLPPHiddenLayer::get_alpha);
	ClassDB::bind_method(D_METHOD("set_alpha", "val"), &MLPPHiddenLayer::set_alpha);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alpha"), "set_alpha", "get_alpha");

	ClassDB::bind_method(D_METHOD("get_weight_init"), &MLPPHiddenLayer::get_weight_init);
	ClassDB::bind_method(D_METHOD("set_weight_init", "val"), &MLPPHiddenLayer::set_weight_init);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "set_weight_init"), "set_weight_init", "get_weight_init");

	ClassDB::bind_method(D_METHOD("forward_pass"), &MLPPHiddenLayer::forward_pass);
	ClassDB::bind_method(D_METHOD("test", "x"), &MLPPHiddenLayer::test);
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
