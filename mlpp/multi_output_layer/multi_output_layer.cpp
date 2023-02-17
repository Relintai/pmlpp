//
//  MultiOutputLayer.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "multi_output_layer.h"
#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

int MLPPMultiOutputLayer::get_n_output() {
	return _n_output;
}
void MLPPMultiOutputLayer::set_n_output(const int val) {
	_n_output = val;
}

int MLPPMultiOutputLayer::get_n_hidden() {
	return _n_hidden;
}
void MLPPMultiOutputLayer::set_n_hidden(const int val) {
	_n_hidden = val;
}

MLPPActivation::ActivationFunction MLPPMultiOutputLayer::get_activation() {
	return _activation;
}
void MLPPMultiOutputLayer::set_activation(const MLPPActivation::ActivationFunction val) {
	_activation = val;
}

MLPPCost::CostTypes MLPPMultiOutputLayer::get_cost() {
	return _cost;
}
void MLPPMultiOutputLayer::set_cost(const MLPPCost::CostTypes val) {
	_cost = val;
}

Ref<MLPPMatrix> MLPPMultiOutputLayer::get_input() {
	return _input;
}
void MLPPMultiOutputLayer::set_input(const Ref<MLPPMatrix> &val) {
	_input = val;
}

Ref<MLPPMatrix> MLPPMultiOutputLayer::get_weights() {
	return _weights;
}
void MLPPMultiOutputLayer::set_weights(const Ref<MLPPMatrix> &val) {
	_weights = val;
}

Ref<MLPPVector> MLPPMultiOutputLayer::get_bias() {
	return _bias;
}
void MLPPMultiOutputLayer::set_bias(const Ref<MLPPVector> &val) {
	_bias = val;
}

Ref<MLPPMatrix> MLPPMultiOutputLayer::get_z() {
	return _z;
}
void MLPPMultiOutputLayer::set_z(const Ref<MLPPMatrix> &val) {
	_z = val;
}

Ref<MLPPMatrix> MLPPMultiOutputLayer::get_a() {
	return _a;
}
void MLPPMultiOutputLayer::set_a(const Ref<MLPPMatrix> &val) {
	_a = val;
}

Ref<MLPPVector> MLPPMultiOutputLayer::get_z_test() {
	return _z_test;
}
void MLPPMultiOutputLayer::set_z_test(const Ref<MLPPVector> &val) {
	_z_test = val;
}

Ref<MLPPVector> MLPPMultiOutputLayer::get_a_test() {
	return _a_test;
}
void MLPPMultiOutputLayer::set_a_test(const Ref<MLPPVector> &val) {
	_a_test = val;
}

Ref<MLPPMatrix> MLPPMultiOutputLayer::get_delta() {
	return _delta;
}
void MLPPMultiOutputLayer::set_delta(const Ref<MLPPMatrix> &val) {
	_delta = val;
}

MLPPReg::RegularizationType MLPPMultiOutputLayer::get_reg() {
	return _reg;
}
void MLPPMultiOutputLayer::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;
}

real_t MLPPMultiOutputLayer::get_lambda() {
	return _lambda;
}
void MLPPMultiOutputLayer::set_lambda(const real_t val) {
	_lambda = val;
}

real_t MLPPMultiOutputLayer::get_alpha() {
	return _alpha;
}
void MLPPMultiOutputLayer::set_alpha(const real_t val) {
	_alpha = val;
}

MLPPUtilities::WeightDistributionType MLPPMultiOutputLayer::get_weight_init() {
	return _weight_init;
}
void MLPPMultiOutputLayer::set_weight_init(const MLPPUtilities::WeightDistributionType val) {
	_weight_init = val;
}

void MLPPMultiOutputLayer::forward_pass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	_z = alg.mat_vec_addv(alg.matmultm(_input, _weights), _bias);
	_a = avn.run_activation_norm_matrix(_activation, _z);
}

void MLPPMultiOutputLayer::test(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	_z_test = alg.additionm(alg.mat_vec_multv(alg.transposem(_weights), x), _bias);
	_a_test = avn.run_activation_norm_vector(_activation, _z_test);
}

MLPPMultiOutputLayer::MLPPMultiOutputLayer(int n_output, int p_n_hidden, MLPPActivation::ActivationFunction p_activation, MLPPCost::CostTypes cost, Ref<MLPPMatrix> p_input, MLPPUtilities::WeightDistributionType p_weight_init, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_n_output = n_output;
	_n_hidden = p_n_hidden;
	_activation = p_activation;
	_cost = cost;

	_input = p_input;

	// Regularization Params
	_reg = p_reg;
	_lambda = p_lambda; /* Regularization Parameter */
	_alpha = p_alpha; /* This is the controlling param for Elastic Net*/

	_weight_init = p_weight_init;

	_z.instance();
	_a.instance();

	_z_test.instance();
	_a_test.instance();

	_delta.instance();

	_weights.instance();
	_bias.instance();

	_weights->resize(Size2i(_n_hidden, _n_output));
	_bias->resize(_n_output);

	MLPPUtilities utils;

	utils.weight_initializationm(_weights, _weight_init);
	utils.bias_initializationv(_bias);
}

MLPPMultiOutputLayer::MLPPMultiOutputLayer() {
	_n_hidden = 0;
	_activation = MLPPActivation::ACTIVATION_FUNCTION_LINEAR;

	// Regularization Params
	//reg = 0;
	_lambda = 0; /* Regularization Parameter */
	_alpha = 0; /* This is the controlling param for Elastic Net*/

	_weight_init = MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT;

	_z.instance();
	_a.instance();

	_z_test.instance();
	_a_test.instance();

	_delta.instance();

	_weights.instance();
	_bias.instance();
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
