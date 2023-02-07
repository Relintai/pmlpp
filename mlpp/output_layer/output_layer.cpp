//
//  OutputLayer.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "output_layer.h"
#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

int MLPPOutputLayer::get_n_hidden() {
	return n_hidden;
}
void MLPPOutputLayer::set_n_hidden(const int val) {
	n_hidden = val;
	_initialized = false;
}

MLPPActivation::ActivationFunction MLPPOutputLayer::get_activation() {
	return activation;
}
void MLPPOutputLayer::set_activation(const MLPPActivation::ActivationFunction val) {
	activation = val;
	_initialized = false;
}

MLPPCost::CostTypes MLPPOutputLayer::get_cost() {
	return cost;
}
void MLPPOutputLayer::set_cost(const MLPPCost::CostTypes val) {
	cost = val;
	_initialized = false;
}

Ref<MLPPMatrix> MLPPOutputLayer::get_input() {
	return input;
}
void MLPPOutputLayer::set_input(const Ref<MLPPMatrix> &val) {
	input = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPOutputLayer::get_weights() {
	return weights;
}
void MLPPOutputLayer::set_weights(const Ref<MLPPVector> &val) {
	weights = val;
	_initialized = false;
}

real_t MLPPOutputLayer::MLPPOutputLayer::get_bias() {
	return bias;
}
void MLPPOutputLayer::set_bias(const real_t val) {
	bias = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPOutputLayer::get_z() {
	return z;
}
void MLPPOutputLayer::set_z(const Ref<MLPPVector> &val) {
	z = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPOutputLayer::get_a() {
	return a;
}
void MLPPOutputLayer::set_a(const Ref<MLPPVector> &val) {
	a = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPOutputLayer::get_z_test() {
	return z_test;
}
void MLPPOutputLayer::set_z_test(const Ref<MLPPVector> &val) {
	z_test = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPOutputLayer::get_a_test() {
	return a_test;
}
void MLPPOutputLayer::set_a_test(const Ref<MLPPVector> &val) {
	a_test = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPOutputLayer::get_delta() {
	return delta;
}
void MLPPOutputLayer::set_delta(const Ref<MLPPVector> &val) {
	delta = val;
	_initialized = false;
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
	_initialized = false;
}

real_t MLPPOutputLayer::get_alpha() {
	return alpha;
}
void MLPPOutputLayer::set_alpha(const real_t val) {
	alpha = val;
	_initialized = false;
}

MLPPUtilities::WeightDistributionType MLPPOutputLayer::get_weight_init() {
	return weight_init;
}
void MLPPOutputLayer::set_weight_init(const MLPPUtilities::WeightDistributionType val) {
	weight_init = val;
	_initialized = false;
}

bool MLPPOutputLayer::is_initialized() {
	return _initialized;
}
void MLPPOutputLayer::initialize() {
	if (_initialized) {
		return;
	}

	weights->resize(n_hidden);

	MLPPUtilities utils;

	utils.weight_initializationv(weights, weight_init);
	bias = utils.bias_initializationr();

	_initialized = true;
}

void MLPPOutputLayer::forward_pass() {
	if (!_initialized) {
		initialize();
	}

	MLPPLinAlg alg;
	MLPPActivation avn;

	z = alg.scalar_addnv(bias, alg.mat_vec_multv(input, weights));
	a = avn.run_activation_norm_vector(activation, z);
}

void MLPPOutputLayer::test(const Ref<MLPPVector> &x) {
	if (!_initialized) {
		initialize();
	}

	MLPPLinAlg alg;
	MLPPActivation avn;

	z_test = alg.dotv(weights, x) + bias;
	a_test = avn.run_activation_norm_vector(activation, z_test);
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

	weights->resize(n_hidden);

	MLPPUtilities utils;

	utils.weight_initializationv(weights, weight_init);
	bias = utils.bias_initializationr();

	_initialized = true;
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

	_initialized = false;
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

	ClassDB::bind_method(D_METHOD("get_cost"), &MLPPOutputLayer::get_cost);
	ClassDB::bind_method(D_METHOD("set_cost", "val"), &MLPPOutputLayer::set_cost);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cost"), "set_cost", "get_cost");

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

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPOutputLayer::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPOutputLayer::initialize);

	ClassDB::bind_method(D_METHOD("forward_pass"), &MLPPOutputLayer::forward_pass);
	ClassDB::bind_method(D_METHOD("test", "x"), &MLPPOutputLayer::test);
}
