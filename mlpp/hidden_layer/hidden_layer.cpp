//
//  HiddenLayer.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "hidden_layer.h"
#include "../activation/activation.h"

#include <iostream>
#include <random>

int MLPPHiddenLayer::get_n_hidden() const {
	return _n_hidden;
}
void MLPPHiddenLayer::set_n_hidden(const int val) {
	_n_hidden = val;
	_initialized = false;
}

MLPPActivation::ActivationFunction MLPPHiddenLayer::get_activation() const {
	return _activation;
}
void MLPPHiddenLayer::set_activation(const MLPPActivation::ActivationFunction val) {
	_activation = val;
	_initialized = false;
}

Ref<MLPPMatrix> MLPPHiddenLayer::get_input() {
	return _input;
}
void MLPPHiddenLayer::set_input(const Ref<MLPPMatrix> &val) {
	_input = val;
	_initialized = false;
}

Ref<MLPPMatrix> MLPPHiddenLayer::get_weights() {
	return _weights;
}
void MLPPHiddenLayer::set_weights(const Ref<MLPPMatrix> &val) {
	_weights = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPHiddenLayer::MLPPHiddenLayer::get_bias() {
	return _bias;
}
void MLPPHiddenLayer::set_bias(const Ref<MLPPVector> &val) {
	_bias = val;
	_initialized = false;
}

Ref<MLPPMatrix> MLPPHiddenLayer::get_z() {
	return _z;
}
void MLPPHiddenLayer::set_z(const Ref<MLPPMatrix> &val) {
	_z = val;
	_initialized = false;
}

Ref<MLPPMatrix> MLPPHiddenLayer::get_a() {
	return _a;
}
void MLPPHiddenLayer::set_a(const Ref<MLPPMatrix> &val) {
	_a = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPHiddenLayer::get_z_test() {
	return _z_test;
}
void MLPPHiddenLayer::set_z_test(const Ref<MLPPVector> &val) {
	_z_test = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPHiddenLayer::get_a_test() {
	return _a_test;
}
void MLPPHiddenLayer::set_a_test(const Ref<MLPPVector> &val) {
	_a_test = val;
	_initialized = false;
}

Ref<MLPPMatrix> MLPPHiddenLayer::get_delta() {
	return _delta;
}
void MLPPHiddenLayer::set_delta(const Ref<MLPPMatrix> &val) {
	_delta = val;
	_initialized = false;
}

MLPPReg::RegularizationType MLPPHiddenLayer::get_reg() const {
	return _reg;
}
void MLPPHiddenLayer::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;
	_initialized = false;
}

real_t MLPPHiddenLayer::get_lambda() const {
	return _lambda;
}
void MLPPHiddenLayer::set_lambda(const real_t val) {
	_lambda = val;
	_initialized = false;
}

real_t MLPPHiddenLayer::get_alpha() const {
	return _alpha;
}
void MLPPHiddenLayer::set_alpha(const real_t val) {
	_alpha = val;
	_initialized = false;
}

MLPPUtilities::WeightDistributionType MLPPHiddenLayer::get_weight_init() const {
	return _weight_init;
}
void MLPPHiddenLayer::set_weight_init(const MLPPUtilities::WeightDistributionType val) {
	_weight_init = val;
	_initialized = false;
}

bool MLPPHiddenLayer::is_initialized() {
	return _initialized;
}
void MLPPHiddenLayer::initialize() {
	if (_initialized) {
		return;
	}

	_weights->resize(Size2i(_n_hidden, _input->size().x));
	_bias->resize(_n_hidden);

	MLPPUtilities utils;

	utils.weight_initializationm(_weights, _weight_init);
	utils.bias_initializationv(_bias);

	_initialized = true;
}

void MLPPHiddenLayer::forward_pass() {
	if (!_initialized) {
		initialize();
	}

	MLPPActivation avn;

	_z->multb(_input, _weights);
	_z->add_vec(_bias);

	_a = avn.run_activation_norm_matrix(_activation, _z);
}

void MLPPHiddenLayer::test(const Ref<MLPPVector> &x) {
	if (!_initialized) {
		initialize();
	}

	MLPPActivation avn;

	_z_test = _weights->transposen()->mult_vec(x);
	_z_test->add(_bias);

	_a_test = avn.run_activation_norm_vector(_activation, _z_test);
}

MLPPHiddenLayer::MLPPHiddenLayer(int p_n_hidden, MLPPActivation::ActivationFunction p_activation, Ref<MLPPMatrix> p_input, MLPPUtilities::WeightDistributionType p_weight_init, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_n_hidden = p_n_hidden;
	_activation = p_activation;

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

	_initialized = false;

	initialize();
}

MLPPHiddenLayer::MLPPHiddenLayer() {
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

	_initialized = false;
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

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPHiddenLayer::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPHiddenLayer::initialize);

	ClassDB::bind_method(D_METHOD("forward_pass"), &MLPPHiddenLayer::forward_pass);
	ClassDB::bind_method(D_METHOD("test", "x"), &MLPPHiddenLayer::test);
}
