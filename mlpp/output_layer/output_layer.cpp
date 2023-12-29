/*************************************************************************/
/*  output_layer.cpp                                                     */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2023-present Péter Magyar.                              */
/* Copyright (c) 2022-2023 Marc Melikyan                                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "output_layer.h"
#include "../utilities/utilities.h"

int MLPPOutputLayer::get_n_hidden() {
	return _n_hidden;
}
void MLPPOutputLayer::set_n_hidden(const int val) {
	_n_hidden = val;
	_initialized = false;
}

MLPPActivation::ActivationFunction MLPPOutputLayer::get_activation() {
	return _activation;
}
void MLPPOutputLayer::set_activation(const MLPPActivation::ActivationFunction val) {
	_activation = val;
	_initialized = false;
}

MLPPCost::CostTypes MLPPOutputLayer::get_cost() {
	return _cost;
}
void MLPPOutputLayer::set_cost(const MLPPCost::CostTypes val) {
	_cost = val;
	_initialized = false;
}

Ref<MLPPMatrix> MLPPOutputLayer::get_input() {
	return _input;
}
void MLPPOutputLayer::set_input(const Ref<MLPPMatrix> &val) {
	_input = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPOutputLayer::get_weights() {
	return _weights;
}
void MLPPOutputLayer::set_weights(const Ref<MLPPVector> &val) {
	_weights = val;
	_initialized = false;
}

real_t MLPPOutputLayer::MLPPOutputLayer::get_bias() {
	return _bias;
}
void MLPPOutputLayer::set_bias(const real_t val) {
	_bias = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPOutputLayer::get_z() {
	return _z;
}
void MLPPOutputLayer::set_z(const Ref<MLPPVector> &val) {
	_z = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPOutputLayer::get_a() {
	return _a;
}
void MLPPOutputLayer::set_a(const Ref<MLPPVector> &val) {
	_a = val;
	_initialized = false;
}

real_t MLPPOutputLayer::get_z_test() {
	return _z_test;
}
void MLPPOutputLayer::set_z_test(const real_t val) {
	_z_test = val;
	_initialized = false;
}

real_t MLPPOutputLayer::get_a_test() {
	return _a_test;
}
void MLPPOutputLayer::set_a_test(const real_t val) {
	_a_test = val;
	_initialized = false;
}

Ref<MLPPVector> MLPPOutputLayer::get_delta() {
	return _delta;
}
void MLPPOutputLayer::set_delta(const Ref<MLPPVector> &val) {
	_delta = val;
	_initialized = false;
}

MLPPReg::RegularizationType MLPPOutputLayer::get_reg() {
	return _reg;
}
void MLPPOutputLayer::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;
}

real_t MLPPOutputLayer::get_lambda() {
	return _lambda;
}
void MLPPOutputLayer::set_lambda(const real_t val) {
	_lambda = val;
	_initialized = false;
}

real_t MLPPOutputLayer::get_alpha() {
	return _alpha;
}
void MLPPOutputLayer::set_alpha(const real_t val) {
	_alpha = val;
	_initialized = false;
}

MLPPUtilities::WeightDistributionType MLPPOutputLayer::get_weight_init() {
	return _weight_init;
}
void MLPPOutputLayer::set_weight_init(const MLPPUtilities::WeightDistributionType val) {
	_weight_init = val;
	_initialized = false;
}

bool MLPPOutputLayer::is_initialized() {
	return _initialized;
}
void MLPPOutputLayer::initialize() {
	if (_initialized) {
		return;
	}

	_weights->resize(_n_hidden);

	MLPPUtilities utils;

	utils.weight_initializationv(_weights, _weight_init);
	_bias = utils.bias_initializationr();

	_initialized = true;
}

void MLPPOutputLayer::forward_pass() {
	if (!_initialized) {
		initialize();
	}

	MLPPActivation avn;

	_z = _input->mult_vec(_weights)->scalar_addn(_bias);
	_a = avn.run_activation_norm_vector(_activation, _z);
}

void MLPPOutputLayer::test(const Ref<MLPPVector> &x) {
	if (!_initialized) {
		initialize();
	}

	MLPPActivation avn;

	_z_test = _weights->dot(x) + _bias;
	_a_test = avn.run_activation_norm_real(_activation, _z_test);
}

MLPPOutputLayer::MLPPOutputLayer(int p_n_hidden, MLPPActivation::ActivationFunction p_activation, MLPPCost::CostTypes p_cost, Ref<MLPPMatrix> p_input, MLPPUtilities::WeightDistributionType p_weight_init, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_n_hidden = p_n_hidden;
	_activation = p_activation;
	_cost = p_cost;

	_input = p_input;

	// Regularization Params
	_reg = p_reg;
	_lambda = p_lambda; /* Regularization Parameter */
	_alpha = p_alpha; /* This is the controlling param for Elastic Net*/

	_weight_init = p_weight_init;

	_z.instance();
	_a.instance();

	_z_test = 0;
	_a_test = 0;

	_delta.instance();

	_weights.instance();
	_bias = 0;

	_weights->resize(_n_hidden);

	MLPPUtilities utils;

	utils.weight_initializationv(_weights, _weight_init);
	_bias = utils.bias_initializationr();

	_initialized = true;
}

MLPPOutputLayer::MLPPOutputLayer() {
	_n_hidden = 0;
	_activation = MLPPActivation::ACTIVATION_FUNCTION_LINEAR;

	// Regularization Params
	//reg = 0;
	_lambda = 0; /* Regularization Parameter */
	_alpha = 0; /* This is the controlling param for Elastic Net*/

	_weight_init = MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT;

	_z.instance();
	_a.instance();

	_z_test = 0;
	_a_test = 0;

	_delta.instance();

	_weights.instance();
	_bias = 0;

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
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "z_test"), "set_z_test", "get_z_test");

	ClassDB::bind_method(D_METHOD("get_a_test"), &MLPPOutputLayer::get_a_test);
	ClassDB::bind_method(D_METHOD("set_a_test", "val"), &MLPPOutputLayer::set_a_test);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "a_test"), "set_a_test", "get_a_test");

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
