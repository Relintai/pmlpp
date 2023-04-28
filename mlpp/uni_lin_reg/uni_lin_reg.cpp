//
//  UniLinReg.cpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#include "uni_lin_reg.h"

#include "../lin_alg/lin_alg.h"
#include "../stat/stat.h"

// General Multivariate Linear Regression Model
// ŷ = b0 + b1x1 + b2x2 + ... + bkxk

// Univariate Linear Regression Model
// ŷ = b0 + b1x1

Ref<MLPPVector> MLPPUniLinReg::get_input_set() const {
	return _input_set;
}
void MLPPUniLinReg::set_input_set(const Ref<MLPPVector> &val) {
	_input_set = val;
}

Ref<MLPPVector> MLPPUniLinReg::get_output_set() const {
	return _output_set;
}
void MLPPUniLinReg::set_output_set(const Ref<MLPPVector> &val) {
	_output_set = val;
}

real_t MLPPUniLinReg::get_b0() const {
	return _b0;
}
void MLPPUniLinReg::set_b0(const real_t val) {
	_b0 = val;
}

real_t MLPPUniLinReg::get_b1() const {
	return _b1;
}
void MLPPUniLinReg::set_b1(const real_t val) {
	_b1 = val;
}

void MLPPUniLinReg::fit() {
	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	MLPPStat estimator;

	_b1 = estimator.b1_estimation(_input_set, _output_set);
	_b0 = estimator.b0_estimation(_input_set, _output_set);
}

Ref<MLPPVector> MLPPUniLinReg::model_set_test(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;

	return alg.scalar_addnv(_b0, alg.scalar_multiplynv(_b1, x));
}

real_t MLPPUniLinReg::model_test(real_t x) {
	return _b0 + _b1 * x;
}

MLPPUniLinReg::MLPPUniLinReg(const Ref<MLPPVector> &p_input_set, const Ref<MLPPVector> &p_output_set) {
	_input_set = p_input_set;
	_output_set = p_output_set;

	fit();
}

MLPPUniLinReg::MLPPUniLinReg() {
	_b0 = 0;
	_b1 = 0;
}
MLPPUniLinReg::~MLPPUniLinReg() {
}

void MLPPUniLinReg::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPUniLinReg::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPUniLinReg::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPUniLinReg::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPUniLinReg::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_b0"), &MLPPUniLinReg::get_b0);
	ClassDB::bind_method(D_METHOD("set_b0", "val"), &MLPPUniLinReg::set_b0);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "b0"), "set_b0", "get_b0");

	ClassDB::bind_method(D_METHOD("get_b1"), &MLPPUniLinReg::get_b1);
	ClassDB::bind_method(D_METHOD("set_b1", "val"), &MLPPUniLinReg::set_b1);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "b1"), "set_b1", "get_b1");

	ClassDB::bind_method(D_METHOD("fit"), &MLPPUniLinReg::fit);

	ClassDB::bind_method(D_METHOD("model_set_test", "x"), &MLPPUniLinReg::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPUniLinReg::model_test);
}
