/*************************************************************************/
/*  uni_lin_reg.cpp                                                      */
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

#include "uni_lin_reg.h"
#include "../core/stat.h"

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

void MLPPUniLinReg::train() {
	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	MLPPStat estimator;

	_b1 = estimator.b1_estimation(_input_set, _output_set);
	_b0 = estimator.b0_estimation(_input_set, _output_set);
}

Ref<MLPPVector> MLPPUniLinReg::model_set_test(const Ref<MLPPVector> &x) {
	return x->scalar_multiplyn(_b1)->scalar_addn(_b0);
}

real_t MLPPUniLinReg::model_test(real_t x) {
	return _b0 + _b1 * x;
}

MLPPUniLinReg::MLPPUniLinReg(const Ref<MLPPVector> &p_input_set, const Ref<MLPPVector> &p_output_set) {
	_input_set = p_input_set;
	_output_set = p_output_set;

	train();
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

	ClassDB::bind_method(D_METHOD("train"), &MLPPUniLinReg::train);

	ClassDB::bind_method(D_METHOD("model_set_test", "x"), &MLPPUniLinReg::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPUniLinReg::model_test);
}
