/*************************************************************************/
/*  gaussian_nb.cpp                                                      */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2022-present Péter Magyar.                              */
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

#include "gaussian_nb.h"

#include "core/math/math_defs.h"

#include "../stat/stat.h"
#include "../utilities/utilities.h"

/*
Ref<MLPPMatrix> MLPPGaussianNB::get_input_set() {
	return _input_set;
}
void MLPPGaussianNB::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;
}

Ref<MLPPVector> MLPPGaussianNB::get_output_set() {
	return _output_set;
}
void MLPPGaussianNB::set_output_set(const Ref<MLPPVector> &val) {
	_output_set = val;
}

int MLPPGaussianNB::get_class_num() {
	return _class_num;
}
void MLPPGaussianNB::set_class_num(const int val) {
	_class_num = val;
}
*/

Ref<MLPPVector> MLPPGaussianNB::model_set_test(const Ref<MLPPMatrix> &X) {
	Ref<MLPPVector> y_hat;
	y_hat.instance();
	y_hat->resize(X->size().y);

	Ref<MLPPVector> x_row_tmp;
	x_row_tmp.instance();
	x_row_tmp->resize(X->size().x);

	for (int i = 0; i < X->size().y; i++) {
		X->row_get_into_mlpp_vector(i, x_row_tmp);

		y_hat->element_set(i, model_test(x_row_tmp));
	}

	return y_hat;
}

real_t MLPPGaussianNB::model_test(const Ref<MLPPVector> &x) {
	LocalVector<real_t> score;
	score.resize(_class_num);

	real_t y_hat_i = 1;

	for (int i = _class_num - 1; i >= 0; i--) {
		real_t sigma_i = _sigma->element_get(i);
		real_t x_i = x->element_get(i);
		real_t mu_i = _mu->element_get(i);

		y_hat_i += Math::log(_priors->element_get(i) * (1 / Math::sqrt(2 * Math_PI * sigma_i * sigma_i)) * Math::exp(-(x_i * mu_i) * (x_i * mu_i) / (2 * sigma_i * sigma_i)));
		score[i] = Math::exp(y_hat_i);
	}

	real_t max_element = -Math_INF;
	int max_element_index = 0;

	for (int i = 0; i < _class_num; ++i) {
		real_t score_i = score[i];

		if (score_i > max_element) {
			max_element = score_i;
			max_element_index = i;
		}
	}

	return max_element_index;
}

real_t MLPPGaussianNB::score() {
	MLPPUtilities util;
	return util.performance_vec(_y_hat, _output_set);
}

bool MLPPGaussianNB::is_initialized() {
	return _initialized;
}
void MLPPGaussianNB::initialize() {
	if (_initialized) {
		return;
	}

	//ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	_initialized = true;
}

MLPPGaussianNB::MLPPGaussianNB(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, int p_class_num) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_class_num = p_class_num;

	_mu.instance();
	_sigma.instance();
	_priors.instance();

	_y_hat.instance();
	_y_hat->resize(_output_set->size());

	evaluate();

	_initialized = true;
}

MLPPGaussianNB::MLPPGaussianNB() {
	_initialized = false;
}
MLPPGaussianNB::~MLPPGaussianNB() {
}

void MLPPGaussianNB::evaluate() {
	MLPPStat stat;

	// Computing mu_k_y and sigma_k_y
	_mu->resize(_class_num);
	_sigma->resize(_class_num);

	Ref<MLPPVector> set_vec;
	set_vec.instance();

	for (int i = _class_num - 1; i >= 0; i--) {
		PoolRealArray set;

		for (int j = 0; j < _input_set->size().y; j++) {
			for (int k = 0; k < _input_set->size().x; k++) {
				if (_output_set->element_get(j) == i) {
					set.push_back(_input_set->element_get(j, k));
				}
			}
		}

		set_vec->set_from_pool_vector(set);

		_mu->element_set(i, stat.meanv(set_vec));
		_sigma->element_set(i, stat.standard_deviationv(set_vec));
	}

	// Priors
	_priors->resize(_class_num);
	_priors->fill(0);
	for (int i = 0; i < _output_set->size(); i++) {
		int indx = static_cast<int>(_output_set->element_get(i));
		_priors->element_set(indx, _priors->element_get(indx));
	}

	_priors->scalar_multiply(real_t(1) / real_t(_output_set->size()));

	for (int i = 0; i < _output_set->size(); i++) {
		LocalVector<real_t> score;
		score.resize(_class_num);

		real_t y_hat_i = 1;

		for (int j = _class_num - 1; j >= 0; j--) {
			for (int k = 0; k < _input_set->size().x; k++) {
				real_t sigma_j = _sigma->element_get(j);
				real_t mu_j = _mu->element_get(j);
				real_t input_set_i_k = _input_set->element_get(i, k);

				y_hat_i += Math::log(_priors->element_get(j) * (1 / Math::sqrt(2 * Math_PI * sigma_j * sigma_j)) * Math::exp(-(input_set_i_k * mu_j) * (input_set_i_k * mu_j) / (2 * sigma_j * sigma_j)));
			}

			score[j] = Math::exp(y_hat_i);
		}

		real_t max_element = -Math_INF;
		int max_element_index = 0;

		for (int ii = 0; ii < _class_num; ++ii) {
			real_t score_ii = score[ii];

			if (score_ii > max_element) {
				max_element = score_ii;
				max_element_index = ii;
			}
		}

		_y_hat->element_set(i, max_element_index);
	}
}

void MLPPGaussianNB::_bind_methods() {
	/*
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPGaussianNB::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "value"), &MLPPGaussianNB::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPGaussianNB::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "value"), &MLPPGaussianNB::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_k"), &MLPPGaussianNB::get_k);
	ClassDB::bind_method(D_METHOD("set_k", "value"), &MLPPGaussianNB::set_k);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "k"), "set_k", "get_k");

	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPGaussianNB::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPGaussianNB::model_test);
	ClassDB::bind_method(D_METHOD("score"), &MLPPGaussianNB::score);
	*/
}
