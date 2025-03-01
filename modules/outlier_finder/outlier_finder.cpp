/*************************************************************************/
/*  outlier_finder.cpp                                                   */
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

#include "outlier_finder.h"

#include "../core/stat.h"

real_t MLPPOutlierFinder::get_threshold() {
	return _threshold;
}
void MLPPOutlierFinder::set_threshold(real_t val) {
	_threshold = val;
}

Vector<Vector<real_t>> MLPPOutlierFinder::model_set_test(const Ref<MLPPMatrix> &input_set) {
	ERR_FAIL_COND_V(!input_set.is_valid(), Vector<Vector<real_t>>());

	MLPPStat stat;

	Size2i input_set_size = input_set->size();

	Vector<Vector<real_t>> outliers;
	outliers.resize(input_set_size.y);

	Ref<MLPPVector> input_set_i_row_tmp;
	input_set_i_row_tmp.instance();
	input_set_i_row_tmp->resize(input_set_size.x);

	for (int i = 0; i < input_set_size.y; ++i) {
		input_set->row_get_into_mlpp_vector(i, input_set_i_row_tmp);
		real_t meanv = stat.meanv(input_set_i_row_tmp);
		real_t s_dev_v = stat.standard_deviationv(input_set_i_row_tmp);

		for (int j = 0; j < input_set_size.x; ++j) {
			real_t input_set_i_j = input_set->element_get(i, j);

			real_t z = (input_set_i_j - meanv) / s_dev_v;

			if (ABS(z) > _threshold) {
				outliers.write[i].push_back(input_set_i_j);
			}
		}
	}

	return outliers;
}

Array MLPPOutlierFinder::model_set_test_bind(const Ref<MLPPMatrix> &input_set) {
	Vector<Vector<real_t>> res = model_set_test(input_set);

	Array arr;

	for (int i = 0; i < res.size(); ++i) {
		//will get converted to PoolRealArray
		arr.push_back(Variant(res[i]));
	}

	return arr;
}

PoolVector2iArray MLPPOutlierFinder::model_set_test_indices(const Ref<MLPPMatrix> &input_set) {
	ERR_FAIL_COND_V(!input_set.is_valid(), PoolVector2iArray());

	MLPPStat stat;

	Size2i input_set_size = input_set->size();

	PoolVector2iArray outliers;

	Ref<MLPPVector> input_set_i_row_tmp;
	input_set_i_row_tmp.instance();
	input_set_i_row_tmp->resize(input_set_size.x);

	for (int i = 0; i < input_set_size.y; ++i) {
		input_set->row_get_into_mlpp_vector(i, input_set_i_row_tmp);
		real_t meanv = stat.meanv(input_set_i_row_tmp);
		real_t s_dev_v = stat.standard_deviationv(input_set_i_row_tmp);

		for (int j = 0; j < input_set_size.x; ++j) {
			real_t z = (input_set->element_get(i, j) - meanv) / s_dev_v;

			if (ABS(z) > _threshold) {
				outliers.push_back(Vector2i(j, i));
			}
		}
	}

	return outliers;
}

PoolRealArray MLPPOutlierFinder::model_test(const Ref<MLPPVector> &input_set) {
	ERR_FAIL_COND_V(!input_set.is_valid(), PoolRealArray());

	MLPPStat stat;
	PoolRealArray outliers;

	real_t mean = stat.meanv(input_set);
	real_t s_dev = stat.standard_deviationv(input_set);

	int input_set_size = input_set->size();
	const real_t *input_set_ptr = input_set->ptr();

	for (int i = 0; i < input_set_size; ++i) {
		real_t input_set_i = input_set_ptr[i];

		real_t z = (input_set_i - mean) / s_dev;

		if (ABS(z) > _threshold) {
			outliers.push_back(input_set_i);
		}
	}

	return outliers;
}

MLPPOutlierFinder::MLPPOutlierFinder(real_t threshold) {
	_threshold = threshold;
}

MLPPOutlierFinder::MLPPOutlierFinder() {
	_threshold = 0;
}
MLPPOutlierFinder::~MLPPOutlierFinder() {
}

void MLPPOutlierFinder::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_threshold"), &MLPPOutlierFinder::get_threshold);
	ClassDB::bind_method(D_METHOD("set_threshold", "val"), &MLPPOutlierFinder::set_threshold);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "threshold"), "set_threshold", "get_threshold");

	ClassDB::bind_method(D_METHOD("model_set_test", "input_set"), &MLPPOutlierFinder::model_set_test_bind);
	ClassDB::bind_method(D_METHOD("model_set_test_indices", "input_set"), &MLPPOutlierFinder::model_set_test_indices);

	ClassDB::bind_method(D_METHOD("model_test", "input_set"), &MLPPOutlierFinder::model_test);
}
