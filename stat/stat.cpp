/*************************************************************************/
/*  stat.cpp                                                             */
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

#include "stat.h"

#include "../activation/activation.h"
#include "../data/data.h"
#include "../lin_alg/lin_alg.h"

#ifdef USING_SFW
#include "sfw.h"
#else
#include "core/containers/hash_map.h"
#endif

#include <algorithm>
#include <cmath>
#include <map>

#include <iostream>

real_t MLPPStat::b0_estimation(const Ref<MLPPVector> &x, const Ref<MLPPVector> &y) {
	ERR_FAIL_COND_V(!x.is_valid() || !y.is_valid(), 0);

	return meanv(y) - b1_estimation(x, y) * meanv(x);
}
real_t MLPPStat::b1_estimation(const Ref<MLPPVector> &x, const Ref<MLPPVector> &y) {
	ERR_FAIL_COND_V(!x.is_valid() || !y.is_valid(), 0);

	return covariancev(x, y) / variancev(x);
}

real_t MLPPStat::median(const Ref<MLPPVector> &p_x) {
	ERR_FAIL_COND_V(!p_x.is_valid(), 0);

	Ref<MLPPVector> x = p_x->duplicate_fast();

	int center = x->size() / 2;
	x->sort();

	if (x->size() % 2 == 0) {
		return (x->element_get(center - 1) + x->element_get(center)) / 2.0;
	} else {
		return x->element_get(center - 1);
	}
}

Ref<MLPPVector> MLPPStat::mode(const Ref<MLPPVector> &p_x) {
	ERR_FAIL_COND_V(!p_x.is_valid(), 0);

	MLPPData data;
	Ref<MLPPVector> x_set = data.vec_to_setnv(p_x);
	const real_t *x_set_ptr = x_set->ptr();
	int x_set_size = x_set->size();

	int x_size = p_x->size();

	const MLPPVector &x = *(p_x.ptr());
	HashMap<real_t, int> element_num;

	for (int i = 0; i < x_set_size; ++i) {
		element_num[x[i]] = 0;
	}

	for (int i = 0; i < x_size; ++i) {
		element_num[x[i]]++;
	}

	Ref<MLPPVector> rmodes;
	rmodes.instance();
	MLPPVector &modes = *(rmodes.ptr());

	real_t max_num = element_num[x_set_ptr[0]];

	for (int i = 0; i < x_set_size; ++i) {
		if (element_num[x_set_ptr[i]] > max_num) {
			max_num = element_num[x_set_ptr[i]];
			modes.clear();
			modes.push_back(x_set_ptr[i]);
		} else if (element_num[x_set_ptr[i]] == max_num) {
			modes.push_back(x_set_ptr[i]);
		}
	}

	return rmodes;
}

real_t MLPPStat::range(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!x.is_valid(), 0);

	MLPPLinAlg alg;
	return alg.maxvr(x) - alg.minvr(x);
}

real_t MLPPStat::midrange(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!x.is_valid(), 0);

	return range(x) / 2;
}

real_t MLPPStat::abs_avg_deviation(const Ref<MLPPVector> &p_x) {
	ERR_FAIL_COND_V(!p_x.is_valid(), 0);

	real_t x_mean = meanv(p_x);
	int x_size = p_x->size();
	const real_t *x_ptr = p_x->ptr();

	real_t sum = 0;
	for (int i = 0; i < x_size; ++i) {
		real_t s = x_ptr[i] - x_mean;
		sum += ABS(s);
	}

	return sum / x_size;
}

real_t MLPPStat::correlation(const Ref<MLPPVector> &x, const Ref<MLPPVector> &y) {
	ERR_FAIL_COND_V(!x.is_valid() || !y.is_valid(), 0);

	return covariancev(x, y) / (standard_deviationv(x) * standard_deviationv(y));
}

real_t MLPPStat::r2(const Ref<MLPPVector> &x, const Ref<MLPPVector> &y) {
	ERR_FAIL_COND_V(!x.is_valid() || !y.is_valid(), 0);

	return correlation(x, y) * correlation(x, y);
}

real_t MLPPStat::chebyshev_ineq(const real_t k) {
	// X may or may not belong to a Gaussian Distribution
	return 1 - 1 / (k * k);
}

real_t MLPPStat::meanv(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!x.is_valid(), 0);

	int x_size = x->size();
	const real_t *x_ptr = x->ptr();

	real_t sum = 0;
	for (int i = 0; i < x_size; ++i) {
		sum += x_ptr[i];
	}

	return sum / x_size;
}

real_t MLPPStat::standard_deviationv(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!x.is_valid(), 0);

	return Math::sqrt(variancev(x));
}

real_t MLPPStat::variancev(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!x.is_valid(), 0);

	real_t x_mean = meanv(x);

	int x_size = x->size();
	const real_t *x_ptr = x->ptr();

	real_t sum = 0;
	for (int i = 0; i < x_size; ++i) {
		real_t xi = x_ptr[i];

		sum += (xi - x_mean) * (xi - x_mean);
	}
	return sum / (x_size - 1);
}

real_t MLPPStat::covariancev(const Ref<MLPPVector> &x, const Ref<MLPPVector> &y) {
	ERR_FAIL_COND_V(!x.is_valid() || !y.is_valid(), 0);
	ERR_FAIL_COND_V(x->size() != y->size(), 0);

	real_t x_mean = meanv(x);
	real_t y_mean = meanv(y);

	int x_size = x->size();
	const real_t *x_ptr = x->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;

	for (int i = 0; i < x_size; ++i) {
		sum += (x_ptr[i] - x_mean) * (y_ptr[i] - y_mean);
	}

	return sum / (x_size - 1);
}

real_t MLPPStat::weighted_mean(const Ref<MLPPVector> &x, const Ref<MLPPVector> &weights) {
	ERR_FAIL_COND_V(!x.is_valid() || !weights.is_valid(), 0);
	ERR_FAIL_COND_V(x->size() != weights->size(), 0);

	int x_size = x->size();
	const real_t *x_ptr = x->ptr();
	const real_t *weights_ptr = weights->ptr();

	real_t sum = 0;
	real_t weights_sum = 0;
	for (int i = 0; i < x_size; ++i) {
		sum += x_ptr[i] * weights_ptr[i];
		weights_sum += weights_ptr[i];
	}
	return sum / weights_sum;
}

real_t MLPPStat::geometric_mean(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!x.is_valid(), 0);

	int x_size = x->size();
	const real_t *x_ptr = x->ptr();

	real_t product = 1;
	for (int i = 0; i < x_size; ++i) {
		product *= x_ptr[i];
	}

	return Math::pow(product, (real_t)(1.0 / x_size));
}

real_t MLPPStat::harmonic_mean(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!x.is_valid(), 0);

	int x_size = x->size();
	const real_t *x_ptr = x->ptr();

	real_t sum = 0;
	for (int i = 0; i < x_size; ++i) {
		sum += 1 / x_ptr[i];
	}
	return x_size / sum;
}

real_t MLPPStat::rms(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!x.is_valid(), 0);

	int x_size = x->size();
	const real_t *x_ptr = x->ptr();

	real_t sum = 0;
	for (int i = 0; i < x_size; ++i) {
		real_t x_i = x_ptr[i];

		sum += x_i * x_i;
	}

	return Math::sqrt(sum / x_size);
}

real_t MLPPStat::power_mean(const Ref<MLPPVector> &x, const real_t p) {
	ERR_FAIL_COND_V(!x.is_valid(), 0);

	int x_size = x->size();
	const real_t *x_ptr = x->ptr();

	real_t sum = 0;
	for (int i = 0; i < x_size; ++i) {
		sum += Math::pow(x_ptr[i], p);
	}
	return Math::pow(sum / x_size, 1 / p);
}

real_t MLPPStat::lehmer_mean(const Ref<MLPPVector> &x, const real_t p) {
	ERR_FAIL_COND_V(!x.is_valid(), 0);

	int x_size = x->size();
	const real_t *x_ptr = x->ptr();

	real_t num = 0;
	real_t den = 0;
	for (int i = 0; i < x_size; ++i) {
		num += Math::pow(x_ptr[i], p);
		den += Math::pow(x_ptr[i], p - 1);
	}
	return num / den;
}

real_t MLPPStat::weighted_lehmer_mean(const Ref<MLPPVector> &x, const Ref<MLPPVector> &weights, const real_t p) {
	ERR_FAIL_COND_V(!x.is_valid() || !weights.is_valid(), 0);
	ERR_FAIL_COND_V(x->size() != weights->size(), 0);

	int x_size = x->size();
	const real_t *x_ptr = x->ptr();
	const real_t *weights_ptr = weights->ptr();

	real_t num = 0;
	real_t den = 0;
	for (int i = 0; i < x_size; ++i) {
		num += weights_ptr[i] * Math::pow(x_ptr[i], p);
		den += weights_ptr[i] * Math::pow(x_ptr[i], p - 1);
	}
	return num / den;
}

real_t MLPPStat::heronian_mean(const real_t A, const real_t B) {
	return (A + sqrt(A * B) + B) / 3;
}

real_t MLPPStat::contra_harmonic_mean(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!x.is_valid(), 0);

	return lehmer_mean(x, 2);
}

real_t MLPPStat::heinz_mean(const real_t A, const real_t B, const real_t x) {
	return (Math::pow(A, x) * Math::pow(B, 1 - x) + Math::pow(A, 1 - x) * Math::pow(B, x)) / 2;
}

real_t MLPPStat::neuman_sandor_mean(const real_t a, const real_t b) {
	MLPPActivation avn;
	return (a - b) / 2 * avn.arsinh_normr((a - b) / (a + b));
}

real_t MLPPStat::stolarsky_mean(const real_t x, const real_t y, const real_t p) {
	if (x == y) {
		return x;
	}
	return Math::pow((Math::pow(x, p) - Math::pow(y, p)) / (p * (x - y)), 1 / (p - 1));
}

real_t MLPPStat::identric_mean(const real_t x, const real_t y) {
	if (x == y) {
		return x;
	}
	return (1 / M_E) * Math::pow(Math::pow(x, x) / Math::pow(y, y), 1 / (x - y));
}

real_t MLPPStat::log_mean(const real_t x, const real_t y) {
	if (x == y) {
		return x;
	}
	return (y - x) / (log(y) - Math::log(x));
}

void MLPPStat::_bind_methods() {
	ClassDB::bind_method(D_METHOD("b0_estimation", "x", "y"), &MLPPStat::b0_estimation);
	ClassDB::bind_method(D_METHOD("b1_estimation", "x", "y"), &MLPPStat::b1_estimation);

	ClassDB::bind_method(D_METHOD("median", "x"), &MLPPStat::median);
	ClassDB::bind_method(D_METHOD("mode", "x"), &MLPPStat::mode);
	ClassDB::bind_method(D_METHOD("range", "x"), &MLPPStat::range);
	ClassDB::bind_method(D_METHOD("midrange", "x"), &MLPPStat::midrange);
	ClassDB::bind_method(D_METHOD("abs_avg_deviation", "x"), &MLPPStat::abs_avg_deviation);
	ClassDB::bind_method(D_METHOD("correlation", "x", "y"), &MLPPStat::correlation);
	ClassDB::bind_method(D_METHOD("r2", "x", "y"), &MLPPStat::r2);
	ClassDB::bind_method(D_METHOD("chebyshev_ineq", "k"), &MLPPStat::chebyshev_ineq);

	ClassDB::bind_method(D_METHOD("meanv", "x"), &MLPPStat::meanv);
	ClassDB::bind_method(D_METHOD("standard_deviationv", "x"), &MLPPStat::standard_deviationv);
	ClassDB::bind_method(D_METHOD("variancev", "x"), &MLPPStat::variancev);

	ClassDB::bind_method(D_METHOD("covariancev", "x", "y"), &MLPPStat::covariancev);

	ClassDB::bind_method(D_METHOD("weighted_mean", "x", "weights"), &MLPPStat::weighted_mean);
	ClassDB::bind_method(D_METHOD("geometric_mean", "x"), &MLPPStat::geometric_mean);
	ClassDB::bind_method(D_METHOD("harmonic_mean", "x"), &MLPPStat::harmonic_mean);
	ClassDB::bind_method(D_METHOD("rms", "x"), &MLPPStat::rms);

	ClassDB::bind_method(D_METHOD("power_mean", "x", "p"), &MLPPStat::power_mean);
	ClassDB::bind_method(D_METHOD("lehmer_mean", "x", "p"), &MLPPStat::lehmer_mean);

	ClassDB::bind_method(D_METHOD("weighted_lehmer_mean", "x", "weights", "p"), &MLPPStat::weighted_lehmer_mean);

	ClassDB::bind_method(D_METHOD("contra_harmonic_mean", "x"), &MLPPStat::contra_harmonic_mean);

	ClassDB::bind_method(D_METHOD("heronian_mean", "A", "B"), &MLPPStat::heronian_mean);
	ClassDB::bind_method(D_METHOD("heinz_mean", "A", "B", "x"), &MLPPStat::heinz_mean);
	ClassDB::bind_method(D_METHOD("neuman_sandor_mean", "a", "b"), &MLPPStat::neuman_sandor_mean);
	ClassDB::bind_method(D_METHOD("stolarsky_mean", "x", "y", "p"), &MLPPStat::stolarsky_mean);
	ClassDB::bind_method(D_METHOD("identric_mean", "x", "y"), &MLPPStat::identric_mean);
	ClassDB::bind_method(D_METHOD("log_mean", "x", "y"), &MLPPStat::log_mean);
}
