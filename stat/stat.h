#ifndef MLPP_STAT_H
#define MLPP_STAT_H

/*************************************************************************/
/*  stat.h                                                               */
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

#ifdef USING_SFW
#include "sfw.h"
#else
#include "core/math/math_defs.h"

#include "core/object/reference.h"
#endif

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <vector>

class MLPPStat : public Reference {
	GDCLASS(MLPPStat, Reference);

public:
	// These functions are for univariate lin reg module- not for users.
	real_t b0_estimation(const Ref<MLPPVector> &x, const Ref<MLPPVector> &y);
	real_t b1_estimation(const Ref<MLPPVector> &x, const Ref<MLPPVector> &y);

	// Statistical Functions
	real_t median(const Ref<MLPPVector> &x);
	Ref<MLPPVector> mode(const Ref<MLPPVector> &x);
	real_t range(const Ref<MLPPVector> &x);
	real_t midrange(const Ref<MLPPVector> &x);
	real_t abs_avg_deviation(const Ref<MLPPVector> &x);
	real_t correlation(const Ref<MLPPVector> &x, const Ref<MLPPVector> &y);
	real_t r2(const Ref<MLPPVector> &x, const Ref<MLPPVector> &y);
	real_t chebyshev_ineq(const real_t k);

	real_t meanv(const Ref<MLPPVector> &x);
	real_t standard_deviationv(const Ref<MLPPVector> &x);
	real_t variancev(const Ref<MLPPVector> &x);
	real_t covariancev(const Ref<MLPPVector> &x, const Ref<MLPPVector> &y);

	// Extras
	real_t weighted_mean(const Ref<MLPPVector> &x, const Ref<MLPPVector> &weights);
	real_t geometric_mean(const Ref<MLPPVector> &x);
	real_t harmonic_mean(const Ref<MLPPVector> &x);
	real_t rms(const Ref<MLPPVector> &x);
	real_t power_mean(const Ref<MLPPVector> &x, const real_t p);
	real_t lehmer_mean(const Ref<MLPPVector> &x, const real_t p);
	real_t weighted_lehmer_mean(const Ref<MLPPVector> &x, const Ref<MLPPVector> &weights, const real_t p);
	real_t contra_harmonic_mean(const Ref<MLPPVector> &x);
	real_t heronian_mean(const real_t A, const real_t B);
	real_t heinz_mean(const real_t A, const real_t B, const real_t x);
	real_t neuman_sandor_mean(const real_t a, const real_t b);
	real_t stolarsky_mean(const real_t x, const real_t y, const real_t p);
	real_t identric_mean(const real_t x, const real_t y);
	real_t log_mean(const real_t x, const real_t y);

protected:
	static void _bind_methods();
};

#endif /* Stat_hpp */
