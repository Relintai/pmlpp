
#ifndef MLPP_STAT_H
#define MLPP_STAT_H

//
//  Stat.hpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

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
