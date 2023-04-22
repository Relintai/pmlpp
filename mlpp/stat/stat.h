
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
	/*
	real_t median(std::vector<real_t> x);
	std::vector<real_t> mode(const std::vector<real_t> &x);
	real_t range(const std::vector<real_t> &x);
	real_t midrange(const std::vector<real_t> &x);
	real_t absAvgDeviation(const std::vector<real_t> &x);
	real_t correlation(const std::vector<real_t> &x, const std::vector<real_t> &y);
	real_t R2(const std::vector<real_t> &x, const std::vector<real_t> &y);
	real_t chebyshevIneq(const real_t k);
	*/

	real_t meanv(const Ref<MLPPVector> &x);
	real_t standard_deviationv(const Ref<MLPPVector> &x);
	real_t variancev(const Ref<MLPPVector> &x);
	real_t covariancev(const Ref<MLPPVector> &x, const Ref<MLPPVector> &y);

	// Extras
	/*
	real_t weightedMean(const std::vector<real_t> &x, const std::vector<real_t> &weights);
	real_t geometricMean(const std::vector<real_t> &x);
	real_t harmonicMean(const std::vector<real_t> &x);
	real_t RMS(const std::vector<real_t> &x);
	real_t powerMean(const std::vector<real_t> &x, const real_t p);
	real_t lehmerMean(const std::vector<real_t> &x, const real_t p);
	real_t weightedLehmerMean(const std::vector<real_t> &x, const std::vector<real_t> &weights, const real_t p);
	real_t contraHarmonicMean(const std::vector<real_t> &x);
	real_t heronianMean(const real_t A, const real_t B);
	real_t heinzMean(const real_t A, const real_t B, const real_t x);
	real_t neumanSandorMean(const real_t a, const real_t b);
	real_t stolarskyMean(const real_t x, const real_t y, const real_t p);
	real_t identricMean(const real_t x, const real_t y);
	real_t logMean(const real_t x, const real_t y);
	*/

protected:
	static void _bind_methods();
};

#endif /* Stat_hpp */
