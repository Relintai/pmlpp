
#ifndef MLPP_STAT_OLD_H
#define MLPP_STAT_OLD_H

//
//  Stat.hpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#include "core/math/math_defs.h"

#include <vector>

class MLPPStatOld {
public:
	// These functions are for univariate lin reg module- not for users.
	real_t b0Estimation(const std::vector<real_t> &x, const std::vector<real_t> &y);
	real_t b1Estimation(const std::vector<real_t> &x, const std::vector<real_t> &y);

	// Statistical Functions
	real_t mean(const std::vector<real_t> &x);
	real_t median(std::vector<real_t> x);
	std::vector<real_t> mode(const std::vector<real_t> &x);
	real_t range(const std::vector<real_t> &x);
	real_t midrange(const std::vector<real_t> &x);
	real_t absAvgDeviation(const std::vector<real_t> &x);
	real_t standardDeviation(const std::vector<real_t> &x);
	real_t variance(const std::vector<real_t> &x);
	real_t covariance(const std::vector<real_t> &x, const std::vector<real_t> &y);
	real_t correlation(const std::vector<real_t> &x, const std::vector<real_t> &y);
	real_t R2(const std::vector<real_t> &x, const std::vector<real_t> &y);
	real_t chebyshevIneq(const real_t k);

	// Extras
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
};

#endif /* Stat_hpp */
