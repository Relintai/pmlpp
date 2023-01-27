//
//  Stat.cpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#include "stat.h"
#include "../activation/activation.h"
#include "../data/data.h"
#include "../lin_alg/lin_alg.h"
#include <algorithm>
#include <cmath>
#include <map>

#include <iostream>


real_t MLPPStat::b0Estimation(const std::vector<real_t> &x, const std::vector<real_t> &y) {
	return mean(y) - b1Estimation(x, y) * mean(x);
}

real_t MLPPStat::b1Estimation(const std::vector<real_t> &x, const std::vector<real_t> &y) {
	return covariance(x, y) / variance(x);
}

real_t MLPPStat::mean(const std::vector<real_t> &x) {
	real_t sum = 0;
	for (int i = 0; i < x.size(); i++) {
		sum += x[i];
	}
	return sum / x.size();
}

real_t MLPPStat::median(std::vector<real_t> x) {
	real_t center = real_t(x.size()) / real_t(2);
	sort(x.begin(), x.end());
	if (x.size() % 2 == 0) {
		return mean({ x[center - 1], x[center] });
	} else {
		return x[center - 1 + 0.5];
	}
}

std::vector<real_t> MLPPStat::mode(const std::vector<real_t> &x) {
	MLPPData data;
	std::vector<real_t> x_set = data.vecToSet(x);
	std::map<real_t, int> element_num;
	for (int i = 0; i < x_set.size(); i++) {
		element_num[x[i]] = 0;
	}
	for (int i = 0; i < x.size(); i++) {
		element_num[x[i]]++;
	}
	std::vector<real_t> modes;
	real_t max_num = element_num[x_set[0]];
	for (int i = 0; i < x_set.size(); i++) {
		if (element_num[x_set[i]] > max_num) {
			max_num = element_num[x_set[i]];
			modes.clear();
			modes.push_back(x_set[i]);
		} else if (element_num[x_set[i]] == max_num) {
			modes.push_back(x_set[i]);
		}
	}
	return modes;
}

real_t MLPPStat::range(const std::vector<real_t> &x) {
	MLPPLinAlg alg;
	return alg.max(x) - alg.min(x);
}

real_t MLPPStat::midrange(const std::vector<real_t> &x) {
	return range(x) / 2;
}

real_t MLPPStat::absAvgDeviation(const std::vector<real_t> &x) {
	real_t sum = 0;
	for (int i = 0; i < x.size(); i++) {
		sum += std::abs(x[i] - mean(x));
	}
	return sum / x.size();
}

real_t MLPPStat::standardDeviation(const std::vector<real_t> &x) {
	return std::sqrt(variance(x));
}

real_t MLPPStat::variance(const std::vector<real_t> &x) {
	real_t sum = 0;
	for (int i = 0; i < x.size(); i++) {
		sum += (x[i] - mean(x)) * (x[i] - mean(x));
	}
	return sum / (x.size() - 1);
}

real_t MLPPStat::covariance(const std::vector<real_t> &x, const std::vector<real_t> &y) {
	real_t sum = 0;
	for (int i = 0; i < x.size(); i++) {
		sum += (x[i] - mean(x)) * (y[i] - mean(y));
	}
	return sum / (x.size() - 1);
}

real_t MLPPStat::correlation(const std::vector<real_t> &x, const std::vector<real_t> &y) {
	return covariance(x, y) / (standardDeviation(x) * standardDeviation(y));
}

real_t MLPPStat::R2(const std::vector<real_t> &x, const std::vector<real_t> &y) {
	return correlation(x, y) * correlation(x, y);
}

real_t MLPPStat::chebyshevIneq(const real_t k) {
	// X may or may not belong to a Gaussian Distribution
	return 1 - 1 / (k * k);
}

real_t MLPPStat::weightedMean(const std::vector<real_t> &x, const std::vector<real_t> &weights) {
	real_t sum = 0;
	real_t weights_sum = 0;
	for (int i = 0; i < x.size(); i++) {
		sum += x[i] * weights[i];
		weights_sum += weights[i];
	}
	return sum / weights_sum;
}

real_t MLPPStat::geometricMean(const std::vector<real_t> &x) {
	real_t product = 1;
	for (int i = 0; i < x.size(); i++) {
		product *= x[i];
	}
	return std::pow(product, 1.0 / x.size());
}

real_t MLPPStat::harmonicMean(const std::vector<real_t> &x) {
	real_t sum = 0;
	for (int i = 0; i < x.size(); i++) {
		sum += 1 / x[i];
	}
	return x.size() / sum;
}

real_t MLPPStat::RMS(const std::vector<real_t> &x) {
	real_t sum = 0;
	for (int i = 0; i < x.size(); i++) {
		sum += x[i] * x[i];
	}
	return sqrt(sum / x.size());
}

real_t MLPPStat::powerMean(const std::vector<real_t> &x, const real_t p) {
	real_t sum = 0;
	for (int i = 0; i < x.size(); i++) {
		sum += std::pow(x[i], p);
	}
	return std::pow(sum / x.size(), 1 / p);
}

real_t MLPPStat::lehmerMean(const std::vector<real_t> &x, const real_t p) {
	real_t num = 0;
	real_t den = 0;
	for (int i = 0; i < x.size(); i++) {
		num += std::pow(x[i], p);
		den += std::pow(x[i], p - 1);
	}
	return num / den;
}

real_t MLPPStat::weightedLehmerMean(const std::vector<real_t> &x, const std::vector<real_t> &weights, const real_t p) {
	real_t num = 0;
	real_t den = 0;
	for (int i = 0; i < x.size(); i++) {
		num += weights[i] * std::pow(x[i], p);
		den += weights[i] * std::pow(x[i], p - 1);
	}
	return num / den;
}

real_t MLPPStat::heronianMean(const real_t A, const real_t B) {
	return (A + sqrt(A * B) + B) / 3;
}

real_t MLPPStat::contraHarmonicMean(const std::vector<real_t> &x) {
	return lehmerMean(x, 2);
}

real_t MLPPStat::heinzMean(const real_t A, const real_t B, const real_t x) {
	return (std::pow(A, x) * std::pow(B, 1 - x) + std::pow(A, 1 - x) * std::pow(B, x)) / 2;
}

real_t MLPPStat::neumanSandorMean(const real_t a, const real_t b) {
	MLPPActivation avn;
	return (a - b) / 2 * avn.arsinh((a - b) / (a + b));
}

real_t MLPPStat::stolarskyMean(const real_t x, const real_t y, const real_t p) {
	if (x == y) {
		return x;
	}
	return std::pow((std::pow(x, p) - std::pow(y, p)) / (p * (x - y)), 1 / (p - 1));
}

real_t MLPPStat::identricMean(const real_t x, const real_t y) {
	if (x == y) {
		return x;
	}
	return (1 / M_E) * std::pow(std::pow(x, x) / std::pow(y, y), 1 / (x - y));
}

real_t MLPPStat::logMean(const real_t x, const real_t y) {
	if (x == y) {
		return x;
	}
	return (y - x) / (log(y) - std::log(x));
}
