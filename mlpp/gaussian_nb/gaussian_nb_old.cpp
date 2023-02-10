//
//  GaussianNB.cpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "gaussian_nb_old.h"

#include "../lin_alg/lin_alg.h"
#include "../stat/stat.h"
#include "../utilities/utilities.h"

#include <algorithm>
#include <iostream>
#include <random>

MLPPGaussianNBOld::MLPPGaussianNBOld(std::vector<std::vector<real_t>> p_inputSet, std::vector<real_t> p_outputSet, int p_class_num) {
	inputSet = p_inputSet;
	outputSet = p_outputSet;
	class_num = p_class_num;

	y_hat.resize(outputSet.size());
	Evaluate();
}

std::vector<real_t> MLPPGaussianNBOld::modelSetTest(std::vector<std::vector<real_t>> X) {
	std::vector<real_t> y_hat;
	for (uint32_t i = 0; i < X.size(); i++) {
		y_hat.push_back(modelTest(X[i]));
	}
	return y_hat;
}

real_t MLPPGaussianNBOld::modelTest(std::vector<real_t> x) {
	real_t score[class_num];
	real_t y_hat_i = 1;
	for (int i = class_num - 1; i >= 0; i--) {
		y_hat_i += std::log(priors[i] * (1 / sqrt(2 * M_PI * sigma[i] * sigma[i])) * exp(-(x[i] * mu[i]) * (x[i] * mu[i]) / (2 * sigma[i] * sigma[i])));
		score[i] = exp(y_hat_i);
	}
	return std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(real_t)));
}

real_t MLPPGaussianNBOld::score() {
	MLPPUtilities util;
	return util.performance(y_hat, outputSet);
}

void MLPPGaussianNBOld::Evaluate() {
	MLPPStat stat;
	MLPPLinAlg alg;

	// Computing mu_k_y and sigma_k_y
	mu.resize(class_num);
	sigma.resize(class_num);
	for (int i = class_num - 1; i >= 0; i--) {
		std::vector<real_t> set;
		for (uint32_t j = 0; j < inputSet.size(); j++) {
			for (uint32_t k = 0; k < inputSet[j].size(); k++) {
				if (outputSet[j] == i) {
					set.push_back(inputSet[j][k]);
				}
			}
		}
		mu[i] = stat.mean(set);
		sigma[i] = stat.standardDeviation(set);
	}

	// Priors
	priors.resize(class_num);
	for (uint32_t i = 0; i < outputSet.size(); i++) {
		priors[int(outputSet[i])]++;
	}
	priors = alg.scalarMultiply(real_t(1) / real_t(outputSet.size()), priors);

	for (uint32_t i = 0; i < outputSet.size(); i++) {
		real_t score[class_num];
		real_t y_hat_i = 1;
		for (int j = class_num - 1; j >= 0; j--) {
			for (uint32_t k = 0; k < inputSet[i].size(); k++) {
				y_hat_i += std::log(priors[j] * (1 / sqrt(2 * M_PI * sigma[j] * sigma[j])) * exp(-(inputSet[i][k] * mu[j]) * (inputSet[i][k] * mu[j]) / (2 * sigma[j] * sigma[j])));
			}
			score[j] = exp(y_hat_i);
			std::cout << score[j] << std::endl;
		}
		y_hat[i] = std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(real_t)));
		std::cout << std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(real_t))) << std::endl;
	}
}
