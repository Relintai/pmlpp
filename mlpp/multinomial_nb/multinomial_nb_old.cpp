//
//  MultinomialNB.cpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "multinomial_nb_old.h"

#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

#include <algorithm>
#include <iostream>
#include <random>

MLPPMultinomialNBOld::MLPPMultinomialNBOld(std::vector<std::vector<real_t>> pinputSet, std::vector<real_t> poutputSet, int pclass_num) {
	inputSet = pinputSet;
	outputSet = poutputSet;
	class_num = pclass_num;

	y_hat.resize(outputSet.size());
	Evaluate();
}

std::vector<real_t> MLPPMultinomialNBOld::modelSetTest(std::vector<std::vector<real_t>> X) {
	std::vector<real_t> y_hat;
	for (uint32_t i = 0; i < X.size(); i++) {
		y_hat.push_back(modelTest(X[i]));
	}
	return y_hat;
}

real_t MLPPMultinomialNBOld::modelTest(std::vector<real_t> x) {
	real_t score[class_num];
	computeTheta();

	for (uint32_t j = 0; j < x.size(); j++) {
		for (uint32_t k = 0; k < vocab.size(); k++) {
			if (x[j] == vocab[k]) {
				for (int p = class_num - 1; p >= 0; p--) {
					score[p] += std::log(theta[p][vocab[k]]);
				}
			}
		}
	}

	for (uint32_t i = 0; i < priors.size(); i++) {
		score[i] += std::log(priors[i]);
	}

	return std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(real_t)));
}

real_t MLPPMultinomialNBOld::score() {
	MLPPUtilities util;
	return util.performance(y_hat, outputSet);
}

void MLPPMultinomialNBOld::computeTheta() {
	// Resizing theta for the sake of ease & proper access of the elements.
	theta.resize(class_num);

	// Setting all values in the hasmap by default to 0.
	for (int i = class_num - 1; i >= 0; i--) {
		for (uint32_t j = 0; j < vocab.size(); j++) {
			theta[i][vocab[j]] = 0;
		}
	}

	for (uint32_t i = 0; i < inputSet.size(); i++) {
		for (uint32_t j = 0; j < inputSet[0].size(); j++) {
			theta[outputSet[i]][inputSet[i][j]]++;
		}
	}

	for (uint32_t i = 0; i < theta.size(); i++) {
		for (uint32_t j = 0; j < theta[i].size(); j++) {
			theta[i][j] /= priors[i] * y_hat.size();
		}
	}
}

void MLPPMultinomialNBOld::Evaluate() {
	MLPPLinAlg alg;
	for (uint32_t i = 0; i < outputSet.size(); i++) {
		// Pr(B | A) * Pr(A)
		real_t score[class_num];

		// Easy computation of priors, i.e. Pr(C_k)
		priors.resize(class_num);
		for (uint32_t ii = 0; ii < outputSet.size(); ii++) {
			priors[int(outputSet[ii])]++;
		}
		priors = alg.scalarMultiply(real_t(1) / real_t(outputSet.size()), priors);

		// Evaluating Theta...
		computeTheta();

		for (uint32_t j = 0; j < inputSet.size(); j++) {
			for (uint32_t k = 0; k < vocab.size(); k++) {
				if (inputSet[i][j] == vocab[k]) {
					for (int p = class_num - 1; p >= 0; p--) {
						score[p] += std::log(theta[i][vocab[k]]);
					}
				}
			}
		}

		for (uint32_t ii = 0; ii < priors.size(); ii++) {
			score[ii] += std::log(priors[ii]);
			score[ii] = exp(score[ii]);
		}

		for (int ii = 0; ii < 2; ii++) {
			std::cout << score[ii] << std::endl;
		}

		// Assigning the traning example's y_hat to a class
		y_hat[i] = std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(real_t)));
	}
}
