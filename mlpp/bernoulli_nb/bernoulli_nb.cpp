//
//  BernoulliNB.cpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "bernoulli_nb.h"
#include "../data/data.h"
#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

MLPPBernoulliNB::MLPPBernoulliNB(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet) :
		inputSet(inputSet), outputSet(outputSet), class_num(2) {
	y_hat.resize(outputSet.size());
	Evaluate();
}

std::vector<real_t> MLPPBernoulliNB::modelSetTest(std::vector<std::vector<real_t>> X) {
	std::vector<real_t> y_hat;
	for (int i = 0; i < X.size(); i++) {
		y_hat.push_back(modelTest(X[i]));
	}
	return y_hat;
}

real_t MLPPBernoulliNB::modelTest(std::vector<real_t> x) {
	real_t score_0 = 1;
	real_t score_1 = 1;

	std::vector<int> foundIndices;

	for (int j = 0; j < x.size(); j++) {
		for (int k = 0; k < vocab.size(); k++) {
			if (x[j] == vocab[k]) {
				score_0 *= theta[0][vocab[k]];
				score_1 *= theta[1][vocab[k]];

				foundIndices.push_back(k);
			}
		}
	}

	for (int i = 0; i < vocab.size(); i++) {
		bool found = false;
		for (int j = 0; j < foundIndices.size(); j++) {
			if (vocab[i] == vocab[foundIndices[j]]) {
				found = true;
			}
		}
		if (!found) {
			score_0 *= 1 - theta[0][vocab[i]];
			score_1 *= 1 - theta[1][vocab[i]];
		}
	}

	score_0 *= prior_0;
	score_1 *= prior_1;

	// Assigning the traning example to a class

	if (score_0 > score_1) {
		return 0;
	} else {
		return 1;
	}
}

real_t MLPPBernoulliNB::score() {
	MLPPUtilities   util;
	return util.performance(y_hat, outputSet);
}

void MLPPBernoulliNB::computeVocab() {
	MLPPLinAlg alg;
	MLPPData data;
	vocab = data.vecToSet<real_t>(alg.flatten(inputSet));
}

void MLPPBernoulliNB::computeTheta() {
	// Resizing theta for the sake of ease & proper access of the elements.
	theta.resize(class_num);

	// Setting all values in the hasmap by default to 0.
	for (int i = class_num - 1; i >= 0; i--) {
		for (int j = 0; j < vocab.size(); j++) {
			theta[i][vocab[j]] = 0;
		}
	}

	for (int i = 0; i < inputSet.size(); i++) {
		for (int j = 0; j < inputSet[0].size(); j++) {
			theta[outputSet[i]][inputSet[i][j]]++;
		}
	}

	for (int i = 0; i < theta.size(); i++) {
		for (int j = 0; j < theta[i].size(); j++) {
			if (i == 0) {
				theta[i][j] /= prior_0 * y_hat.size();
			} else {
				theta[i][j] /= prior_1 * y_hat.size();
			}
		}
	}
}

void MLPPBernoulliNB::Evaluate() {
	for (int i = 0; i < outputSet.size(); i++) {
		// Pr(B | A) * Pr(A)
		real_t score_0 = 1;
		real_t score_1 = 1;

		real_t sum = 0;
		for (int i = 0; i < outputSet.size(); i++) {
			if (outputSet[i] == 1) {
				sum += outputSet[i];
			}
		}

		// Easy computation of priors, i.e. Pr(C_k)
		prior_1 = sum / y_hat.size();
		prior_0 = 1 - prior_1;

		// Evaluating Theta...
		computeTheta();

		// Evaluating the vocab set...
		computeVocab();

		std::vector<int> foundIndices;

		for (int j = 0; j < inputSet.size(); j++) {
			for (int k = 0; k < vocab.size(); k++) {
				if (inputSet[i][j] == vocab[k]) {
					score_0 += std::log(theta[0][vocab[k]]);
					score_1 += std::log(theta[1][vocab[k]]);

					foundIndices.push_back(k);
				}
			}
		}

		for (int i = 0; i < vocab.size(); i++) {
			bool found = false;
			for (int j = 0; j < foundIndices.size(); j++) {
				if (vocab[i] == vocab[foundIndices[j]]) {
					found = true;
				}
			}
			if (!found) {
				score_0 += std::log(1 - theta[0][vocab[i]]);
				score_1 += std::log(1 - theta[1][vocab[i]]);
			}
		}

		score_0 += std::log(prior_0);
		score_1 += std::log(prior_1);

		score_0 = exp(score_0);
		score_1 = exp(score_1);

		std::cout << score_0 << std::endl;
		std::cout << score_1 << std::endl;

		// Assigning the traning example to a class

		if (score_0 > score_1) {
			y_hat[i] = 0;
		} else {
			y_hat[i] = 1;
		}
	}
}
