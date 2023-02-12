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

std::vector<real_t> MLPPBernoulliNB::model_set_test(std::vector<std::vector<real_t>> X) {
	std::vector<real_t> y_hat;
	for (uint32_t i = 0; i < X.size(); i++) {
		y_hat.push_back(model_test(X[i]));
	}
	return y_hat;
}

real_t MLPPBernoulliNB::model_test(std::vector<real_t> x) {
	real_t score_0 = 1;
	real_t score_1 = 1;

	std::vector<int> foundIndices;

	for (uint32_t j = 0; j < x.size(); j++) {
		for (uint32_t k = 0; k < _vocab.size(); k++) {
			if (x[j] == _vocab[k]) {
				score_0 *= _theta[0][_vocab[k]];
				score_1 *= _theta[1][_vocab[k]];

				foundIndices.push_back(k);
			}
		}
	}

	for (uint32_t i = 0; i < _vocab.size(); i++) {
		bool found = false;
		for (uint32_t j = 0; j < foundIndices.size(); j++) {
			if (_vocab[i] == _vocab[foundIndices[j]]) {
				found = true;
			}
		}
		if (!found) {
			score_0 *= 1 - _theta[0][_vocab[i]];
			score_1 *= 1 - _theta[1][_vocab[i]];
		}
	}

	score_0 *= _prior_0;
	score_1 *= _prior_1;

	// Assigning the traning example to a class

	if (score_0 > score_1) {
		return 0;
	} else {
		return 1;
	}
}

real_t MLPPBernoulliNB::score() {
	MLPPUtilities util;
	return util.performance(_y_hat, _output_set);
}

MLPPBernoulliNB::MLPPBernoulliNB(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_class_num = 2;

	_prior_1 = 0;
	_prior_0 = 0;

	_y_hat.resize(_output_set.size());
	evaluate();
}

MLPPBernoulliNB::MLPPBernoulliNB() {
	_prior_1 = 0;
	_prior_0 = 0;
}
MLPPBernoulliNB::~MLPPBernoulliNB() {
}

void MLPPBernoulliNB::compute_vocab() {
	MLPPLinAlg alg;
	MLPPData data;
	_vocab = data.vecToSet<real_t>(alg.flatten(_input_set));
}

void MLPPBernoulliNB::compute_theta() {
	// Resizing theta for the sake of ease & proper access of the elements.
	_theta.resize(_class_num);

	// Setting all values in the hasmap by default to 0.
	for (int i = _class_num - 1; i >= 0; i--) {
		for (uint32_t j = 0; j < _vocab.size(); j++) {
			_theta[i][_vocab[j]] = 0;
		}
	}

	for (uint32_t i = 0; i < _input_set.size(); i++) {
		for (uint32_t j = 0; j < _input_set[0].size(); j++) {
			_theta[_output_set[i]][_input_set[i][j]]++;
		}
	}

	for (uint32_t i = 0; i < _theta.size(); i++) {
		for (uint32_t j = 0; j < _theta[i].size(); j++) {
			if (i == 0) {
				_theta[i][j] /= _prior_0 * _y_hat.size();
			} else {
				_theta[i][j] /= _prior_1 * _y_hat.size();
			}
		}
	}
}

void MLPPBernoulliNB::evaluate() {
	for (uint32_t i = 0; i < _output_set.size(); i++) {
		// Pr(B | A) * Pr(A)
		real_t score_0 = 1;
		real_t score_1 = 1;

		real_t sum = 0;
		for (uint32_t ii = 0; ii < _output_set.size(); ii++) {
			if (_output_set[ii] == 1) {
				sum += _output_set[ii];
			}
		}

		// Easy computation of priors, i.e. Pr(C_k)
		_prior_1 = sum / _y_hat.size();
		_prior_0 = 1 - _prior_1;

		// Evaluating Theta...
		compute_theta();

		// Evaluating the vocab set...
		compute_vocab();

		std::vector<int> foundIndices;

		for (uint32_t j = 0; j < _input_set.size(); j++) {
			for (uint32_t k = 0; k < _vocab.size(); k++) {
				if (_input_set[i][j] == _vocab[k]) {
					score_0 += std::log(_theta[0][_vocab[k]]);
					score_1 += std::log(_theta[1][_vocab[k]]);

					foundIndices.push_back(k);
				}
			}
		}

		for (uint32_t ii = 0; ii < _vocab.size(); ii++) {
			bool found = false;
			for (uint32_t j = 0; j < foundIndices.size(); j++) {
				if (_vocab[ii] == _vocab[foundIndices[j]]) {
					found = true;
				}
			}
			if (!found) {
				score_0 += std::log(1 - _theta[0][_vocab[ii]]);
				score_1 += std::log(1 - _theta[1][_vocab[ii]]);
			}
		}

		score_0 += std::log(_prior_0);
		score_1 += std::log(_prior_1);

		score_0 = exp(score_0);
		score_1 = exp(score_1);

		std::cout << score_0 << std::endl;
		std::cout << score_1 << std::endl;

		// Assigning the traning example to a class

		if (score_0 > score_1) {
			_y_hat[i] = 0;
		} else {
			_y_hat[i] = 1;
		}
	}
}

void MLPPBernoulliNB::_bind_methods() {
}
