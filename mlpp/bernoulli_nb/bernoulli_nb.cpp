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

Ref<MLPPVector> MLPPBernoulliNB::model_set_test(const Ref<MLPPMatrix> &X) {
	Ref<MLPPVector> y_hat;
	y_hat.instance();
	y_hat->resize(X->size().y);

	Ref<MLPPVector> x_row_tmp;
	x_row_tmp.instance();
	x_row_tmp->resize(X->size().x);

	for (int i = 0; i < X->size().y; i++) {
		X->get_row_into_mlpp_vector(i, x_row_tmp);

		y_hat->set_element(i, model_test(x_row_tmp));
	}

	return y_hat;
}

real_t MLPPBernoulliNB::model_test(const Ref<MLPPVector> &x) {
	real_t score_0 = 1;
	real_t score_1 = 1;

	Vector<int> found_indices;

	for (int j = 0; j < x->size(); j++) {
		for (int k = 0; k < _vocab->size(); k++) {
			if (x->get_element(j) == _vocab->get_element(k)) {
				score_0 *= _theta[0][_vocab->get_element(k)];
				score_1 *= _theta[1][_vocab->get_element(k)];

				found_indices.push_back(k);
			}
		}
	}

	for (int i = 0; i < _vocab->size(); i++) {
		bool found = false;
		for (int j = 0; j < found_indices.size(); j++) {
			if (_vocab->get_element(i) == _vocab->get_element(found_indices[j])) {
				found = true;
			}
		}
		if (!found) {
			score_0 *= 1 - _theta[0][_vocab->get_element(i)];
			score_1 *= 1 - _theta[1][_vocab->get_element(i)];
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

	return util.performance_vec(_y_hat, _output_set);
}

MLPPBernoulliNB::MLPPBernoulliNB(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_class_num = 2;

	_prior_1 = 0;
	_prior_0 = 0;

	_y_hat.instance();
	_y_hat->resize(_output_set->size());

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

	_vocab = data.vec_to_setnv(alg.flattenv(_input_set));
}

void MLPPBernoulliNB::compute_theta() {
	// Resizing theta for the sake of ease & proper access of the elements.
	_theta.resize(_class_num);

	// Setting all values in the hasmap by default to 0.
	for (int i = _class_num - 1; i >= 0; i--) {
		for (int j = 0; j < _vocab->size(); j++) {
			_theta.write[i][_vocab->get_element(j)] = 0;
		}
	}

	for (int i = 0; i < _input_set->size().y; i++) {
		for (int j = 0; j < _input_set->size().x; j++) {
			_theta.write[_output_set->get_element(i)][_input_set->get_element(i, j)]++;
		}
	}

	for (int i = 0; i < _theta.size(); i++) {
		for (uint32_t j = 0; j < _theta[i].size(); j++) {
			if (i == 0) {
				_theta.write[i][j] /= _prior_0 * _y_hat->size();
			} else {
				_theta.write[i][j] /= _prior_1 * _y_hat->size();
			}
		}
	}
}

void MLPPBernoulliNB::evaluate() {
	for (int i = 0; i < _output_set->size(); i++) {
		// Pr(B | A) * Pr(A)
		real_t score_0 = 1;
		real_t score_1 = 1;

		real_t sum = 0;
		for (int ii = 0; ii < _output_set->size(); ii++) {
			if (_output_set->get_element(ii) == 1) {
				sum += 1;
			}
		}

		// Easy computation of priors, i.e. Pr(C_k)
		_prior_1 = sum / _y_hat->size();
		_prior_0 = 1 - _prior_1;

		// Evaluating Theta...
		compute_theta();

		// Evaluating the vocab set...
		compute_vocab();

		Vector<int> found_indices;

		for (int j = 0; j < _input_set->size().y; j++) {
			for (int k = 0; k < _vocab->size(); k++) {
				if (_input_set->get_element(i, j) == _vocab->get_element(k)) {
					score_0 += Math::log(static_cast<real_t>(_theta[0][_vocab->get_element(k)]));
					score_1 += Math::log(static_cast<real_t>(_theta[1][_vocab->get_element(k)]));

					found_indices.push_back(k);
				}
			}
		}

		for (int ii = 0; ii < _vocab->size(); ii++) {
			bool found = false;
			for (int j = 0; j < found_indices.size(); j++) {
				if (_vocab->get_element(ii) == _vocab->get_element(found_indices[j])) {
					found = true;
				}
			}
			if (!found) {
				score_0 += Math::log(1.0 - _theta[0][_vocab->get_element(ii)]);
				score_1 += Math::log(1.0 - _theta[1][_vocab->get_element(ii)]);
			}
		}

		score_0 += Math::log(_prior_0);
		score_1 += Math::log(_prior_1);

		score_0 = Math::exp(score_0);
		score_1 = Math::exp(score_1);

		// Assigning the traning example to a class

		if (score_0 > score_1) {
			_y_hat->set_element(i, 0);
		} else {
			_y_hat->set_element(i, 1);
		}
	}
}

void MLPPBernoulliNB::_bind_methods() {
}
