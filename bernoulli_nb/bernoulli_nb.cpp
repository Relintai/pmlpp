/*************************************************************************/
/*  bernoulli_nb.cpp                                                     */
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

#include "bernoulli_nb.h"
#include "../data/data.h"
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
		X->row_get_into_mlpp_vector(i, x_row_tmp);

		y_hat->element_set(i, model_test(x_row_tmp));
	}

	return y_hat;
}

real_t MLPPBernoulliNB::model_test(const Ref<MLPPVector> &x) {
	real_t score_0 = 1;
	real_t score_1 = 1;

	Vector<int> found_indices;

	for (int j = 0; j < x->size(); j++) {
		for (int k = 0; k < _vocab->size(); k++) {
			if (x->element_get(j) == _vocab->element_get(k)) {
				score_0 *= _theta[0][_vocab->element_get(k)];
				score_1 *= _theta[1][_vocab->element_get(k)];

				found_indices.push_back(k);
			}
		}
	}

	for (int i = 0; i < _vocab->size(); i++) {
		bool found = false;
		for (int j = 0; j < found_indices.size(); j++) {
			if (_vocab->element_get(i) == _vocab->element_get(found_indices[j])) {
				found = true;
			}
		}
		if (!found) {
			score_0 *= 1 - _theta[0][_vocab->element_get(i)];
			score_1 *= 1 - _theta[1][_vocab->element_get(i)];
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

	_vocab.instance();
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
	MLPPData data;

	_vocab = data.vec_to_setnv(_input_set->flatten());
}

void MLPPBernoulliNB::compute_theta() {
	// Resizing theta for the sake of ease & proper access of the elements.
	_theta.resize(_class_num);

	// Setting all values in the hasmap by default to 0.
	for (int i = _class_num - 1; i >= 0; i--) {
		for (int j = 0; j < _vocab->size(); j++) {
			_theta.write[i][_vocab->element_get(j)] = 0;
		}
	}

	for (int i = 0; i < _input_set->size().y; i++) {
		for (int j = 0; j < _input_set->size().x; j++) {
			_theta.write[_output_set->element_get(i)][_input_set->element_get(i, j)]++;
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
			if (_output_set->element_get(ii) == 1) {
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

		for (int j = 0; j < _input_set->size().x; j++) {
			for (int k = 0; k < _vocab->size(); k++) {
				if (_input_set->element_get(i, j) == _vocab->element_get(k)) {
					score_0 += Math::log(static_cast<real_t>(_theta[0][_vocab->element_get(k)]));
					score_1 += Math::log(static_cast<real_t>(_theta[1][_vocab->element_get(k)]));

					found_indices.push_back(k);
				}
			}
		}

		for (int ii = 0; ii < _vocab->size(); ii++) {
			bool found = false;
			for (int j = 0; j < found_indices.size(); j++) {
				if (_vocab->element_get(ii) == _vocab->element_get(found_indices[j])) {
					found = true;
				}
			}
			if (!found) {
				score_0 += Math::log(1.0 - _theta[0][_vocab->element_get(ii)]);
				score_1 += Math::log(1.0 - _theta[1][_vocab->element_get(ii)]);
			}
		}

		score_0 += Math::log(_prior_0);
		score_1 += Math::log(_prior_1);

		score_0 = Math::exp(score_0);
		score_1 = Math::exp(score_1);

		// Assigning the traning example to a class

		if (score_0 > score_1) {
			_y_hat->element_set(i, 0);
		} else {
			_y_hat->element_set(i, 1);
		}
	}
}

void MLPPBernoulliNB::_bind_methods() {
}
