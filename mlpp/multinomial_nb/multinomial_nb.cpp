//
//  MultinomialNB.cpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "multinomial_nb.h"
#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

#include <algorithm>
#include <iostream>
#include <random>

/*
Ref<MLPPMatrix> MLPPMultinomialNB::get_input_set() {
	return _input_set;
}
void MLPPMultinomialNB::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;

	_initialized = false;
}

Ref<MLPPVector> MLPPMultinomialNB::get_output_set() {
	return _output_set;
}
void MLPPMultinomialNB::set_output_set(const Ref<MLPPMatrix> &val) {
	_output_set = val;

	_initialized = false;
}

real_t MLPPMultinomialNB::get_class_num() {
	return _class_num;
}
void MLPPMultinomialNB::set_class_num(const real_t val) {
	_class_num = val;

	_initialized = false;
}
*/

std::vector<real_t> MLPPMultinomialNB::model_set_test(std::vector<std::vector<real_t>> X) {
	ERR_FAIL_COND_V(!_initialized, std::vector<real_t>());

	std::vector<real_t> y_hat;
	for (uint32_t i = 0; i < X.size(); i++) {
		y_hat.push_back(model_test(X[i]));
	}
	return y_hat;
}

real_t MLPPMultinomialNB::model_test(std::vector<real_t> x) {
	ERR_FAIL_COND_V(!_initialized, 0);

	real_t score[_class_num];

	compute_theta();

	for (uint32_t j = 0; j < x.size(); j++) {
		for (uint32_t k = 0; k < _vocab.size(); k++) {
			if (x[j] == _vocab[k]) {
				for (int p = _class_num - 1; p >= 0; p--) {
					score[p] += std::log(_theta[p][_vocab[k]]);
				}
			}
		}
	}

	for (uint32_t i = 0; i < _priors.size(); i++) {
		score[i] += std::log(_priors[i]);
	}

	return std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(real_t)));
}

real_t MLPPMultinomialNB::score() {
	ERR_FAIL_COND_V(!_initialized, 0);

	MLPPUtilities util;

	return util.performance(_y_hat, _output_set);
}

bool MLPPMultinomialNB::is_initialized() {
	return _initialized;
}
void MLPPMultinomialNB::initialize() {
	if (_initialized) {
		return;
	}

	//ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	_initialized = true;
}

MLPPMultinomialNB::MLPPMultinomialNB(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, int pclass_num) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_class_num = pclass_num;

	_y_hat.resize(_output_set.size());

	_initialized = true;

	evaluate();
}

MLPPMultinomialNB::MLPPMultinomialNB() {
	_initialized = false;
}
MLPPMultinomialNB::~MLPPMultinomialNB() {
}

void MLPPMultinomialNB::compute_theta() {
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
			_theta[i][j] /= _priors[i] * _y_hat.size();
		}
	}
}

void MLPPMultinomialNB::evaluate() {
	MLPPLinAlg alg;

	for (uint32_t i = 0; i < _output_set.size(); i++) {
		// Pr(B | A) * Pr(A)
		real_t score[_class_num];

		// Easy computation of priors, i.e. Pr(C_k)
		_priors.resize(_class_num);
		for (uint32_t ii = 0; ii < _output_set.size(); ii++) {
			_priors[int(_output_set[ii])]++;
		}

		_priors = alg.scalarMultiply(real_t(1) / real_t(_output_set.size()), _priors);

		// Evaluating Theta...
		compute_theta();

		for (uint32_t j = 0; j < _input_set.size(); j++) {
			for (uint32_t k = 0; k < _vocab.size(); k++) {
				if (_input_set[i][j] == _vocab[k]) {
					for (int p = _class_num - 1; p >= 0; p--) {
						score[p] += std::log(_theta[i][_vocab[k]]);
					}
				}
			}
		}

		for (uint32_t ii = 0; ii < _priors.size(); ii++) {
			score[ii] += std::log(_priors[ii]);
			score[ii] = exp(score[ii]);
		}

		for (int ii = 0; ii < 2; ii++) {
			std::cout << score[ii] << std::endl;
		}

		// Assigning the traning example's y_hat to a class
		_y_hat[i] = std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(real_t)));
	}
}

void MLPPMultinomialNB::_bind_methods() {
	/*
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPMultinomialNB::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPMultinomialNB::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPMultinomialNB::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPMultinomialNB::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_c"), &MLPPMultinomialNB::get_c);
	ClassDB::bind_method(D_METHOD("set_c", "val"), &MLPPMultinomialNB::set_c);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "c"), "set_c", "get_c");

	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPMultinomialNB::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPMultinomialNB::model_test);

	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPMultinomialNB::gradient_descent, false);
	ClassDB::bind_method(D_METHOD("sgd", "learning_rate", "max_epoch", "ui"), &MLPPMultinomialNB::sgd, false);
	ClassDB::bind_method(D_METHOD("mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPMultinomialNB::mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPMultinomialNB::score);

	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPMultinomialNB::save);

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPMultinomialNB::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPMultinomialNB::initialize);
	*/
}
