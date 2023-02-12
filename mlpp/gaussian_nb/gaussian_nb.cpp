//
//  GaussianNB.cpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "gaussian_nb.h"
#include "../lin_alg/lin_alg.h"
#include "../stat/stat.h"
#include "../utilities/utilities.h"

#include <algorithm>
#include <iostream>
#include <random>

/*
Ref<MLPPMatrix> MLPPGaussianNB::get_input_set() {
	return _input_set;
}
void MLPPGaussianNB::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;
}

Ref<MLPPVector> MLPPGaussianNB::get_output_set() {
	return _output_set;
}
void MLPPGaussianNB::set_output_set(const Ref<MLPPVector> &val) {
	_output_set = val;
}

int MLPPGaussianNB::get_class_num() {
	return _class_num;
}
void MLPPGaussianNB::set_class_num(const int val) {
	_class_num = val;
}
*/

std::vector<real_t> MLPPGaussianNB::model_set_test(std::vector<std::vector<real_t>> X) {
	std::vector<real_t> y_hat;
	for (uint32_t i = 0; i < X.size(); i++) {
		y_hat.push_back(model_test(X[i]));
	}
	return y_hat;
}

real_t MLPPGaussianNB::model_test(std::vector<real_t> x) {
	real_t score[_class_num];
	real_t y_hat_i = 1;

	for (int i = _class_num - 1; i >= 0; i--) {
		y_hat_i += std::log(_priors[i] * (1 / sqrt(2 * M_PI * _sigma[i] * _sigma[i])) * exp(-(x[i] * _mu[i]) * (x[i] * _mu[i]) / (2 * _sigma[i] * _sigma[i])));
		score[i] = exp(y_hat_i);
	}

	return std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(real_t)));
}

real_t MLPPGaussianNB::score() {
	MLPPUtilities util;
	return util.performance(_y_hat, _output_set);
}

bool MLPPGaussianNB::is_initialized() {
	return _initialized;
}
void MLPPGaussianNB::initialize() {
	if (_initialized) {
		return;
	}

	//ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	_initialized = true;
}

MLPPGaussianNB::MLPPGaussianNB(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, int p_class_num) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_class_num = p_class_num;

	_y_hat.resize(_output_set.size());

	evaluate();

	_initialized = true;
}

MLPPGaussianNB::MLPPGaussianNB() {
	_initialized = false;
}
MLPPGaussianNB::~MLPPGaussianNB() {
}

void MLPPGaussianNB::evaluate() {
	MLPPStat stat;
	MLPPLinAlg alg;

	// Computing mu_k_y and sigma_k_y
	_mu.resize(_class_num);
	_sigma.resize(_class_num);

	for (int i = _class_num - 1; i >= 0; i--) {
		std::vector<real_t> set;
		for (uint32_t j = 0; j < _input_set.size(); j++) {
			for (uint32_t k = 0; k < _input_set[j].size(); k++) {
				if (_output_set[j] == i) {
					set.push_back(_input_set[j][k]);
				}
			}
		}

		_mu[i] = stat.mean(set);
		_sigma[i] = stat.standardDeviation(set);
	}

	// Priors
	_priors.resize(_class_num);
	for (uint32_t i = 0; i < _output_set.size(); i++) {
		_priors[int(_output_set[i])]++;
	}
	_priors = alg.scalarMultiply(real_t(1) / real_t(_output_set.size()), _priors);

	for (uint32_t i = 0; i < _output_set.size(); i++) {
		real_t score[_class_num];
		real_t y_hat_i = 1;

		for (int j = _class_num - 1; j >= 0; j--) {
			for (uint32_t k = 0; k < _input_set[i].size(); k++) {
				y_hat_i += std::log(_priors[j] * (1 / sqrt(2 * M_PI * _sigma[j] * _sigma[j])) * exp(-(_input_set[i][k] * _mu[j]) * (_input_set[i][k] * _mu[j]) / (2 * _sigma[j] * _sigma[j])));
			}
			score[j] = exp(y_hat_i);
			std::cout << score[j] << std::endl;
		}

		_y_hat[i] = std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(real_t)));
		std::cout << std::distance(score, std::max_element(score, score + sizeof(score) / sizeof(real_t))) << std::endl;
	}
}

void MLPPGaussianNB::_bind_methods() {
	/*
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPGaussianNB::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "value"), &MLPPGaussianNB::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPGaussianNB::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "value"), &MLPPGaussianNB::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_k"), &MLPPGaussianNB::get_k);
	ClassDB::bind_method(D_METHOD("set_k", "value"), &MLPPGaussianNB::set_k);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "k"), "set_k", "get_k");

	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPGaussianNB::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPGaussianNB::model_test);
	ClassDB::bind_method(D_METHOD("score"), &MLPPGaussianNB::score);
	*/
}
