//
//  MultinomialNB.cpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "multinomial_nb.h"

#include "core/containers/local_vector.h"

#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

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

Ref<MLPPVector> MLPPMultinomialNB::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPVector>());

	Size2i x_size = X->size();

	Ref<MLPPVector> x_row_tmp;
	x_row_tmp.instance();
	x_row_tmp->resize(x_size.x);

	Ref<MLPPVector> y_hat;
	y_hat.instance();
	y_hat->resize(x_size.y);

	for (int i = 0; i < x_size.y; i++) {
		X->get_row_into_mlpp_vector(i, x_row_tmp);

		y_hat->element_set(i, model_test(x_row_tmp));
	}

	return y_hat;
}

real_t MLPPMultinomialNB::model_test(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!_initialized, 0);

	int x_size = x->size();

	LocalVector<real_t> score;
	score.resize(_class_num);

	compute_theta();

	int vocab_size = _vocab->size();

	for (int j = 0; j < x_size; j++) {
		for (int k = 0; k < vocab_size; k++) {
			real_t x_j = x->element_get(j);
			real_t vocab_k = _vocab->element_get(k);

			if (Math::is_equal_approx(x_j, vocab_k)) {
				for (int p = _class_num - 1; p >= 0; p--) {
					real_t theta_p_k = _theta[p][vocab_k];

					score[p] += Math::log(theta_p_k);
				}
			}
		}
	}

	for (int i = 0; i < _priors->size(); i++) {
		score[i] += std::log(_priors->element_get(i));
	}

	int max_index = 0;
	real_t max_element = score[0];

	for (uint32_t i = 1; i < score.size(); ++i) {
		real_t si = score[i];

		if (si > max_element) {
			max_index = i;
			max_element = si;
		}
	}

	return max_index;
}

real_t MLPPMultinomialNB::score() {
	ERR_FAIL_COND_V(!_initialized, 0);

	MLPPUtilities util;

	return util.performance_vec(_y_hat, _output_set);
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

MLPPMultinomialNB::MLPPMultinomialNB(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, int pclass_num) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_class_num = pclass_num;

	_y_hat.instance();
	_y_hat->resize(_output_set->size());

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

	int vocab_size = _vocab->size();

	// Setting all values in the hasmap by default to 0.
	for (int i = _class_num - 1; i >= 0; i--) {
		for (int j = 0; j < vocab_size; j++) {
			_theta.write[i][_vocab->element_get(j)] = 0;
		}
	}

	Size2i input_set_size = _input_set->size();

	for (int i = 0; i < input_set_size.y; i++) {
		for (int j = 0; j < input_set_size.x; j++) {
			_theta.write[_output_set->element_get(i)][_input_set->element_get(i, j)]++;
		}
	}

	for (int i = 0; i < _theta.size(); i++) {
		uint32_t theta_i_size = _theta[i].size();

		for (uint32_t j = 0; j < theta_i_size; j++) {
			_theta.write[i][j] /= _priors->element_get(i) * _y_hat->size();
		}
	}
}

void MLPPMultinomialNB::evaluate() {
	MLPPLinAlg alg;

	int output_set_size = _output_set->size();
	Size2i input_set_size = _input_set->size();

	for (int i = 0; i < output_set_size; i++) {
		// Pr(B | A) * Pr(A)
		LocalVector<real_t> score;
		score.resize(_class_num);

		// Easy computation of priors, i.e. Pr(C_k)
		_priors->resize(_class_num);
		for (int ii = 0; ii < _output_set->size(); ii++) {
			int osii = static_cast<int>(_output_set->element_get(ii));
			_priors->element_set(osii, _priors->element_get(osii) + 1);
		}

		_priors = alg.scalar_multiplynv(real_t(1) / real_t(output_set_size), _priors);

		// Evaluating Theta...
		compute_theta();

		for (int j = 0; j < input_set_size.y; j++) {
			for (int k = 0; k < _vocab->size(); k++) {
				real_t input_set_i_j = _input_set->element_get(i, j);
				real_t vocab_k = _vocab->element_get(k);

				if (Math::is_equal_approx(input_set_i_j, vocab_k)) {
					real_t theta_i_k = _theta[i][vocab_k];
					theta_i_k = Math::log(theta_i_k);

					for (int p = _class_num - 1; p >= 0; p--) {
						score[p] += theta_i_k;
					}
				}
			}
		}

		int priors_size = _priors->size();

		for (int ii = 0; ii < priors_size; ii++) {
			score[ii] += Math::log(_priors->element_get(ii));
			score[ii] = Math::exp(score[ii]);
		}

		// Assigning the traning example's y_hat to a class

		int max_index = 0;
		real_t max_element = score[0];

		for (uint32_t ii = 1; ii < score.size(); ++ii) {
			real_t si = score[ii];

			if (si > max_element) {
				max_index = ii;
				max_element = si;
			}
		}

		_y_hat->element_set(i, max_index);
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
