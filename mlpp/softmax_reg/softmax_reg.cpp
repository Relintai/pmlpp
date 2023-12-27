//
//  SoftmaxReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "softmax_reg.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <random>

Ref<MLPPMatrix> MLPPSoftmaxReg::get_input_set() const {
	return _input_set;
}
void MLPPSoftmaxReg::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;
}

Ref<MLPPMatrix> MLPPSoftmaxReg::get_output_set() const {
	return _output_set;
}
void MLPPSoftmaxReg::set_output_set(const Ref<MLPPMatrix> &val) {
	_output_set = val;
}

MLPPReg::RegularizationType MLPPSoftmaxReg::get_reg() const {
	return _reg;
}
void MLPPSoftmaxReg::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;
}

real_t MLPPSoftmaxReg::get_lambda() const {
	return _lambda;
}
void MLPPSoftmaxReg::set_lambda(const real_t val) {
	_lambda = val;
}

real_t MLPPSoftmaxReg::get_alpha() const {
	return _alpha;
}
void MLPPSoftmaxReg::set_alpha(const real_t val) {
	_alpha = val;
}

Ref<MLPPMatrix> MLPPSoftmaxReg::data_y_hat_get() const {
	return _y_hat;
}
void MLPPSoftmaxReg::data_y_hat_set(const Ref<MLPPMatrix> &val) {
	_y_hat = val;
}

Ref<MLPPMatrix> MLPPSoftmaxReg::data_weights_get() const {
	return _weights;
}
void MLPPSoftmaxReg::data_weights_set(const Ref<MLPPMatrix> &val) {
	_weights = val;
}

Ref<MLPPVector> MLPPSoftmaxReg::data_bias_get() const {
	return _bias;
}
void MLPPSoftmaxReg::data_bias_set(const Ref<MLPPVector> &val) {
	_bias = val;
}

Ref<MLPPVector> MLPPSoftmaxReg::model_test(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!_input_set.is_valid() || !_output_set.is_valid(), Ref<MLPPVector>());
	ERR_FAIL_COND_V(needs_init(), Ref<MLPPVector>());

	return evaluatev(x);
}

Ref<MLPPMatrix> MLPPSoftmaxReg::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(!_input_set.is_valid() || !_output_set.is_valid(), Ref<MLPPVector>());
	ERR_FAIL_COND_V(needs_init(), Ref<MLPPMatrix>());

	return evaluatem(X);
}

void MLPPSoftmaxReg::train_gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());
	ERR_FAIL_COND(needs_init());

	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPMatrix> error = _y_hat->subn(_output_set);

		//Calculating the weight gradients
		Ref<MLPPMatrix> w_gradient = _input_set->transposen()->multn(error);

		//Weight updation
		_weights->sub(w_gradient->scalar_multiplyn(learning_rate));
		_weights = regularization.reg_weightsm(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		//real_t b_gradient = alg.sum_elements(error);

		// Bias Updation
		_bias->subtract_matrix_rows(error->scalar_multiplyn(learning_rate));

		forward_pass();

		// UI PORTION
		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_y_hat, _output_set));
			MLPPUtilities::print_ui_mb(_weights, _bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPSoftmaxReg::train_sgd(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());
	ERR_FAIL_COND(needs_init());

	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;
	int n = _input_set->size().y;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(n - 1));

	Ref<MLPPVector> input_set_row_tmp;
	input_set_row_tmp.instance();
	input_set_row_tmp->resize(_input_set->size().x);

	Ref<MLPPMatrix> y_hat_matrix_tmp;
	y_hat_matrix_tmp.instance();
	//y_hat_matrix_tmp->resize(Size2i(_input_set->size().y, 1));

	Ref<MLPPVector> output_set_row_tmp;
	output_set_row_tmp.instance();
	output_set_row_tmp->resize(_output_set->size().x);

	Ref<MLPPMatrix> output_set_row_matrix_tmp;
	output_set_row_matrix_tmp.instance();
	output_set_row_matrix_tmp->resize(Size2i(_output_set->size().x, 1));

	while (true) {
		real_t output_index = distribution(generator);

		_input_set->row_get_into_mlpp_vector(output_index, input_set_row_tmp);

		Ref<MLPPVector> y_hat = evaluatev(input_set_row_tmp);
		y_hat_matrix_tmp->resize(Size2i(y_hat->size(), 1));
		y_hat_matrix_tmp->row_set_mlpp_vector(0, y_hat);

		_output_set->row_get_into_mlpp_vector(output_index, output_set_row_tmp);
		output_set_row_matrix_tmp->row_set_mlpp_vector(0, output_set_row_tmp);

		cost_prev = cost(y_hat_matrix_tmp, output_set_row_matrix_tmp);

		// Calculating the weight gradients
		Ref<MLPPMatrix> w_gradient = input_set_row_tmp->outer_product(y_hat->subn(output_set_row_tmp));

		// Weight Updation
		_weights->sub(w_gradient->scalar_multiplyn(learning_rate));
		_weights = regularization.reg_weightsm(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		Ref<MLPPVector> b_gradient = y_hat->subn(output_set_row_tmp);

		// Bias updation
		_bias->sub(b_gradient->scalar_multiplyn(learning_rate));

		y_hat = evaluatev(output_set_row_tmp);

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat_matrix_tmp, output_set_row_matrix_tmp));
			MLPPUtilities::print_ui_mb(_weights, _bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPSoftmaxReg::train_mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());
	ERR_FAIL_COND(needs_init());

	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	int n = _input_set->size().y;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMMBatch batches = MLPPUtilities::create_mini_batchesmm(_input_set, _output_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_inputs = batches.input_sets[i];
			Ref<MLPPMatrix> current_outputs = batches.output_sets[i];

			Ref<MLPPMatrix> y_hat = evaluatem(current_inputs);
			cost_prev = cost(y_hat, current_outputs);

			Ref<MLPPMatrix> error = y_hat->subn(current_outputs);

			// Calculating the weight gradients
			Ref<MLPPMatrix> w_gradient = current_inputs->transposen()->multn(error);

			//Weight updation
			_weights->sub(w_gradient->scalar_multiplyn(learning_rate));
			_weights = regularization.reg_weightsm(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients
			_bias->subtract_matrix_rows(error->scalar_multiplyn(learning_rate));
			y_hat = evaluatem(current_inputs);

			if (ui) {
				MLPPUtilities::CostInfo(epoch, cost_prev, cost(y_hat, current_outputs));
				MLPPUtilities::print_ui_mb(_weights, _bias);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

real_t MLPPSoftmaxReg::score() {
	ERR_FAIL_COND_V(!_input_set.is_valid() || !_output_set.is_valid(), 0);
	ERR_FAIL_COND_V(needs_init(), 0);

	MLPPUtilities util;

	return util.performance_mat(_y_hat, _output_set);
}

bool MLPPSoftmaxReg::needs_init() const {
	if (!_input_set.is_valid()) {
		return true;
	}

	if (!_output_set.is_valid()) {
		return true;
	}

	int n = _input_set->size().y;
	int k = _input_set->size().x;
	int n_class = _output_set->size().x;

	if (_y_hat->size().y != n) {
		return true;
	}

	if (_weights->size() != Size2i(n_class, k)) {
		return true;
	}

	if (_bias->size() != n_class) {
		return true;
	}

	return false;
}
void MLPPSoftmaxReg::initialize() {
	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	int n = _input_set->size().y;
	int k = _input_set->size().x;
	int n_class = _output_set->size().x;

	_y_hat->resize(Size2i(0, n));

	MLPPUtilities util;

	_weights->resize(Size2i(n_class, k));
	_bias->resize(n_class);

	util.weight_initializationm(_weights);
	util.bias_initializationv(_bias);
}

MLPPSoftmaxReg::MLPPSoftmaxReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPMatrix> &p_output_set, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.instance();
	_weights.instance();
	_bias.instance();

	initialize();
}

MLPPSoftmaxReg::MLPPSoftmaxReg() {
	// Regularization Params
	_reg = MLPPReg::REGULARIZATION_TYPE_NONE;
	_lambda = 0.5;
	_alpha = 0.5; /* This is the controlling param for Elastic Net*/

	_y_hat.instance();
	_weights.instance();
	_bias.instance();
}
MLPPSoftmaxReg::~MLPPSoftmaxReg() {
}

real_t MLPPSoftmaxReg::cost(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPReg regularization;
	class MLPPCost cost;

	return cost.cross_entropym(y_hat, y) + regularization.reg_termm(_weights, _lambda, _alpha, _reg);
}

Ref<MLPPVector> MLPPSoftmaxReg::evaluatev(const Ref<MLPPVector> &x) {
	MLPPActivation avn;
	return avn.softmax_normv(_bias->addn(_weights->transposen()->mult_vec(x)));
}

Ref<MLPPMatrix> MLPPSoftmaxReg::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPActivation avn;

	return avn.softmax_normm(X->multn(_weights)->add_vecn(_bias));
}

// softmax ( wTx + b )
void MLPPSoftmaxReg::forward_pass() {
	MLPPActivation avn;

	_y_hat = avn.softmax_normm(_input_set->multn(_weights)->add_vecn(_bias));
}

void MLPPSoftmaxReg::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPSoftmaxReg::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPSoftmaxReg::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPSoftmaxReg::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPSoftmaxReg::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_reg"), &MLPPSoftmaxReg::get_reg);
	ClassDB::bind_method(D_METHOD("set_reg", "val"), &MLPPSoftmaxReg::set_reg);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reg"), "set_reg", "get_reg");

	ClassDB::bind_method(D_METHOD("get_lambda"), &MLPPSoftmaxReg::get_lambda);
	ClassDB::bind_method(D_METHOD("set_lambda", "val"), &MLPPSoftmaxReg::set_lambda);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lambda"), "set_lambda", "get_lambda");

	ClassDB::bind_method(D_METHOD("get_alpha"), &MLPPSoftmaxReg::get_alpha);
	ClassDB::bind_method(D_METHOD("set_alpha", "val"), &MLPPSoftmaxReg::set_alpha);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alpha"), "set_alpha", "get_alpha");

	ClassDB::bind_method(D_METHOD("data_y_hat_get"), &MLPPSoftmaxReg::data_y_hat_get);
	ClassDB::bind_method(D_METHOD("data_y_hat_set", "val"), &MLPPSoftmaxReg::data_y_hat_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "data_y_hat", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "data_y_hat_set", "data_y_hat_get");

	ClassDB::bind_method(D_METHOD("data_weights_get"), &MLPPSoftmaxReg::data_weights_get);
	ClassDB::bind_method(D_METHOD("data_weights_set", "val"), &MLPPSoftmaxReg::data_weights_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "data_weights", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "data_weights_set", "data_weights_get");

	ClassDB::bind_method(D_METHOD("data_bias_get"), &MLPPSoftmaxReg::data_bias_get);
	ClassDB::bind_method(D_METHOD("data_bias_set", "val"), &MLPPSoftmaxReg::data_bias_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "data_bias", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "data_bias_set", "data_bias_get");

	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPSoftmaxReg::model_test);
	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPSoftmaxReg::model_set_test);

	ClassDB::bind_method(D_METHOD("train_gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPSoftmaxReg::train_gradient_descent, false);
	ClassDB::bind_method(D_METHOD("train_sgd", "learning_rate", "max_epoch", "ui"), &MLPPSoftmaxReg::train_sgd, false);
	ClassDB::bind_method(D_METHOD("train_mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPSoftmaxReg::train_mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPSoftmaxReg::score);

	ClassDB::bind_method(D_METHOD("needs_init"), &MLPPSoftmaxReg::needs_init);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPSoftmaxReg::initialize);
}
