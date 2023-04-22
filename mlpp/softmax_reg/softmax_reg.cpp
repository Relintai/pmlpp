//
//  SoftmaxReg.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "softmax_reg.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <random>

Ref<MLPPMatrix> MLPPSoftmaxReg::get_input_set() {
	return _input_set;
}
void MLPPSoftmaxReg::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;

	_initialized = false;
}

Ref<MLPPMatrix> MLPPSoftmaxReg::get_output_set() {
	return _output_set;
}
void MLPPSoftmaxReg::set_output_set(const Ref<MLPPMatrix> &val) {
	_output_set = val;

	_initialized = false;
}

MLPPReg::RegularizationType MLPPSoftmaxReg::get_reg() {
	return _reg;
}
void MLPPSoftmaxReg::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;

	_initialized = false;
}

real_t MLPPSoftmaxReg::get_lambda() {
	return _lambda;
}
void MLPPSoftmaxReg::set_lambda(const real_t val) {
	_lambda = val;

	_initialized = false;
}

real_t MLPPSoftmaxReg::get_alpha() {
	return _alpha;
}
void MLPPSoftmaxReg::set_alpha(const real_t val) {
	_alpha = val;

	_initialized = false;
}

Ref<MLPPVector> MLPPSoftmaxReg::model_test(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPVector>());

	return evaluatev(x);
}

Ref<MLPPMatrix> MLPPSoftmaxReg::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPMatrix>());

	return evaluatem(X);
}

void MLPPSoftmaxReg::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		Ref<MLPPMatrix> error = alg.subtractionnm(_y_hat, _output_set);

		//Calculating the weight gradients
		Ref<MLPPMatrix> w_gradient = alg.matmultnm(alg.transposenm(_input_set), error);

		//Weight updation
		_weights = alg.subtractionnm(_weights, alg.scalar_multiplynm(learning_rate, w_gradient));
		_weights = regularization.reg_weightsm(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		//real_t b_gradient = alg.sum_elements(error);

		// Bias Updation
		_bias = alg.subtract_matrix_rows(_bias, alg.scalar_multiplynm(learning_rate, error));

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

void MLPPSoftmaxReg::sgd(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(_n - 1));

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

		_input_set->get_row_into_mlpp_vector(output_index, input_set_row_tmp);

		Ref<MLPPVector> y_hat = evaluatev(input_set_row_tmp);
		y_hat_matrix_tmp->resize(Size2i(y_hat->size(), 1));
		y_hat_matrix_tmp->set_row_mlpp_vector(0, y_hat);

		_output_set->get_row_into_mlpp_vector(output_index, output_set_row_tmp);
		output_set_row_matrix_tmp->set_row_mlpp_vector(0, output_set_row_tmp);

		cost_prev = cost(y_hat_matrix_tmp, output_set_row_matrix_tmp);

		// Calculating the weight gradients
		Ref<MLPPMatrix> w_gradient = alg.outer_product(input_set_row_tmp, alg.subtractionnv(y_hat, output_set_row_tmp));

		// Weight Updation
		_weights = alg.subtractionnm(_weights, alg.scalar_multiplynm(learning_rate, w_gradient));
		_weights = regularization.reg_weightsm(_weights, _lambda, _alpha, _reg);

		// Calculating the bias gradients
		Ref<MLPPVector> b_gradient = alg.subtractionnv(y_hat, output_set_row_tmp);

		// Bias updation
		_bias = alg.subtractionnv(_bias, alg.scalar_multiplynv(learning_rate, b_gradient));

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

void MLPPSoftmaxReg::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMMBatch batches = MLPPUtilities::create_mini_batchesmm(_input_set, _output_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_inputs = batches.input_sets[i];
			Ref<MLPPMatrix> current_outputs = batches.output_sets[i];

			Ref<MLPPMatrix> y_hat = evaluatem(current_inputs);
			cost_prev = cost(y_hat, current_outputs);

			Ref<MLPPMatrix> error = alg.subtractionnm(y_hat, current_outputs);

			// Calculating the weight gradients
			Ref<MLPPMatrix> w_gradient = alg.matmultnm(alg.transposenm(current_inputs), error);

			//Weight updation
			_weights = alg.subtractionnm(_weights, alg.scalar_multiplynm(learning_rate, w_gradient));
			_weights = regularization.reg_weightsm(_weights, _lambda, _alpha, _reg);

			// Calculating the bias gradients
			_bias = alg.subtract_matrix_rows(_bias, alg.scalar_multiplynm(learning_rate, error));
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
	ERR_FAIL_COND_V(!_initialized, 0);

	MLPPUtilities util;

	return util.performance_mat(_y_hat, _output_set);
}

void MLPPSoftmaxReg::save(const String &file_name) {
	ERR_FAIL_COND(!_initialized);

	MLPPUtilities util;

	//util.saveParameters(file_name, _weights, _bias);
}

bool MLPPSoftmaxReg::is_initialized() {
	return _initialized;
}
void MLPPSoftmaxReg::initialize() {
	if (_initialized) {
		return;
	}

	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	_n = _input_set->size().y;
	_k = _input_set->size().x;
	_n_class = _output_set->size().x;

	_y_hat.instance();
	_y_hat->resize(Size2i(_n, 0));

	MLPPUtilities util;

	_weights.instance();
	_weights->resize(Size2i(_n_class, _k));

	_bias.instance();
	_bias->resize(_n_class);

	util.weight_initializationm(_weights);
	util.bias_initializationv(_bias);

	_initialized = true;
}

MLPPSoftmaxReg::MLPPSoftmaxReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPMatrix> &p_output_set, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;

	_n = _input_set->size().y;
	_k = _input_set->size().x;
	_n_class = _output_set->size().x;

	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	if (!_y_hat.is_valid()) {
		_y_hat.instance();
	}
	_y_hat->resize(Size2i(_n, 0));

	MLPPUtilities util;

	if (!_weights.is_valid()) {
		_weights.instance();
	}
	_weights->resize(Size2i(_n_class, _k));

	if (!_bias.is_valid()) {
		_bias.instance();
	}
	_bias->resize(_n_class);

	util.weight_initializationm(_weights);
	util.bias_initializationv(_bias);

	_initialized = true;
}

MLPPSoftmaxReg::MLPPSoftmaxReg() {
	_n = 0;
	_k = 0;
	_n_class = 0;

	// Regularization Params
	_reg = MLPPReg::REGULARIZATION_TYPE_NONE;
	_lambda = 0.5;
	_alpha = 0.5; /* This is the controlling param for Elastic Net*/

	_initialized = false;
}
MLPPSoftmaxReg::~MLPPSoftmaxReg() {
}

real_t MLPPSoftmaxReg::cost(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPReg regularization;
	class MLPPCost cost;

	return cost.cross_entropym(y_hat, y) + regularization.reg_termm(_weights, _lambda, _alpha, _reg);
}

Ref<MLPPVector> MLPPSoftmaxReg::evaluatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	return avn.softmax_normv(alg.additionnv(_bias, alg.mat_vec_multv(alg.transposenm(_weights), x)));
}

Ref<MLPPMatrix> MLPPSoftmaxReg::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	return avn.softmax_normm(alg.mat_vec_addv(alg.matmultnm(X, _weights), _bias));
}

// softmax ( wTx + b )
void MLPPSoftmaxReg::forward_pass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	_y_hat = avn.softmax_normm(alg.mat_vec_addv(alg.matmultnm(_input_set, _weights), _bias));
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

	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPSoftmaxReg::model_test);
	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPSoftmaxReg::model_set_test);

	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPSoftmaxReg::gradient_descent, false);
	ClassDB::bind_method(D_METHOD("sgd", "learning_rate", "max_epoch", "ui"), &MLPPSoftmaxReg::sgd, false);
	ClassDB::bind_method(D_METHOD("mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPSoftmaxReg::mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPSoftmaxReg::score);

	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPSoftmaxReg::save);

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPSoftmaxReg::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPSoftmaxReg::initialize);
}
