//
//  MLP.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "mlp.h"

#include "core/log/logger.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <iostream>
#include <random>

Ref<MLPPMatrix> MLPPMLP::get_input_set() {
	return _input_set;
}
void MLPPMLP::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;

	_initialized = false;
}

Ref<MLPPVector> MLPPMLP::get_output_set() {
	return _output_set;
}
void MLPPMLP::set_output_set(const Ref<MLPPVector> &val) {
	_output_set = val;

	_initialized = false;
}

int MLPPMLP::get_n_hidden() {
	return _n_hidden;
}
void MLPPMLP::set_n_hidden(const int val) {
	_n_hidden = val;

	_initialized = false;
}

real_t MLPPMLP::get_lambda() {
	return _lambda;
}
void MLPPMLP::set_lambda(const real_t val) {
	_lambda = val;

	_initialized = false;
}

real_t MLPPMLP::get_alpha() {
	return _alpha;
}
void MLPPMLP::set_alpha(const real_t val) {
	_alpha = val;

	_initialized = false;
}

MLPPReg::RegularizationType MLPPMLP::get_reg() {
	return _reg;
}
void MLPPMLP::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;

	_initialized = false;
}

Ref<MLPPVector> MLPPMLP::model_set_test(const Ref<MLPPMatrix> &X) {
	return evaluatem(X);
}

real_t MLPPMLP::model_test(const Ref<MLPPVector> &x) {
	return evaluatev(x);
}

void MLPPMLP::gradient_descent(real_t learning_rate, int max_epoch, bool UI) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	_y_hat->fill(0);

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		// Calculating the errors
		Ref<MLPPVector> error = alg.subtractionnv(_y_hat, _output_set);

		// Calculating the weight/bias gradients for layer 2

		Ref<MLPPVector> D2_1 = alg.mat_vec_multnv(alg.transposenm(_a2), error);

		// weights and bias updation for layer 2
		_weights2->set_from_mlpp_vector(alg.subtractionnv(_weights2, alg.scalar_multiplynv(learning_rate / static_cast<real_t>(_n), D2_1)));
		_weights2->set_from_mlpp_vector(regularization.reg_weightsv(_weights2, _lambda, _alpha, _reg));

		_bias2 -= learning_rate * alg.sum_elementsv(error) / static_cast<real_t>(_n);

		// Calculating the weight/bias for layer 1

		Ref<MLPPMatrix> D1_1 = alg.outer_product(error, _weights2);
		Ref<MLPPMatrix> D1_2 = alg.hadamard_productnm(alg.transposenm(D1_1), avn.sigmoid_derivm(_z2));
		Ref<MLPPMatrix> D1_3 = alg.matmultnm(alg.transposenm(_input_set), D1_2);

		// weight an bias updation for layer 1
		_weights1->set_from_mlpp_matrix(alg.subtractionnm(_weights1, alg.scalar_multiplynm(learning_rate / _n, D1_3)));
		_weights1->set_from_mlpp_matrix(regularization.reg_weightsm(_weights1, _lambda, _alpha, _reg));

		_bias1->set_from_mlpp_vector(alg.subtract_matrix_rowsnv(_bias1, alg.scalar_multiplynm(learning_rate / _n, D1_2)));

		forward_pass();

		// UI PORTION
		if (UI) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_y_hat, _output_set));
			PLOG_MSG("Layer 1:");
			MLPPUtilities::print_ui_mb(_weights1, _bias1);
			PLOG_MSG("Layer 2:");
			MLPPUtilities::print_ui_vb(_weights2, _bias2);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPMLP::sgd(real_t learning_rate, int max_epoch, bool UI) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
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

	Ref<MLPPVector> output_set_row_tmp;
	output_set_row_tmp.instance();
	output_set_row_tmp->resize(1);

	Ref<MLPPVector> y_hat_row_tmp;
	y_hat_row_tmp.instance();
	y_hat_row_tmp->resize(1);

	Ref<MLPPVector> lz2;
	lz2.instance();
	Ref<MLPPVector> la2;
	la2.instance();

	while (true) {
		int output_Index = distribution(generator);

		_input_set->get_row_into_mlpp_vector(output_Index, input_set_row_tmp);
		real_t output_element = _output_set->element_get(output_Index);
		output_set_row_tmp->element_set(0, output_element);

		real_t ly_hat = evaluatev(input_set_row_tmp);
		y_hat_row_tmp->element_set(0, ly_hat);
		propagatev(input_set_row_tmp, lz2, la2);
		cost_prev = cost(y_hat_row_tmp, output_set_row_tmp);
		real_t error = ly_hat - output_element;

		// Weight updation for layer 2
		Ref<MLPPVector> D2_1 = alg.scalar_multiplynv(error, la2);

		_weights2->set_from_mlpp_vector(alg.subtractionnv(_weights2, alg.scalar_multiplynv(learning_rate, D2_1)));
		_weights2->set_from_mlpp_vector(regularization.reg_weightsv(_weights2, _lambda, _alpha, _reg));

		// Bias updation for layer 2
		_bias2 -= learning_rate * error;

		// Weight updation for layer 1
		Ref<MLPPVector> D1_1 = alg.scalar_multiplynv(error, _weights2);
		Ref<MLPPVector> D1_2 = alg.hadamard_productnv(D1_1, avn.sigmoid_derivv(lz2));
		Ref<MLPPMatrix> D1_3 = alg.outer_product(input_set_row_tmp, D1_2);

		_weights1->set_from_mlpp_matrix(alg.subtractionnm(_weights1, alg.scalar_multiplynm(learning_rate, D1_3)));
		_weights1->set_from_mlpp_matrix(regularization.reg_weightsm(_weights1, _lambda, _alpha, _reg));
		// Bias updation for layer 1

		_bias1->set_from_mlpp_vector(alg.subtractionnv(_bias1, alg.scalar_multiplynv(learning_rate, D1_2)));

		ly_hat = evaluatev(input_set_row_tmp);

		if (UI) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost_prev);
			PLOG_MSG("Layer 1:");
			MLPPUtilities::print_ui_mb(_weights1, _bias1);
			PLOG_MSG("Layer 2:");
			MLPPUtilities::print_ui_vb(_weights2, _bias2);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPMLP::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	Ref<MLPPMatrix> lz2;
	lz2.instance();
	Ref<MLPPMatrix> la2;
	la2.instance();

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;

	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input = batches.input_sets[i];
			Ref<MLPPVector> current_output = batches.output_sets[i];

			Ref<MLPPVector> ly_hat = evaluatem(current_input);
			propagatem(current_input, lz2, la2);
			cost_prev = cost(ly_hat, current_output);

			// Calculating the errors
			Ref<MLPPVector> error = alg.subtractionnv(ly_hat, current_output);

			// Calculating the weight/bias gradients for layer 2
			Ref<MLPPVector> D2_1 = alg.mat_vec_multnv(alg.transposenm(la2), error);

			real_t lr_d_cos = learning_rate / static_cast<real_t>(current_output->size());

			// weights and bias updation for layser 2
			_weights2->set_from_mlpp_vector(alg.subtractionnv(_weights2, alg.scalar_multiplynv(lr_d_cos, D2_1)));
			_weights2->set_from_mlpp_vector(regularization.reg_weightsv(_weights2, _lambda, _alpha, _reg));

			// Calculating the bias gradients for layer 2
			real_t b_gradient = alg.sum_elementsv(error);

			// Bias Updation for layer 2
			_bias2 -= learning_rate * b_gradient / current_output->size();

			//Calculating the weight/bias for layer 1
			Ref<MLPPMatrix> D1_1 = alg.outer_product(error, _weights2);
			Ref<MLPPMatrix> D1_2 = alg.hadamard_productnm(D1_1, avn.sigmoid_derivm(lz2));
			Ref<MLPPMatrix> D1_3 = alg.matmultnm(alg.transposenm(current_input), D1_2);

			// weight an bias updation for layer 1
			_weights1->set_from_mlpp_matrix(alg.subtractionnm(_weights1, alg.scalar_multiplynm(lr_d_cos, D1_3)));
			_weights1->set_from_mlpp_matrix(regularization.reg_weightsm(_weights1, _lambda, _alpha, _reg));

			_bias1->set_from_mlpp_vector(alg.subtract_matrix_rowsnv(_bias1, alg.scalar_multiplynm(lr_d_cos, D1_2)));

			_y_hat = evaluatem(current_input);

			if (UI) {
				MLPPUtilities::CostInfo(epoch, cost_prev, cost(ly_hat, current_output));
				PLOG_MSG("Layer 1:");
				MLPPUtilities::print_ui_mb(_weights1, _bias1);
				PLOG_MSG("Layer 2:");
				MLPPUtilities::print_ui_vb(_weights2, _bias2);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

real_t MLPPMLP::score() {
	MLPPUtilities util;
	return util.performance_vec(_y_hat, _output_set);
}

void MLPPMLP::save(const String &fileName) {
	ERR_FAIL_COND(!_initialized);

	MLPPUtilities util;
	//util.saveParameters(fileName, weights1, bias1, 0, 1);
	//util.saveParameters(fileName, weights2, bias2, 1, 2);
}

bool MLPPMLP::is_initialized() {
	return _initialized;
}

void MLPPMLP::initialize() {
	if (_initialized) {
		return;
	}

	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid() || _n_hidden == 0);

	_n = _input_set->size().y;
	_k = _input_set->size().x;

	MLPPActivation avn;
	_y_hat->resize(_n);

	MLPPUtilities util;

	_weights1->resize(Size2i(_k, _n_hidden));
	_weights2->resize(_n_hidden);
	_bias1->resize(_n_hidden);

	util.weight_initializationm(_weights1);
	util.weight_initializationv(_weights2);
	util.bias_initializationv(_bias1);

	_bias2 = util.bias_initializationr();

	_z2.instance();
	_a2.instance();

	_initialized = true;
}

real_t MLPPMLP::cost(const Ref<MLPPVector> &p_y_hat, const Ref<MLPPVector> &p_y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;

	return mlpp_cost.log_lossv(p_y_hat, p_y) + regularization.reg_termv(_weights2, _lambda, _alpha, _reg) + regularization.reg_termm(_weights1, _lambda, _alpha, _reg);
}

Ref<MLPPVector> MLPPMLP::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	Ref<MLPPMatrix> pz2 = alg.mat_vec_addnm(alg.matmultnm(X, _weights1), _bias1);
	Ref<MLPPMatrix> pa2 = avn.sigmoid_normm(pz2);

	return avn.sigmoid_normv(alg.scalar_addnv(_bias2, alg.mat_vec_multnv(pa2, _weights2)));
}

void MLPPMLP::propagatem(const Ref<MLPPMatrix> &X, Ref<MLPPMatrix> z2_out, Ref<MLPPMatrix> a2_out) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	z2_out->set_from_mlpp_matrix(alg.mat_vec_addnm(alg.matmultnm(X, _weights1), _bias1));
	a2_out->set_from_mlpp_matrix(avn.sigmoid_normm(z2_out));
}

real_t MLPPMLP::evaluatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	Ref<MLPPVector> pz2 = alg.additionnv(alg.mat_vec_multnv(alg.transposenm(_weights1), x), _bias1);
	Ref<MLPPVector> pa2 = avn.sigmoid_normv(pz2);

	return avn.sigmoid_normr(alg.dotnv(_weights2, pa2) + _bias2);
}

void MLPPMLP::propagatev(const Ref<MLPPVector> &x, Ref<MLPPVector> z2_out, Ref<MLPPVector> a2_out) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	z2_out->set_from_mlpp_vector(alg.additionnv(alg.mat_vec_multnv(alg.transposenm(_weights1), x), _bias1));
	a2_out->set_from_mlpp_vector(avn.sigmoid_normv(z2_out));
}

void MLPPMLP::forward_pass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	_z2->set_from_mlpp_matrix(alg.mat_vec_addnm(alg.matmultnm(_input_set, _weights1), _bias1));
	_a2->set_from_mlpp_matrix(avn.sigmoid_normm(_z2));

	_y_hat->set_from_mlpp_vector(avn.sigmoid_normv(alg.scalar_addnv(_bias2, alg.mat_vec_multnv(_a2, _weights2))));
}

MLPPMLP::MLPPMLP(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, int p_n_hidden, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;

	_y_hat.instance();

	_n_hidden = p_n_hidden;
	_n = _input_set->size().y;
	_k = _input_set->size().x;
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	MLPPActivation avn;
	_y_hat->resize(_n);

	MLPPUtilities util;

	_weights1.instance();
	_weights1->resize(Size2i(_k, _n_hidden));

	_weights2.instance();
	_weights2->resize(_n_hidden);

	_bias1.instance();
	_bias1->resize(_n_hidden);

	util.weight_initializationm(_weights1);
	util.weight_initializationv(_weights2);
	util.bias_initializationv(_bias1);

	_bias2 = util.bias_initializationr();

	_z2.instance();
	_a2.instance();

	_initialized = true;
}

MLPPMLP::MLPPMLP() {
	_y_hat.instance();

	_n_hidden = 0;
	_n = 0;
	_k = 0;
	_reg = MLPPReg::REGULARIZATION_TYPE_NONE;
	_lambda = 0.5;
	_alpha = 0.5;

	_weights1.instance();
	_weights2.instance();
	_bias1.instance();

	_bias2 = 0;

	_z2.instance();
	_a2.instance();

	_initialized = false;
}

MLPPMLP::~MLPPMLP() {
}

void MLPPMLP::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPMLP::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPMLP::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPMLP::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPMLP::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_n_hidden"), &MLPPMLP::get_n_hidden);
	ClassDB::bind_method(D_METHOD("set_n_hidden", "val"), &MLPPMLP::set_n_hidden);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "n_hidden"), "set_n_hidden", "get_n_hidden");

	ClassDB::bind_method(D_METHOD("get_lambda"), &MLPPMLP::get_lambda);
	ClassDB::bind_method(D_METHOD("set_lambda", "val"), &MLPPMLP::set_lambda);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lambda"), "set_lambda", "get_lambda");

	ClassDB::bind_method(D_METHOD("get_alpha"), &MLPPMLP::get_alpha);
	ClassDB::bind_method(D_METHOD("set_alpha", "val"), &MLPPMLP::set_alpha);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alpha"), "set_alpha", "get_alpha");

	ClassDB::bind_method(D_METHOD("get_reg"), &MLPPMLP::get_reg);
	ClassDB::bind_method(D_METHOD("set_reg", "val"), &MLPPMLP::set_reg);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reg"), "set_reg", "get_reg");

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPMLP::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPMLP::initialize);

	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPMLP::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPMLP::model_test);

	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "UI"), &MLPPMLP::gradient_descent, false);
	ClassDB::bind_method(D_METHOD("sgd", "learning_rate", "max_epoch", "UI"), &MLPPMLP::sgd, false);
	ClassDB::bind_method(D_METHOD("mbgd", "learning_rate", "max_epoch", "mini_batch_size", "UI"), &MLPPMLP::mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPMLP::score);
	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPMLP::save);
}
