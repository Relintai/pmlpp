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
	return input_set;
}
void MLPPMLP::set_input_set(const Ref<MLPPMatrix> &val) {
	input_set = val;

	_initialized = false;
}

Ref<MLPPVector> MLPPMLP::get_output_set() {
	return output_set;
}
void MLPPMLP::set_output_set(const Ref<MLPPVector> &val) {
	output_set = val;

	_initialized = false;
}

int MLPPMLP::get_n_hidden() {
	return n_hidden;
}
void MLPPMLP::set_n_hidden(const int val) {
	n_hidden = val;

	_initialized = false;
}

real_t MLPPMLP::get_lambda() {
	return lambda;
}
void MLPPMLP::set_lambda(const real_t val) {
	lambda = val;

	_initialized = false;
}

real_t MLPPMLP::get_alpha() {
	return alpha;
}
void MLPPMLP::set_alpha(const real_t val) {
	alpha = val;

	_initialized = false;
}

MLPPReg::RegularizationType MLPPMLP::get_reg() {
	return reg;
}
void MLPPMLP::set_reg(const MLPPReg::RegularizationType val) {
	reg = val;

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

	y_hat->fill(0);

	forward_pass();

	while (true) {
		cost_prev = cost(y_hat, output_set);

		// Calculating the errors
		Ref<MLPPVector> error = alg.subtractionnv(y_hat, output_set);

		// Calculating the weight/bias gradients for layer 2

		Ref<MLPPVector> D2_1 = alg.mat_vec_multv(alg.transposem(a2), error);

		// weights and bias updation for layer 2
		weights2->set_from_mlpp_vector(alg.subtractionnv(weights2, alg.scalar_multiplynv(learning_rate / static_cast<real_t>(n), D2_1)));
		weights2->set_from_mlpp_vector(regularization.reg_weightsv(weights2, lambda, alpha, reg));

		bias2 -= learning_rate * alg.sum_elementsv(error) / static_cast<real_t>(n);

		// Calculating the weight/bias for layer 1

		Ref<MLPPMatrix> D1_1 = alg.outer_product(error, weights2);
		Ref<MLPPMatrix> D1_2 = alg.hadamard_productm(alg.transposem(D1_1), avn.sigmoid_derivm(z2));
		Ref<MLPPMatrix> D1_3 = alg.matmultm(alg.transposem(input_set), D1_2);

		// weight an bias updation for layer 1
		weights1->set_from_mlpp_matrix(alg.subtractionm(weights1, alg.scalar_multiplym(learning_rate / n, D1_3)));
		weights1->set_from_mlpp_matrix(regularization.reg_weightsm(weights1, lambda, alpha, reg));

		bias1->set_from_mlpp_vector(alg.subtract_matrix_rows(bias1, alg.scalar_multiplym(learning_rate / n, D1_2)));

		forward_pass();

		// UI PORTION
		if (UI) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, output_set));
			PLOG_MSG("Layer 1:");
			MLPPUtilities::print_ui_mb(weights1, bias1);
			PLOG_MSG("Layer 2:");
			MLPPUtilities::print_ui_vb(weights2, bias2);
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
	std::uniform_int_distribution<int> distribution(0, int(n - 1));

	Ref<MLPPVector> input_set_row_tmp;
	input_set_row_tmp.instance();
	input_set_row_tmp->resize(input_set->size().x);

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

		input_set->get_row_into_mlpp_vector(output_Index, input_set_row_tmp);
		real_t output_element = output_set->get_element(output_Index);
		output_set_row_tmp->set_element(0, output_element);

		real_t ly_hat = evaluatev(input_set_row_tmp);
		y_hat_row_tmp->set_element(0, ly_hat);
		propagatev(input_set_row_tmp, lz2, la2);
		cost_prev = cost(y_hat_row_tmp, output_set_row_tmp);
		real_t error = ly_hat - output_element;

		// Weight updation for layer 2
		Ref<MLPPVector> D2_1 = alg.scalar_multiplynv(error, la2);

		weights2->set_from_mlpp_vector(alg.subtractionnv(weights2, alg.scalar_multiplynv(learning_rate, D2_1)));
		weights2->set_from_mlpp_vector(regularization.reg_weightsv(weights2, lambda, alpha, reg));

		// Bias updation for layer 2
		bias2 -= learning_rate * error;

		// Weight updation for layer 1
		Ref<MLPPVector> D1_1 = alg.scalar_multiplynv(error, weights2);
		Ref<MLPPVector> D1_2 = alg.hadamard_productnv(D1_1, avn.sigmoid_derivv(lz2));
		Ref<MLPPMatrix> D1_3 = alg.outer_product(input_set_row_tmp, D1_2);

		weights1->set_from_mlpp_matrix(alg.subtractionm(weights1, alg.scalar_multiplym(learning_rate, D1_3)));
		weights1->set_from_mlpp_matrix(regularization.reg_weightsm(weights1, lambda, alpha, reg));
		// Bias updation for layer 1

		bias1->set_from_mlpp_vector(alg.subtractionnv(bias1, alg.scalar_multiplynv(learning_rate, D1_2)));

		ly_hat = evaluatev(input_set_row_tmp);

		if (UI) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost_prev);
			PLOG_MSG("Layer 1:");
			MLPPUtilities::print_ui_mb(weights1, bias1);
			PLOG_MSG("Layer 2:");
			MLPPUtilities::print_ui_vb(weights2, bias2);
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
	int n_mini_batch = n / mini_batch_size;

	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(input_set, output_set, n_mini_batch);

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
			Ref<MLPPVector> D2_1 = alg.mat_vec_multv(alg.transposem(la2), error);

			real_t lr_d_cos = learning_rate / static_cast<real_t>(current_output->size());

			// weights and bias updation for layser 2
			weights2->set_from_mlpp_vector(alg.subtractionnv(weights2, alg.scalar_multiplynv(lr_d_cos, D2_1)));
			weights2->set_from_mlpp_vector(regularization.reg_weightsv(weights2, lambda, alpha, reg));

			// Calculating the bias gradients for layer 2
			real_t b_gradient = alg.sum_elementsv(error);

			// Bias Updation for layer 2
			bias2 -= learning_rate * b_gradient / current_output->size();

			//Calculating the weight/bias for layer 1
			Ref<MLPPMatrix> D1_1 = alg.outer_product(error, weights2);
			Ref<MLPPMatrix> D1_2 = alg.hadamard_productm(D1_1, avn.sigmoid_derivm(lz2));
			Ref<MLPPMatrix> D1_3 = alg.matmultm(alg.transposem(current_input), D1_2);

			// weight an bias updation for layer 1
			weights1->set_from_mlpp_matrix(alg.subtractionm(weights1, alg.scalar_multiplym(lr_d_cos, D1_3)));
			weights1->set_from_mlpp_matrix(regularization.reg_weightsm(weights1, lambda, alpha, reg));

			bias1->set_from_mlpp_vector(alg.subtract_matrix_rows(bias1, alg.scalar_multiplym(lr_d_cos, D1_2)));

			y_hat = evaluatem(current_input);

			if (UI) {
				MLPPUtilities::CostInfo(epoch, cost_prev, cost(ly_hat, current_output));
				PLOG_MSG("Layer 1:");
				MLPPUtilities::print_ui_mb(weights1, bias1);
				PLOG_MSG("Layer 2:");
				MLPPUtilities::print_ui_vb(weights2, bias2);
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
	return util.performance_vec(y_hat, output_set);
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

	ERR_FAIL_COND(!input_set.is_valid() || !output_set.is_valid() || n_hidden == 0);

	n = input_set->size().y;
	k = input_set->size().x;

	MLPPActivation avn;
	y_hat->resize(n);

	MLPPUtilities util;

	weights1->resize(Size2i(k, n_hidden));
	weights2->resize(n_hidden);
	bias1->resize(n_hidden);

	util.weight_initializationm(weights1);
	util.weight_initializationv(weights2);
	util.bias_initializationv(bias1);

	bias2 = util.bias_initializationr();

	z2.instance();
	a2.instance();

	_initialized = true;
}

real_t MLPPMLP::cost(const Ref<MLPPVector> &p_y_hat, const Ref<MLPPVector> &p_y) {
	MLPPReg regularization;
	class MLPPCost cost;

	return cost.log_lossv(p_y_hat, p_y) + regularization.reg_termv(weights2, lambda, alpha, reg) + regularization.reg_termm(weights1, lambda, alpha, reg);
}

Ref<MLPPVector> MLPPMLP::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	Ref<MLPPMatrix> pz2 = alg.mat_vec_addv(alg.matmultm(X, weights1), bias1);
	Ref<MLPPMatrix> pa2 = avn.sigmoid_normm(pz2);

	return avn.sigmoid_normv(alg.scalar_addnv(bias2, alg.mat_vec_multv(pa2, weights2)));
}

void MLPPMLP::propagatem(const Ref<MLPPMatrix> &X, Ref<MLPPMatrix> z2_out, Ref<MLPPMatrix> a2_out) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	z2_out->set_from_mlpp_matrix(alg.mat_vec_addv(alg.matmultm(X, weights1), bias1));
	a2_out->set_from_mlpp_matrix(avn.sigmoid_normm(z2_out));
}

real_t MLPPMLP::evaluatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	Ref<MLPPVector> pz2 = alg.additionnv(alg.mat_vec_multv(alg.transposem(weights1), x), bias1);
	Ref<MLPPVector> pa2 = avn.sigmoid_normv(pz2);

	return avn.sigmoid(alg.dotv(weights2, pa2) + bias2);
}

void MLPPMLP::propagatev(const Ref<MLPPVector> &x, Ref<MLPPVector> z2_out, Ref<MLPPVector> a2_out) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	z2_out->set_from_mlpp_vector(alg.additionnv(alg.mat_vec_multv(alg.transposem(weights1), x), bias1));
	a2_out->set_from_mlpp_vector(avn.sigmoid_normv(z2_out));
}

void MLPPMLP::forward_pass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	z2->set_from_mlpp_matrix(alg.mat_vec_addv(alg.matmultm(input_set, weights1), bias1));
	a2->set_from_mlpp_matrix(avn.sigmoid_normm(z2));

	y_hat->set_from_mlpp_vector(avn.sigmoid_normv(alg.scalar_addnv(bias2, alg.mat_vec_multv(a2, weights2))));
}

MLPPMLP::MLPPMLP(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, int p_n_hidden, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	input_set = p_input_set;
	output_set = p_output_set;

	y_hat.instance();

	n_hidden = p_n_hidden;
	n = input_set->size().y;
	k = input_set->size().x;
	reg = p_reg;
	lambda = p_lambda;
	alpha = p_alpha;

	MLPPActivation avn;
	y_hat->resize(n);

	MLPPUtilities util;

	weights1.instance();
	weights1->resize(Size2i(k, n_hidden));

	weights2.instance();
	weights2->resize(n_hidden);

	bias1.instance();
	bias1->resize(n_hidden);

	util.weight_initializationm(weights1);
	util.weight_initializationv(weights2);
	util.bias_initializationv(bias1);

	bias2 = util.bias_initializationr();

	z2.instance();
	a2.instance();

	_initialized = true;
}

MLPPMLP::MLPPMLP() {
	y_hat.instance();

	n_hidden = 0;
	n = 0;
	k = 0;
	reg = MLPPReg::REGULARIZATION_TYPE_NONE;
	lambda = 0.5;
	alpha = 0.5;

	weights1.instance();
	weights2.instance();
	bias1.instance();

	bias2 = 0;

	z2.instance();
	a2.instance();

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

// =======    OLD    =======

MLPPMLPOld::MLPPMLPOld(std::vector<std::vector<real_t>> p_inputSet, std::vector<real_t> p_outputSet, int p_n_hidden, std::string p_reg, real_t p_lambda, real_t p_alpha) {
	inputSet = p_inputSet;
	outputSet = p_outputSet;
	n_hidden = p_n_hidden;
	n = p_inputSet.size();
	k = p_inputSet[0].size();
	reg = p_reg;
	lambda = p_lambda;
	alpha = p_alpha;

	MLPPActivation avn;
	y_hat.resize(n);

	weights1 = MLPPUtilities::weightInitialization(k, n_hidden);
	weights2 = MLPPUtilities::weightInitialization(n_hidden);
	bias1 = MLPPUtilities::biasInitialization(n_hidden);
	bias2 = MLPPUtilities::biasInitialization();
}

std::vector<real_t> MLPPMLPOld::modelSetTest(std::vector<std::vector<real_t>> X) {
	return Evaluate(X);
}

real_t MLPPMLPOld::modelTest(std::vector<real_t> x) {
	return Evaluate(x);
}

void MLPPMLPOld::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, outputSet);

		// Calculating the errors
		std::vector<real_t> error = alg.subtraction(y_hat, outputSet);

		// Calculating the weight/bias gradients for layer 2

		std::vector<real_t> D2_1 = alg.mat_vec_mult(alg.transpose(a2), error);

		// weights and bias updation for layer 2
		weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate / n, D2_1));
		weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

		bias2 -= learning_rate * alg.sum_elements(error) / n;

		// Calculating the weight/bias for layer 1

		std::vector<std::vector<real_t>> D1_1;
		D1_1.resize(n);

		D1_1 = alg.outerProduct(error, weights2);

		std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, true));

		std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(inputSet), D1_2);

		// weight an bias updation for layer 1
		weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate / n, D1_3));
		weights1 = regularization.regWeights(weights1, lambda, alpha, reg);

		bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate / n, D1_2));

		forwardPass();

		// UI PORTION
		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
			std::cout << "Layer 1:" << std::endl;
			MLPPUtilities::UI(weights1, bias1);
			std::cout << "Layer 2:" << std::endl;
			MLPPUtilities::UI(weights2, bias2);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPMLPOld::SGD(real_t learning_rate, int max_epoch, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	while (true) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(n - 1));
		int outputIndex = distribution(generator);

		real_t y_hat = Evaluate(inputSet[outputIndex]);
		auto propagate_result = propagate(inputSet[outputIndex]);
		auto z2 = std::get<0>(propagate_result);
		auto a2 = std::get<1>(propagate_result);
		cost_prev = Cost({ y_hat }, { outputSet[outputIndex] });
		real_t error = y_hat - outputSet[outputIndex];

		// Weight updation for layer 2
		std::vector<real_t> D2_1 = alg.scalarMultiply(error, a2);
		weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate, D2_1));
		weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

		// Bias updation for layer 2
		bias2 -= learning_rate * error;

		// Weight updation for layer 1
		std::vector<real_t> D1_1 = alg.scalarMultiply(error, weights2);
		std::vector<real_t> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, true));
		std::vector<std::vector<real_t>> D1_3 = alg.outerProduct(inputSet[outputIndex], D1_2);

		weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate, D1_3));
		weights1 = regularization.regWeights(weights1, lambda, alpha, reg);
		// Bias updation for layer 1

		bias1 = alg.subtraction(bias1, alg.scalarMultiply(learning_rate, D1_2));

		y_hat = Evaluate(inputSet[outputIndex]);
		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, Cost({ y_hat }, { outputSet[outputIndex] }));
			std::cout << "Layer 1:" << std::endl;
			MLPPUtilities::UI(weights1, bias1);
			std::cout << "Layer 2:" << std::endl;
			MLPPUtilities::UI(weights2, bias2);
		}
		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPMLPOld::MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	auto minibatches = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
	auto inputMiniBatches = std::get<0>(minibatches);
	auto outputMiniBatches = std::get<1>(minibatches);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = Evaluate(inputMiniBatches[i]);
			auto propagate_result = propagate(inputMiniBatches[i]);
			auto z2 = std::get<0>(propagate_result);
			auto a2 = std::get<1>(propagate_result);

			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			// Calculating the errors
			std::vector<real_t> error = alg.subtraction(y_hat, outputMiniBatches[i]);

			// Calculating the weight/bias gradients for layer 2

			std::vector<real_t> D2_1 = alg.mat_vec_mult(alg.transpose(a2), error);

			// weights and bias updation for layser 2
			weights2 = alg.subtraction(weights2, alg.scalarMultiply(learning_rate / outputMiniBatches[i].size(), D2_1));
			weights2 = regularization.regWeights(weights2, lambda, alpha, reg);

			// Calculating the bias gradients for layer 2
			//real_t b_gradient = alg.sum_elements(error);

			// Bias Updation for layer 2
			bias2 -= learning_rate * alg.sum_elements(error) / outputMiniBatches[i].size();

			//Calculating the weight/bias for layer 1

			std::vector<std::vector<real_t>> D1_1 = alg.outerProduct(error, weights2);

			std::vector<std::vector<real_t>> D1_2 = alg.hadamard_product(D1_1, avn.sigmoid(z2, true));

			std::vector<std::vector<real_t>> D1_3 = alg.matmult(alg.transpose(inputMiniBatches[i]), D1_2);

			// weight an bias updation for layer 1
			weights1 = alg.subtraction(weights1, alg.scalarMultiply(learning_rate / outputMiniBatches[i].size(), D1_3));
			weights1 = regularization.regWeights(weights1, lambda, alpha, reg);

			bias1 = alg.subtractMatrixRows(bias1, alg.scalarMultiply(learning_rate / outputMiniBatches[i].size(), D1_2));

			y_hat = Evaluate(inputMiniBatches[i]);

			if (UI) {
				MLPPUtilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputMiniBatches[i]));
				std::cout << "Layer 1:" << std::endl;
				MLPPUtilities::UI(weights1, bias1);
				std::cout << "Layer 2:" << std::endl;
				MLPPUtilities::UI(weights2, bias2);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

real_t MLPPMLPOld::score() {
	MLPPUtilities util;
	return util.performance(y_hat, outputSet);
}

void MLPPMLPOld::save(std::string fileName) {
	MLPPUtilities util;
	util.saveParameters(fileName, weights1, bias1, false, 1);
	util.saveParameters(fileName, weights2, bias2, true, 2);
}

real_t MLPPMLPOld::Cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPReg regularization;
	class MLPPCost cost;
	return cost.LogLoss(y_hat, y) + regularization.regTerm(weights2, lambda, alpha, reg) + regularization.regTerm(weights1, lambda, alpha, reg);
}

std::vector<real_t> MLPPMLPOld::Evaluate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);
	return avn.sigmoid(alg.scalarAdd(bias2, alg.mat_vec_mult(a2, weights2)));
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPMLPOld::propagate(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<std::vector<real_t>> z2 = alg.mat_vec_add(alg.matmult(X, weights1), bias1);
	std::vector<std::vector<real_t>> a2 = avn.sigmoid(z2);
	return { z2, a2 };
}

real_t MLPPMLPOld::Evaluate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);
	return avn.sigmoid(alg.dot(weights2, a2) + bias2);
}

std::tuple<std::vector<real_t>, std::vector<real_t>> MLPPMLPOld::propagate(std::vector<real_t> x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	std::vector<real_t> z2 = alg.addition(alg.mat_vec_mult(alg.transpose(weights1), x), bias1);
	std::vector<real_t> a2 = avn.sigmoid(z2);
	return { z2, a2 };
}

void MLPPMLPOld::forwardPass() {
	MLPPLinAlg alg;
	MLPPActivation avn;
	z2 = alg.mat_vec_add(alg.matmult(inputSet, weights1), bias1);
	a2 = avn.sigmoid(z2);
	y_hat = avn.sigmoid(alg.scalarAdd(bias2, alg.mat_vec_mult(a2, weights2)));
}
