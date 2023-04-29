//
//  SoftmaxNet.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "softmax_net.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../data/data.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include "core/log/logger.h"

#include <random>

/*
Ref<MLPPMatrix> MLPPSoftmaxNet::get_input_set() {
	return _input_set;
}
void MLPPSoftmaxNet::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;

	_initialized = false;
}

Ref<MLPPMatrix> MLPPSoftmaxNet::get_output_set() {
	return _output_set;
}
void MLPPSoftmaxNet::set_output_set(const Ref<MLPPMatrix> &val) {
	_output_set = val;

	_initialized = false;
}

MLPPReg::RegularizationType MLPPSoftmaxNet::get_reg() {
	return _reg;
}
void MLPPSoftmaxNet::set_reg(const MLPPReg::RegularizationType val) {
	_reg = val;

	_initialized = false;
}

real_t MLPPSoftmaxNet::get_lambda() {
	return _lambda;
}
void MLPPSoftmaxNet::set_lambda(const real_t val) {
	_lambda = val;

	_initialized = false;
}

real_t MLPPSoftmaxNet::get_alpha() {
	return _alpha;
}
void MLPPSoftmaxNet::set_alpha(const real_t val) {
	_alpha = val;

	_initialized = false;
}
*/

Ref<MLPPVector> MLPPSoftmaxNet::model_test(const Ref<MLPPVector> &x) {
	return evaluatev(x);
}

Ref<MLPPMatrix> MLPPSoftmaxNet::model_set_test(const Ref<MLPPMatrix> &X) {
	return evaluatem(X);
}

void MLPPSoftmaxNet::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	MLPPActivation avn;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set);

		// Calculating the errors
		Ref<MLPPMatrix> error = _y_hat->subn(_output_set);

		// Calculating the weight/bias gradients for layer 2
		Ref<MLPPMatrix> D2_1 = _a2->transposen()->multn(error);

		// weights and bias updation for layer 2
		_weights2->sub(D2_1->scalar_multiplyn(learning_rate));
		_weights2 = regularization.reg_weightsm(_weights2, _lambda, _alpha, _reg);

		_bias2->subtract_matrix_rows(error->scalar_multiplyn(learning_rate));

		//Calculating the weight/bias for layer 1
		Ref<MLPPMatrix> D1_1 = error->multn(_weights2->transposen());
		Ref<MLPPMatrix> D1_2 = D1_1->hadamard_productn(avn.sigmoid_derivm(_z2));
		Ref<MLPPMatrix> D1_3 = _input_set->transposen()->multn(D1_2);

		// weight an bias updation for layer 1
		_weights1->sub(D1_3->scalar_multiplyn(learning_rate));
		_weights1 = regularization.reg_weightsm(_weights1, _lambda, _alpha, _reg);

		_bias1->subtract_matrix_rows(D1_2->scalar_multiplyn(learning_rate));

		forward_pass();

		// UI PORTION
		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_y_hat, _output_set));
			PLOG_MSG("Layer 1:");
			MLPPUtilities::print_ui_mb(_weights1, _bias1);
			PLOG_MSG("Layer 2:");
			MLPPUtilities::print_ui_mb(_weights2, _bias2);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPSoftmaxNet::sgd(real_t learning_rate, int max_epoch, bool ui) {
	MLPPActivation avn;
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
	output_set_row_tmp->resize(_output_set->size().x);

	Ref<MLPPMatrix> y_hat_mat_tmp;
	y_hat_mat_tmp.instance();
	y_hat_mat_tmp->resize(Size2i(_bias1->size(), 1));

	Ref<MLPPMatrix> output_row_mat_tmp;
	output_row_mat_tmp.instance();
	output_row_mat_tmp->resize(Size2i(_output_set->size().x, 1));

	while (true) {
		int output_index = distribution(generator);

		_input_set->get_row_into_mlpp_vector(output_index, input_set_row_tmp);
		_output_set->get_row_into_mlpp_vector(output_index, output_set_row_tmp);
		output_row_mat_tmp->set_row_mlpp_vector(0, output_set_row_tmp);

		Ref<MLPPVector> y_hat = evaluatev(input_set_row_tmp);
		y_hat_mat_tmp->set_row_mlpp_vector(0, y_hat);

		PropagateVResult prop_res = propagatev(input_set_row_tmp);

		cost_prev = cost(y_hat_mat_tmp, output_row_mat_tmp);

		Ref<MLPPVector> error = y_hat->subn(output_set_row_tmp);

		// Weight updation for layer 2

		Ref<MLPPMatrix> D2_1 = error->outer_product(prop_res.a2);

		_weights2->sub(D2_1->transposen()->scalar_multiplyn(learning_rate));
		_weights2 = regularization.reg_weightsm(_weights2, _lambda, _alpha, _reg);

		// Bias updation for layer 2
		_bias2->sub(error->scalar_multiplyn(learning_rate));

		// Weight updation for layer 1
		Ref<MLPPVector> D1_1 = _weights2->mult_vec(error);
		Ref<MLPPVector> D1_2 = D1_1->hadamard_productn(avn.sigmoid_derivv(prop_res.z2));
		Ref<MLPPMatrix> D1_3 = input_set_row_tmp->outer_product(D1_2);

		_weights1->sub(D1_3->scalar_multiplyn(learning_rate));
		_weights1 = regularization.reg_weightsm(_weights1, _lambda, _alpha, _reg);
		// Bias updation for layer 1

		_bias1->sub(D1_2->scalar_multiplyn(learning_rate));

		y_hat = evaluatev(input_set_row_tmp);

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat_mat_tmp, output_row_mat_tmp));
			PLOG_MSG("Layer 1:");
			MLPPUtilities::print_ui_mb(_weights1, _bias1);
			PLOG_MSG("Layer 2:");
			MLPPUtilities::print_ui_mb(_weights2, _bias2);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPSoftmaxNet::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	MLPPActivation avn;
	MLPPReg regularization;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;

	MLPPUtilities::CreateMiniBatchMMBatch batches = MLPPUtilities::create_mini_batchesmm(_input_set, _output_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_mini_batch = batches.input_sets[i];
			Ref<MLPPMatrix> current_output_mini_batch = batches.output_sets[i];

			Ref<MLPPMatrix> y_hat = evaluatem(current_input_mini_batch);

			PropagateMResult prop_res = propagatem(current_input_mini_batch);

			cost_prev = cost(y_hat, current_output_mini_batch);

			// Calculating the errors
			Ref<MLPPMatrix> error = y_hat->subn(current_output_mini_batch);

			// Calculating the weight/bias gradients for layer 2

			Ref<MLPPMatrix> D2_1 = prop_res.a2->transposen()->multn(error);

			// weights and bias updation for layser 2
			_weights2->sub(D2_1->scalar_multiplyn(learning_rate));
			_weights2 = regularization.reg_weightsm(_weights2, _lambda, _alpha, _reg);

			// Bias Updation for layer 2
			_bias2->sub(error->scalar_multiplyn(learning_rate));

			//Calculating the weight/bias for layer 1
			Ref<MLPPMatrix> D1_1 = error->multn(_weights2->transposen());
			Ref<MLPPMatrix> D1_2 = D1_1->hadamard_productn(avn.sigmoid_derivm(prop_res.z2));
			Ref<MLPPMatrix> D1_3 = current_input_mini_batch->transposen()->multn(D1_2);

			// weight an bias updation for layer 1
			_weights1->sub(D1_3->scalar_multiplyn(learning_rate));
			_weights1 = regularization.reg_weightsm(_weights1, _lambda, _alpha, _reg);

			_bias1->subtract_matrix_rows(D1_2->scalar_multiplyn(learning_rate));

			y_hat = evaluatem(current_input_mini_batch);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_output_mini_batch));
				PLOG_MSG("Layer 1:");
				MLPPUtilities::print_ui_mb(_weights1, _bias1);
				PLOG_MSG("Layer 2:");
				MLPPUtilities::print_ui_mb(_weights2, _bias2);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

real_t MLPPSoftmaxNet::score() {
	MLPPUtilities util;

	return util.performance_mat(_y_hat, _output_set);
}

void MLPPSoftmaxNet::save(const String &file_name) {
	MLPPUtilities util;

	//util.saveParameters(fileName, _weights1, _bias1, false, 1);
	//util.saveParameters(fileName, _weights2, _bias2, true, 2);
}

Ref<MLPPMatrix> MLPPSoftmaxNet::get_embeddings() {
	return _weights1;
}

bool MLPPSoftmaxNet::is_initialized() {
	return _initialized;
}
void MLPPSoftmaxNet::initialize() {
	if (_initialized) {
		return;
	}

	//ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	_initialized = true;
}

MLPPSoftmaxNet::MLPPSoftmaxNet(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPMatrix> &p_output_set, int p_n_hidden, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha) {
	_input_set = p_input_set;
	_output_set = p_output_set;
	_n = p_input_set->size().y;
	_k = p_input_set->size().x;
	_n_hidden = p_n_hidden;
	_n_class = p_output_set->size().x;
	_reg = p_reg;
	_lambda = p_lambda;
	_alpha = p_alpha;

	_y_hat.instance();
	_y_hat->resize(Size2i(0, _n));

	MLPPUtilities utils;

	_weights1.instance();
	_weights1->resize(Size2i(_n_hidden, _k));
	utils.weight_initializationm(_weights1);

	_weights2.instance();
	_weights2->resize(Size2i(_n_class, _n_hidden));
	utils.weight_initializationm(_weights2);

	_bias1.instance();
	_bias1->resize(_n_hidden);
	utils.bias_initializationv(_bias1);

	_bias2.instance();
	_bias2->resize(_n_class);
	utils.bias_initializationv(_bias2);

	_initialized = true;
}

MLPPSoftmaxNet::MLPPSoftmaxNet() {
	_initialized = false;
}
MLPPSoftmaxNet::~MLPPSoftmaxNet() {
}

real_t MLPPSoftmaxNet::cost(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPReg regularization;
	MLPPData data;
	MLPPCost mlpp_cost;

	return mlpp_cost.cross_entropym(y_hat, y) + regularization.reg_termm(_weights1, _lambda, _alpha, _reg) + regularization.reg_termm(_weights2, _lambda, _alpha, _reg);
}

Ref<MLPPVector> MLPPSoftmaxNet::evaluatev(const Ref<MLPPVector> &x) {
	MLPPActivation avn;

	Ref<MLPPVector> z2 = _weights1->transposen()->mult_vec(x)->addn(_bias1);
	Ref<MLPPVector> a2 = avn.sigmoid_normv(z2);

	return avn.adj_softmax_normv(_weights2->transposen()->mult_vec(a2)->addn(_bias2));
}

MLPPSoftmaxNet::PropagateVResult MLPPSoftmaxNet::propagatev(const Ref<MLPPVector> &x) {
	MLPPActivation avn;

	PropagateVResult res;

	res.z2 = _weights1->transposen()->mult_vec(x)->addn(_bias1);
	res.a2 = avn.sigmoid_normv(res.z2);

	return res;
}

Ref<MLPPMatrix> MLPPSoftmaxNet::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPActivation avn;

	Ref<MLPPMatrix> z2 = X->multn(_weights1)->add_vecn(_bias1);
	Ref<MLPPMatrix> a2 = avn.sigmoid_normm(z2);

	return avn.adj_softmax_normm(a2->multn(_weights2)->add_vecn(_bias2));
}

MLPPSoftmaxNet::PropagateMResult MLPPSoftmaxNet::propagatem(const Ref<MLPPMatrix> &X) {
	MLPPActivation avn;

	MLPPSoftmaxNet::PropagateMResult res;

	res.z2 = X->multn(_weights1)->add_vecn(_bias1);
	res.a2 = avn.sigmoid_normm(res.z2);

	return res;
}

void MLPPSoftmaxNet::forward_pass() {
	MLPPActivation avn;

	_z2 = _input_set->multn(_weights1)->add_vecn(_bias1);
	_a2 = avn.sigmoid_normm(_z2);

	_y_hat = avn.adj_softmax_normm(_a2->multn(_weights2)->add_vecn(_bias2));
}

void MLPPSoftmaxNet::_bind_methods() {
	/*
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPSoftmaxNet::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPSoftmaxNet::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPSoftmaxNet::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPSoftmaxNet::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_reg"), &MLPPSoftmaxNet::get_reg);
	ClassDB::bind_method(D_METHOD("set_reg", "val"), &MLPPSoftmaxNet::set_reg);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reg"), "set_reg", "get_reg");

	ClassDB::bind_method(D_METHOD("get_lambda"), &MLPPSoftmaxNet::get_lambda);
	ClassDB::bind_method(D_METHOD("set_lambda", "val"), &MLPPSoftmaxNet::set_lambda);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lambda"), "set_lambda", "get_lambda");

	ClassDB::bind_method(D_METHOD("get_alpha"), &MLPPSoftmaxNet::get_alpha);
	ClassDB::bind_method(D_METHOD("set_alpha", "val"), &MLPPSoftmaxNet::set_alpha);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "alpha"), "set_alpha", "get_alpha");

	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPSoftmaxNet::model_test);
	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPSoftmaxNet::model_set_test);

	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPSoftmaxNet::gradient_descent, false);
	ClassDB::bind_method(D_METHOD("sgd", "learning_rate", "max_epoch", "ui"), &MLPPSoftmaxNet::sgd, false);
	ClassDB::bind_method(D_METHOD("mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPSoftmaxNet::mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPSoftmaxNet::score);

	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPSoftmaxNet::save);

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPSoftmaxNet::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPSoftmaxNet::initialize);
	*/
}
