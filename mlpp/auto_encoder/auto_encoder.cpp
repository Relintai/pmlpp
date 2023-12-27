//
//  AutoEncoder.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "auto_encoder.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../utilities/utilities.h"

#include "core/log/logger.h"

#include <random>

//UDPATE
Ref<MLPPMatrix> MLPPAutoEncoder::get_input_set() {
	return _input_set;
}
void MLPPAutoEncoder::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;

	_initialized = false;
}

int MLPPAutoEncoder::get_n_hidden() {
	return _n_hidden;
}
void MLPPAutoEncoder::set_n_hidden(const int val) {
	_n_hidden = val;

	_initialized = false;
}

Ref<MLPPMatrix> MLPPAutoEncoder::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPMatrix>());

	return evaluatem(X);
}

Ref<MLPPVector> MLPPAutoEncoder::model_test(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPVector>());

	return evaluatev(x);
}

void MLPPAutoEncoder::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _input_set);

		// Calculating the errors
		Ref<MLPPMatrix> error = _y_hat->subn(_input_set);

		// Calculating the weight/bias gradients for layer 2
		Ref<MLPPMatrix> D2_1 = _a2->transposen()->multn(error);

		// weights and bias updation for layer 2
		_weights2->sub(D2_1->scalar_multiplyn(learning_rate / _n));

		// Calculating the bias gradients for layer 2
		_bias2->subtract_matrix_rows(error->scalar_multiplyn(learning_rate));

		//Calculating the weight/bias for layer 1
		Ref<MLPPMatrix> D1_1 = error->multn(_weights2->transposen());
		Ref<MLPPMatrix> D1_2 = D1_1->hadamard_productn(avn.sigmoid_derivm(_z2));
		Ref<MLPPMatrix> D1_3 = _input_set->transposen()->multn(D1_2);

		// weight an bias updation for layer 1
		_weights1->sub(D1_3->scalar_multiplyn(learning_rate / _n));
		_bias1->subtract_matrix_rows(D1_2->scalar_multiplyn(learning_rate / _n));

		forward_pass();

		// UI PORTION
		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_y_hat, _input_set));
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

void MLPPAutoEncoder::sgd(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	real_t cost_prev = 0;
	int epoch = 1;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(_n - 1));

	Ref<MLPPVector> input_set_row_tmp;
	input_set_row_tmp.instance();
	input_set_row_tmp->resize(_input_set->size().x);

	Ref<MLPPMatrix> input_set_mat_tmp;
	input_set_mat_tmp.instance();
	input_set_mat_tmp->resize(Size2i(_input_set->size().x, 1));

	Ref<MLPPMatrix> y_hat_mat_tmp;
	y_hat_mat_tmp.instance();
	y_hat_mat_tmp->resize(Size2i(_bias2->size(), 1));

	while (true) {
		int output_index = distribution(generator);

		_input_set->row_get_into_mlpp_vector(output_index, input_set_row_tmp);
		input_set_mat_tmp->row_set_mlpp_vector(0, input_set_row_tmp);

		Ref<MLPPVector> y_hat = evaluatev(input_set_row_tmp);
		y_hat_mat_tmp->row_set_mlpp_vector(0, y_hat);

		PropagateVResult prop_res = propagatev(input_set_row_tmp);

		cost_prev = cost(y_hat_mat_tmp, input_set_mat_tmp);
		Ref<MLPPVector> error = y_hat->subn(input_set_row_tmp);

		// Weight updation for layer 2
		Ref<MLPPMatrix> D2_1 = error->outer_product(prop_res.a2);
		_weights2->sub(D2_1->transposen()->scalar_multiplyn(learning_rate));

		// Bias updation for layer 2
		_bias2->sub(error->scalar_multiplyn(learning_rate));

		// Weight updation for layer 1
		Ref<MLPPVector> D1_1 = _weights2->mult_vec(error);
		Ref<MLPPVector> D1_2 = D1_1->hadamard_productn(avn.sigmoid_derivv(prop_res.z2));
		Ref<MLPPMatrix> D1_3 = input_set_row_tmp->outer_product(D1_2);

		_weights1->sub(D1_3->scalar_multiplyn(learning_rate));

		// Bias updation for layer 1

		_bias1->sub(D1_2->scalar_multiplyn(learning_rate));

		y_hat = evaluatev(input_set_row_tmp);

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat_mat_tmp, input_set_mat_tmp));

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

void MLPPAutoEncoder::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPActivation avn;
	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	Vector<Ref<MLPPMatrix>> batches = MLPPUtilities::create_mini_batchesm(_input_set, n_mini_batch);

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_batch = batches[i];

			Ref<MLPPMatrix> y_hat = evaluatem(current_batch);

			PropagateMResult prop_res = propagatem(current_batch);

			cost_prev = cost(y_hat, current_batch);

			// Calculating the errors
			Ref<MLPPMatrix> error = y_hat->subn(current_batch);

			// Calculating the weight/bias gradients for layer 2
			Ref<MLPPMatrix> D2_1 = prop_res.a2->transposen()->multn(error);

			// weights and bias updation for layer 2
			_weights2->sub(D2_1->scalar_multiplyn(learning_rate / current_batch->size().y));

			// Bias Updation for layer 2
			_bias2->sub(error->scalar_multiplyn(learning_rate));

			//Calculating the weight/bias for layer 1

			Ref<MLPPMatrix> D1_1 = _weights2->transposen()->multn(error);
			Ref<MLPPMatrix> D1_2 = D1_1->hadamard_productn(avn.sigmoid_derivm(prop_res.z2));
			Ref<MLPPMatrix> D1_3 = current_batch->transposen()->multn(D1_2);

			// weight an bias updation for layer 1
			_weights2->sub(D1_3->scalar_multiplyn(learning_rate / current_batch->size().x));
			_bias1->subtract_matrix_rows(D1_2->scalar_multiplyn(learning_rate / current_batch->size().x));

			y_hat = evaluatem(current_batch);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, current_batch));
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

real_t MLPPAutoEncoder::score() {
	ERR_FAIL_COND_V(!_initialized, 0);

	MLPPUtilities util;
	return util.performance_mat(_y_hat, _input_set);
}

void MLPPAutoEncoder::save(const String &file_name) {
	ERR_FAIL_COND(!_initialized);

	//MLPPUtilities util;
	//util.saveParameters(fileName, _weights1, _bias1, false, 1);
	//util.saveParameters(fileName, _weights2, _bias2, true, 2);
}

MLPPAutoEncoder::MLPPAutoEncoder(const Ref<MLPPMatrix> &p_input_set, int p_n_hidden) {
	_input_set = p_input_set;
	_n_hidden = p_n_hidden;
	_n = _input_set->size().y;
	_k = _input_set->size().x;

	_y_hat.instance();
	_y_hat->resize(_input_set->size());

	MLPPUtilities utilities;

	_weights1.instance();
	_weights1->resize(Size2i(_n_hidden, _k));
	utilities.weight_initializationm(_weights1);

	_weights2.instance();
	_weights2->resize(Size2i(_k, _n_hidden));
	utilities.weight_initializationm(_weights2);

	_bias1.instance();
	_bias1->resize(_n_hidden);
	utilities.bias_initializationv(_bias1);

	_bias2.instance();
	_bias2->resize(_k);
	utilities.bias_initializationv(_bias2);

	_initialized = true;
}

MLPPAutoEncoder::MLPPAutoEncoder() {
	_initialized = false;
}
MLPPAutoEncoder::~MLPPAutoEncoder() {
}

real_t MLPPAutoEncoder::cost(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPCost mlpp_cost;

	return mlpp_cost.msem(y_hat, y);
}

Ref<MLPPVector> MLPPAutoEncoder::evaluatev(const Ref<MLPPVector> &x) {
	MLPPActivation avn;

	Ref<MLPPVector> z2 = _weights1->transposen()->mult_vec(x)->addn(_bias1);
	Ref<MLPPVector> a2 = avn.sigmoid_normv(z2);

	return _weights2->transposen()->mult_vec(a2)->addn(_bias2);
}

MLPPAutoEncoder::PropagateVResult MLPPAutoEncoder::propagatev(const Ref<MLPPVector> &x) {
	MLPPActivation avn;

	PropagateVResult res;

	res.z2 = _weights1->transposen()->mult_vec(x)->addn(_bias1);
	res.a2 = avn.sigmoid_normv(res.z2);

	return res;
}

Ref<MLPPMatrix> MLPPAutoEncoder::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPActivation avn;

	Ref<MLPPMatrix> z2 = X->multn(_weights1)->add_vecn(_bias1);
	Ref<MLPPMatrix> a2 = avn.sigmoid_normm(z2);

	return a2->multn(_weights2)->add_vecn(_bias2);
}

MLPPAutoEncoder::PropagateMResult MLPPAutoEncoder::propagatem(const Ref<MLPPMatrix> &X) {
	MLPPActivation avn;

	PropagateMResult res;

	res.z2 = X->multn(_weights1)->add_vecn(_bias1);
	res.a2 = avn.sigmoid_normm(res.z2);

	return res;
}

void MLPPAutoEncoder::forward_pass() {
	MLPPActivation avn;

	_z2 = _input_set->multn(_weights1)->add_vecn(_bias1);
	_a2 = avn.sigmoid_normm(_z2);

	_y_hat = _a2->multn(_weights2)->add_vecn(_bias2);
}

void MLPPAutoEncoder::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPAutoEncoder::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPAutoEncoder::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_n_hidden"), &MLPPAutoEncoder::get_n_hidden);
	ClassDB::bind_method(D_METHOD("set_n_hidden", "val"), &MLPPAutoEncoder::set_n_hidden);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "n_hidden"), "set_n_hidden", "get_n_hidden");

	/*
	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPAutoEncoder::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPAutoEncoder::model_test);

	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPAutoEncoder::gradient_descent, false);
	ClassDB::bind_method(D_METHOD("sgd", "learning_rate", "max_epoch", "ui"), &MLPPAutoEncoder::sgd, false);
	ClassDB::bind_method(D_METHOD("mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPAutoEncoder::mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPAutoEncoder::score);

	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPAutoEncoder::save);

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPAutoEncoder::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPAutoEncoder::initialize);
	*/
}
