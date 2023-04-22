//
//  AutoEncoder.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "auto_encoder.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
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
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _input_set);

		// Calculating the errors
		Ref<MLPPMatrix> error = alg.subtractionnm(_y_hat, _input_set);

		// Calculating the weight/bias gradients for layer 2
		Ref<MLPPMatrix> D2_1 = alg.matmultnm(alg.transposenm(_a2), error);

		// weights and bias updation for layer 2
		_weights2 = alg.subtractionnm(_weights2, alg.scalar_multiplynm(learning_rate / _n, D2_1));

		// Calculating the bias gradients for layer 2
		_bias2 = alg.subtract_matrix_rows(_bias2, alg.scalar_multiplynm(learning_rate, error));

		//Calculating the weight/bias for layer 1

		Ref<MLPPMatrix> D1_1 = alg.matmultnm(error, alg.transposenm(_weights2));
		Ref<MLPPMatrix> D1_2 = alg.hadamard_productnm(D1_1, avn.sigmoid_derivm(_z2));
		Ref<MLPPMatrix> D1_3 = alg.matmultnm(alg.transposenm(_input_set), D1_2);

		// weight an bias updation for layer 1
		_weights1 = alg.subtractionnm(_weights1, alg.scalar_multiplynm(learning_rate / _n, D1_3));

		_bias1 = alg.subtract_matrix_rows(_bias1, alg.scalar_multiplynm(learning_rate / _n, D1_2));

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
	MLPPLinAlg alg;
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

		_input_set->get_row_into_mlpp_vector(output_index, input_set_row_tmp);
		input_set_mat_tmp->set_row_mlpp_vector(0, input_set_row_tmp);

		Ref<MLPPVector> y_hat = evaluatev(input_set_row_tmp);
		y_hat_mat_tmp->set_row_mlpp_vector(0, y_hat);

		PropagateVResult prop_res = propagatev(input_set_row_tmp);

		cost_prev = cost(y_hat_mat_tmp, input_set_mat_tmp);
		Ref<MLPPVector> error = alg.subtractionnv(y_hat, input_set_row_tmp);

		// Weight updation for layer 2
		Ref<MLPPMatrix> D2_1 = alg.outer_product(error, prop_res.a2);
		_weights2 = alg.subtractionnm(_weights2, alg.scalar_multiplynm(learning_rate, alg.transposenm(D2_1)));

		// Bias updation for layer 2
		_bias2 = alg.subtractionnv(_bias2, alg.scalar_multiplynv(learning_rate, error));

		// Weight updation for layer 1
		Ref<MLPPVector> D1_1 = alg.mat_vec_multv(_weights2, error);
		Ref<MLPPVector> D1_2 = alg.hadamard_productnv(D1_1, avn.sigmoid_derivv(prop_res.z2));
		Ref<MLPPMatrix> D1_3 = alg.outer_product(input_set_row_tmp, D1_2);

		_weights1 = alg.subtractionnm(_weights1, alg.scalar_multiplynm(learning_rate, D1_3));
		// Bias updation for layer 1

		_bias1 = alg.subtractionnv(_bias1, alg.scalar_multiplynv(learning_rate, D1_2));

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
	MLPPLinAlg alg;
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
			Ref<MLPPMatrix> error = alg.subtractionnm(y_hat, current_batch);

			// Calculating the weight/bias gradients for layer 2

			Ref<MLPPMatrix> D2_1 = alg.matmultnm(alg.transposenm(prop_res.a2), error);

			// weights and bias updation for layer 2
			_weights2 = alg.subtractionnm(_weights2, alg.scalar_multiplynm(learning_rate / current_batch->size().y, D2_1));

			// Bias Updation for layer 2
			_bias2 = alg.subtract_matrix_rows(_bias2, alg.scalar_multiplynm(learning_rate, error));

			//Calculating the weight/bias for layer 1

			Ref<MLPPMatrix> D1_1 = alg.matmultnm(error, alg.transposenm(_weights2));
			Ref<MLPPMatrix> D1_2 = alg.hadamard_productnm(D1_1, avn.sigmoid_derivm(prop_res.z2));
			Ref<MLPPMatrix> D1_3 = alg.matmultnm(alg.transposenm(current_batch), D1_2);

			// weight an bias updation for layer 1
			_weights1 = alg.subtractionnm(_weights1, alg.scalar_multiplynm(learning_rate / current_batch->size().x, D1_3));
			_bias1 = alg.subtract_matrix_rows(_bias1, alg.scalar_multiplynm(learning_rate / current_batch->size().x, D1_2));

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

	return mlpp_cost.msem(y_hat, _input_set);
}

Ref<MLPPVector> MLPPAutoEncoder::evaluatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	Ref<MLPPVector> z2 = alg.additionnv(alg.mat_vec_multv(alg.transposenm(_weights1), x), _bias1);
	Ref<MLPPVector> a2 = avn.sigmoid_normv(z2);

	return alg.additionnv(alg.mat_vec_multv(alg.transposenm(_weights2), a2), _bias2);
}

MLPPAutoEncoder::PropagateVResult MLPPAutoEncoder::propagatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	PropagateVResult res;

	res.z2 = alg.additionnv(alg.mat_vec_multv(alg.transposenm(_weights1), x), _bias1);
	res.a2 = avn.sigmoid_normv(res.z2);

	return res;
}

Ref<MLPPMatrix> MLPPAutoEncoder::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	Ref<MLPPMatrix> z2 = alg.mat_vec_addv(alg.matmultnm(X, _weights1), _bias1);
	Ref<MLPPMatrix> a2 = avn.sigmoid_normm(z2);

	return alg.mat_vec_addv(alg.matmultnm(a2, _weights2), _bias2);
}

MLPPAutoEncoder::PropagateMResult MLPPAutoEncoder::propagatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;
	MLPPActivation avn;

	PropagateMResult res;

	res.z2 = alg.mat_vec_addv(alg.matmultnm(X, _weights1), _bias1);
	res.a2 = avn.sigmoid_normm(res.z2);

	return res;
}

void MLPPAutoEncoder::forward_pass() {
	MLPPLinAlg alg;
	MLPPActivation avn;

	_z2 = alg.mat_vec_addv(alg.matmultnm(_input_set, _weights1), _bias1);
	_a2 = avn.sigmoid_normm(_z2);
	_y_hat = alg.mat_vec_addv(alg.matmultnm(_a2, _weights2), _bias2);
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
