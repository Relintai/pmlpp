//
//  SVC.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "svc.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <random>

Ref<MLPPMatrix> MLPPSVC::get_input_set() {
	return _input_set;
}
void MLPPSVC::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;

	_initialized = false;
}

Ref<MLPPVector> MLPPSVC::get_output_set() {
	return _output_set;
}
void MLPPSVC::set_output_set(const Ref<MLPPMatrix> &val) {
	_output_set = val;

	_initialized = false;
}

real_t MLPPSVC::get_c() {
	return _c;
}
void MLPPSVC::set_c(const real_t val) {
	_c = val;

	_initialized = false;
}

Ref<MLPPVector> MLPPSVC::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPVector>());

	return evaluatem(X);
}

real_t MLPPSVC::model_test(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!_initialized, 0);

	return evaluatev(x);
}

void MLPPSVC::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set, _weights, _c);

		_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate / _n, alg.mat_vec_multv(alg.transposem(_input_set), mlpp_cost.hinge_loss_derivwv(_z, _output_set, _c))));
		_weights = regularization.reg_weightsv(_weights, learning_rate / _n, 0, MLPPReg::REGULARIZATION_TYPE_RIDGE);

		// Calculating the bias gradients
		_bias += learning_rate * alg.sum_elementsv(mlpp_cost.hinge_loss_derivwv(_y_hat, _output_set, _c)) / _n;

		forward_pass();

		// UI PORTION
		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(_y_hat, _output_set, _weights, _c));
			MLPPUtilities::print_ui_vb(_weights, _bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPSVC::sgd(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(_n - 1));

	Ref<MLPPVector> input_set_row_tmp;
	input_set_row_tmp.instance();
	input_set_row_tmp->resize(_input_set->size().x);

	Ref<MLPPVector> output_set_row_tmp;
	output_set_row_tmp.instance();
	output_set_row_tmp->resize(1);

	Ref<MLPPVector> z_row_tmp;
	z_row_tmp.instance();
	z_row_tmp->resize(1);

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		int output_index = distribution(generator);

		_input_set->get_row_into_mlpp_vector(output_index, input_set_row_tmp);

		real_t output_set_indx = _output_set->get_element(output_index);
		output_set_row_tmp->set_element(0, output_set_indx);

		//real_t y_hat = Evaluate(input_set_row_tmp);
		real_t z = propagatev(input_set_row_tmp);

		z_row_tmp->set_element(0, z);

		cost_prev = cost(z_row_tmp, output_set_row_tmp, _weights, _c);

		Ref<MLPPVector> cost_deriv_vec = mlpp_cost.hinge_loss_derivwv(z_row_tmp, output_set_row_tmp, _c);

		real_t cost_deriv = cost_deriv_vec->get_element(0);

		// Weight Updation
		_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate * cost_deriv, input_set_row_tmp));
		_weights = regularization.reg_weightsv(_weights, learning_rate, 0, MLPPReg::REGULARIZATION_TYPE_RIDGE);

		// Bias updation
		_bias -= learning_rate * cost_deriv;

		//y_hat = Evaluate({ _input_set[output_index] });

		if (ui) {
			MLPPUtilities::cost_info(epoch, cost_prev, cost(z_row_tmp, output_set_row_tmp, _weights, _c));
			MLPPUtilities::print_ui_vb(_weights, _bias);
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

void MLPPSVC::mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	ERR_FAIL_COND(!_initialized);

	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = _n / mini_batch_size;
	MLPPUtilities::CreateMiniBatchMVBatch batches = MLPPUtilities::create_mini_batchesmv(_input_set, _output_set, n_mini_batch);

	forward_pass();

	while (true) {
		for (int i = 0; i < n_mini_batch; i++) {
			Ref<MLPPMatrix> current_input_batch_entry = batches.input_sets[i];
			Ref<MLPPVector> current_output_batch_entry = batches.output_sets[i];

			Ref<MLPPVector> y_hat = evaluatem(current_input_batch_entry);
			Ref<MLPPVector> z = propagatem(current_input_batch_entry);
			cost_prev = cost(z, current_output_batch_entry, _weights, _c);

			// Calculating the weight gradients
			_weights = alg.subtractionnv(_weights, alg.scalar_multiplynv(learning_rate / _n, alg.mat_vec_multv(alg.transposem(current_input_batch_entry), mlpp_cost.hinge_loss_derivwv(z, current_output_batch_entry, _c))));
			_weights = regularization.reg_weightsv(_weights, learning_rate / _n, 0, MLPPReg::REGULARIZATION_TYPE_RIDGE);

			// Calculating the bias gradients
			_bias -= learning_rate * alg.sum_elementsv(mlpp_cost.hinge_loss_derivwv(y_hat, current_output_batch_entry, _c)) / _n;

			forward_pass();

			y_hat = evaluatem(current_input_batch_entry);

			if (ui) {
				MLPPUtilities::cost_info(epoch, cost_prev, cost(z, current_output_batch_entry, _weights, _c));
				MLPPUtilities::print_ui_vb(_weights, _bias);
			}
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}

	forward_pass();
}

real_t MLPPSVC::score() {
	ERR_FAIL_COND_V(!_initialized, 0);

	MLPPUtilities util;
	return util.performance_vec(_y_hat, _output_set);
}

void MLPPSVC::save(const String &file_name) {
	ERR_FAIL_COND(!_initialized);

	MLPPUtilities util;

	//util.saveParameters(_file_name, _weights, _bias);
}

bool MLPPSVC::is_initialized() {
	return _initialized;
}
void MLPPSVC::initialize() {
	if (_initialized) {
		return;
	}

	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	_n = _input_set->size().y;
	_k = _input_set->size().x;

	if (!_y_hat.is_valid()) {
		_y_hat.instance();
	}

	_y_hat->resize(_n);

	MLPPUtilities util;

	if (!_weights.is_valid()) {
		_weights.instance();
	}

	_weights->resize(_k);

	util.weight_initializationv(_weights);
	_bias = util.bias_initializationr();

	_initialized = true;
}

MLPPSVC::MLPPSVC(const Ref<MLPPMatrix> &input_set, const Ref<MLPPVector> &output_set, real_t c) {
	_input_set = input_set;
	_output_set = output_set;

	_n = _input_set->size().y;
	_k = _input_set->size().x;
	_c = c;

	_y_hat.instance();

	_y_hat->resize(_n);

	MLPPUtilities util;

	_weights.instance();
	_weights->resize(_k);
	util.weight_initializationv(_weights);
	_bias = util.bias_initializationr();

	_initialized = true;
}

MLPPSVC::MLPPSVC() {
	_y_hat.instance();
	_weights.instance();

	_c = 0;
	_n = 0;
	_k = 0;

	_initialized = false;
}
MLPPSVC::~MLPPSVC() {
}

real_t MLPPSVC::cost(const Ref<MLPPVector> &z, const Ref<MLPPVector> &y, const Ref<MLPPVector> &weights, real_t c) {
	MLPPCost mlpp_cost;
	return mlpp_cost.hinge_losswv(z, y, weights, c);
}

Ref<MLPPVector> MLPPSVC::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.sign_normv(alg.scalar_addnv(_bias, alg.mat_vec_multv(X, _weights)));
}

Ref<MLPPVector> MLPPSVC::propagatem(const Ref<MLPPMatrix> &X) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return alg.scalar_addnv(_bias, alg.mat_vec_multv(X, _weights));
}

real_t MLPPSVC::evaluatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return avn.sign_normr(alg.dotv(_weights, x) + _bias);
}

real_t MLPPSVC::propagatev(const Ref<MLPPVector> &x) {
	MLPPLinAlg alg;
	MLPPActivation avn;
	return alg.dotv(_weights, x) + _bias;
}

// sign ( wTx + b )
void MLPPSVC::forward_pass() {
	MLPPActivation avn;

	_z = propagatem(_input_set);
	_y_hat = avn.sign_normv(_z);
}

void MLPPSVC::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPSVC::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPSVC::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPSVC::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPSVC::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_c"), &MLPPSVC::get_c);
	ClassDB::bind_method(D_METHOD("set_c", "val"), &MLPPSVC::set_c);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "c"), "set_c", "get_c");

	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPSVC::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPSVC::model_test);

	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPSVC::gradient_descent, false);
	ClassDB::bind_method(D_METHOD("sgd", "learning_rate", "max_epoch", "ui"), &MLPPSVC::sgd, false);
	ClassDB::bind_method(D_METHOD("mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPSVC::mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPSVC::score);

	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPSVC::save);

	ClassDB::bind_method(D_METHOD("is_initialized"), &MLPPSVC::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPSVC::initialize);
}
