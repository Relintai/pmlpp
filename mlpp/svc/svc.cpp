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

Ref<MLPPMatrix> MLPPSVC::get_input_set() const {
	return _input_set;
}
void MLPPSVC::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;
}

Ref<MLPPVector> MLPPSVC::get_output_set() const {
	return _output_set;
}
void MLPPSVC::set_output_set(const Ref<MLPPMatrix> &val) {
	_output_set = val;
}

real_t MLPPSVC::get_c() const {
	return _c;
}
void MLPPSVC::set_c(const real_t val) {
	_c = val;
}

Ref<MLPPVector> MLPPSVC::data_z_get() const {
	return _z;
}
void MLPPSVC::data_z_set(const Ref<MLPPVector> &val) {
	_z = val;
}

Ref<MLPPVector> MLPPSVC::data_y_hat_get() const {
	return _y_hat;
}
void MLPPSVC::data_y_hat_set(const Ref<MLPPVector> &val) {
	_y_hat = val;
}

Ref<MLPPVector> MLPPSVC::data_weights_get() const {
	return _weights;
}
void MLPPSVC::data_weights_set(const Ref<MLPPVector> &val) {
	_weights = val;
}

real_t MLPPSVC::data_bias_get() const {
	return _bias;
}
void MLPPSVC::data_bias_set(const real_t val) {
	_bias = val;
}

Ref<MLPPVector> MLPPSVC::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(needs_init(), Ref<MLPPVector>());

	return evaluatem(X);
}

real_t MLPPSVC::model_test(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(needs_init(), 0);

	return evaluatev(x);
}

void MLPPSVC::train_gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());
	ERR_FAIL_COND(needs_init());

	int n = _input_set->size().y;

	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, _output_set, _weights, _c);

		_weights->sub(_input_set->transposen()->mult_vec(mlpp_cost.hinge_loss_derivwv(_z, _output_set, _c))->scalar_multiplyn(learning_rate / n));
		_weights = regularization.reg_weightsv(_weights, learning_rate / n, 0, MLPPReg::REGULARIZATION_TYPE_RIDGE);

		// Calculating the bias gradients
		_bias += learning_rate * mlpp_cost.hinge_loss_derivwv(_y_hat, _output_set, _c)->sum_elements() / n;

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

void MLPPSVC::train_sgd(real_t learning_rate, int max_epoch, bool ui) {
	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());
	ERR_FAIL_COND(needs_init());

	int n = _input_set->size().y;

	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPReg regularization;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(n - 1));

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

		_input_set->row_get_into_mlpp_vector(output_index, input_set_row_tmp);

		real_t output_set_indx = _output_set->element_get(output_index);
		output_set_row_tmp->element_set(0, output_set_indx);

		//real_t y_hat = Evaluate(input_set_row_tmp);
		real_t z = propagatev(input_set_row_tmp);

		z_row_tmp->element_set(0, z);

		cost_prev = cost(z_row_tmp, output_set_row_tmp, _weights, _c);

		Ref<MLPPVector> cost_deriv_vec = mlpp_cost.hinge_loss_derivwv(z_row_tmp, output_set_row_tmp, _c);

		real_t cost_deriv = cost_deriv_vec->element_get(0);

		// Weight Updation
		_weights->sub(input_set_row_tmp->scalar_multiplyn(learning_rate * cost_deriv));
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

void MLPPSVC::train_mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui) {
	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());
	ERR_FAIL_COND(needs_init());

	int n = _input_set->size().y;

	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPReg regularization;

	real_t cost_prev = 0;
	int epoch = 1;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
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
			_weights->subn(current_input_batch_entry->transposen()->mult_vec(mlpp_cost.hinge_loss_derivwv(z, current_output_batch_entry, _c))->scalar_multiplyn(learning_rate / n));
			_weights = regularization.reg_weightsv(_weights, learning_rate / n, 0, MLPPReg::REGULARIZATION_TYPE_RIDGE);

			// Calculating the bias gradients
			_bias -= learning_rate * mlpp_cost.hinge_loss_derivwv(y_hat, current_output_batch_entry, _c)->sum_elements() / n;

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
	ERR_FAIL_COND_V(needs_init(), 0);

	MLPPUtilities util;
	return util.performance_vec(_y_hat, _output_set);
}

bool MLPPSVC::needs_init() const {
	if (!_input_set.is_valid()) {
		return true;
	}

	if (!_output_set.is_valid()) {
		return true;
	}

	int n = _input_set->size().y;
	int k = _input_set->size().x;

	if (_y_hat->size() != n) {
		return true;
	}

	if (_weights->size() != k) {
		return true;
	}

	return false;
}
void MLPPSVC::initialize() {
	ERR_FAIL_COND(!_input_set.is_valid() || !_output_set.is_valid());

	int n = _input_set->size().y;
	int k = _input_set->size().x;

	_y_hat->resize(n);

	MLPPUtilities util;

	_weights->resize(k);

	util.weight_initializationv(_weights);
	_bias = util.bias_initializationr();
}

MLPPSVC::MLPPSVC(const Ref<MLPPMatrix> &input_set, const Ref<MLPPVector> &output_set, real_t c) {
	_input_set = input_set;
	_output_set = output_set;
	_c = c;

	_z.instance();
	_y_hat.instance();
	_weights.instance();
	_bias = 0;

	initialize();
}

MLPPSVC::MLPPSVC() {
	_c = 0;

	_z.instance();
	_y_hat.instance();
	_weights.instance();
	_bias = 0;
}
MLPPSVC::~MLPPSVC() {
}

real_t MLPPSVC::cost(const Ref<MLPPVector> &z, const Ref<MLPPVector> &y, const Ref<MLPPVector> &weights, real_t c) {
	MLPPCost mlpp_cost;
	return mlpp_cost.hinge_losswv(z, y, weights, c);
}

Ref<MLPPVector> MLPPSVC::evaluatem(const Ref<MLPPMatrix> &X) {
	MLPPActivation avn;

	return avn.sign_normv(X->mult_vec(_weights)->scalar_addn(_bias));
}

Ref<MLPPVector> MLPPSVC::propagatem(const Ref<MLPPMatrix> &X) {
	return X->mult_vec(_weights)->scalar_addn(_bias);
}

real_t MLPPSVC::evaluatev(const Ref<MLPPVector> &x) {
	MLPPActivation avn;

	return avn.sign_normr(_weights->dot(x) + _bias);
}

real_t MLPPSVC::propagatev(const Ref<MLPPVector> &x) {
	MLPPActivation avn;
	return _weights->dot(x) + _bias;
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

	ClassDB::bind_method(D_METHOD("data_z_get"), &MLPPSVC::data_z_get);
	ClassDB::bind_method(D_METHOD("data_z_set", "val"), &MLPPSVC::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "data_z", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "data_z_set", "data_z_get");

	ClassDB::bind_method(D_METHOD("data_y_hat_get"), &MLPPSVC::data_y_hat_get);
	ClassDB::bind_method(D_METHOD("data_y_hat_set", "val"), &MLPPSVC::data_y_hat_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "data_y_hat", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "data_y_hat_set", "data_y_hat_get");

	ClassDB::bind_method(D_METHOD("data_weights_get"), &MLPPSVC::data_weights_get);
	ClassDB::bind_method(D_METHOD("data_weights_set", "val"), &MLPPSVC::data_weights_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "data_weights", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "data_weights_set", "data_weights_get");

	ClassDB::bind_method(D_METHOD("data_bias_get"), &MLPPSVC::data_bias_get);
	ClassDB::bind_method(D_METHOD("data_bias_set", "val"), &MLPPSVC::data_bias_set);
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "data_bias"), "data_bias_set", "data_bias_get");

	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPSVC::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPSVC::model_test);

	ClassDB::bind_method(D_METHOD("train_gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPSVC::train_gradient_descent, false);
	ClassDB::bind_method(D_METHOD("train_sgd", "learning_rate", "max_epoch", "ui"), &MLPPSVC::train_sgd, false);
	ClassDB::bind_method(D_METHOD("train_mbgd", "learning_rate", "max_epoch", "mini_batch_size", "ui"), &MLPPSVC::train_mbgd, false);

	ClassDB::bind_method(D_METHOD("score"), &MLPPSVC::score);

	ClassDB::bind_method(D_METHOD("needs_init"), &MLPPSVC::needs_init);
	ClassDB::bind_method(D_METHOD("initialize"), &MLPPSVC::initialize);
}
