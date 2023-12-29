/*************************************************************************/
/*  wgan.cpp                                                             */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2023-present Péter Magyar.                              */
/* Copyright (c) 2022-2023 Marc Melikyan                                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "wgan.h"

#include "core/log/logger.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include "core/object/method_bind_ext.gen.inc"

Ref<MLPPMatrix> MLPPWGAN::get_output_set() {
	return _output_set;
}
void MLPPWGAN::set_output_set(const Ref<MLPPMatrix> &val) {
	_output_set = val;
}

int MLPPWGAN::get_k() const {
	return _k;
}
void MLPPWGAN::set_k(const int val) {
	_k = val;
}

Ref<MLPPMatrix> MLPPWGAN::generate_example(int n) {
	return model_set_test_generator(MLPPMatrix::create_gaussian_noise(n, _k));
}

void MLPPWGAN::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	//MLPPCost mlpp_cost;
	real_t cost_prev = 0;
	int epoch = 1;
	int n = _output_set->size().y;

	forward_pass();

	const int CRITIC_INTERATIONS = 5; // Wasserstein GAN specific parameter.

	while (true) {
		cost_prev = cost(_y_hat, MLPPVector::create_vec_one(n));

		Ref<MLPPMatrix> generator_input_set;
		Ref<MLPPMatrix> discriminator_input_set;
		discriminator_input_set.instance();

		Ref<MLPPVector> ly_hat;
		Ref<MLPPVector> loutput_set;

		// Training of the discriminator.
		for (int i = 0; i < CRITIC_INTERATIONS; i++) {
			generator_input_set = MLPPMatrix::create_gaussian_noise(n, _k);
			discriminator_input_set->set_from_mlpp_matrix(model_set_test_generator(generator_input_set));
			discriminator_input_set->rows_add_mlpp_matrix(_output_set); // Fake + real inputs.

			ly_hat = model_set_test_discriminator(discriminator_input_set);
			loutput_set = MLPPVector::create_vec_one(n)->scalar_multiplyn(-1); // WGAN changes y_i = 1 and y_i = 0 to y_i = 1 and y_i = -1
			Ref<MLPPVector> output_set_real = MLPPVector::create_vec_one(n);
			loutput_set->append_mlpp_vector(output_set_real); // Fake + real output scores.

			DiscriminatorGradientResult discriminator_gradient_results = compute_discriminator_gradients(ly_hat, loutput_set);
			Vector<Ref<MLPPMatrix>> cumulative_discriminator_hidden_layer_w_grad = discriminator_gradient_results.cumulative_hidden_layer_w_grad;
			Ref<MLPPVector> output_discriminator_w_grad = discriminator_gradient_results.output_w_grad;

			real_t lrpn = learning_rate / n;

			for (int j = 0; j < cumulative_discriminator_hidden_layer_w_grad.size(); ++j) {
				cumulative_discriminator_hidden_layer_w_grad.write[j]->scalar_multiply(lrpn);
			}

			output_discriminator_w_grad->scalar_multiply(learning_rate / n);
			update_discriminator_parameters(cumulative_discriminator_hidden_layer_w_grad, output_discriminator_w_grad, learning_rate);
		}

		// Training of the generator.
		generator_input_set = MLPPMatrix::create_gaussian_noise(n, _k);
		discriminator_input_set->set_from_mlpp_matrix(model_set_test_generator(generator_input_set));
		ly_hat = model_set_test_discriminator(discriminator_input_set);
		loutput_set = MLPPVector::create_vec_one(n);

		Vector<Ref<MLPPMatrix>> cumulative_generator_hidden_layer_w_grad = compute_generator_gradients(_y_hat, loutput_set);

		real_t lrpn = learning_rate / n;

		for (int i = 0; i < cumulative_generator_hidden_layer_w_grad.size(); ++i) {
			cumulative_generator_hidden_layer_w_grad.write[i]->scalar_multiply(lrpn);
		}

		update_generator_parameters(cumulative_generator_hidden_layer_w_grad, learning_rate);

		forward_pass();

		if (ui) {
			handle_ui(epoch, cost_prev, _y_hat, MLPPVector::create_vec_one(n));
		}

		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
}

real_t MLPPWGAN::score() {
	MLPPUtilities util;
	forward_pass();
	int n = _output_set->size().y;

	return util.performance_vec(_y_hat, MLPPVector::create_vec_one(n));
}

void MLPPWGAN::save(const String &file_name) {
	MLPPUtilities util;

	/*
	if (!network.empty()) {
		util.saveParameters(file_name, network[0].weights, network[0].bias, 0, 1);
		for (uint32_t i = 1; i < network.size(); i++) {
			util.saveParameters(fileName, network[i].weights, network[i].bias, 1, i + 1);
		}
		util.saveParameters(file_name, outputLayer->weights, outputLayer->bias, 1, network.size() + 1);
	} else {
		util.saveParameters(file_name, outputLayer->weights, outputLayer->bias, 0, network.size() + 1);
	}
	*/
}

void MLPPWGAN::create_layer(int n_hidden, MLPPActivation::ActivationFunction activation, MLPPUtilities::WeightDistributionType weight_init, MLPPReg::RegularizationType reg, real_t lambda, real_t alpha) {
	Ref<MLPPHiddenLayer> layer;
	layer.instance();

	layer->set_n_hidden(n_hidden);
	layer->set_activation(activation);
	layer->set_weight_init(weight_init);
	layer->set_reg(reg);
	layer->set_lambda(lambda);
	layer->set_alpha(alpha);

	int n = _output_set->size().y;

	if (_network.empty()) {
		layer->set_input(MLPPMatrix::create_gaussian_noise(n, _k));
	} else {
		layer->set_input(_network.write[_network.size() - 1]->get_a());
	}

	_network.push_back(layer);
	layer->forward_pass();
}
void MLPPWGAN::add_layer(Ref<MLPPHiddenLayer> layer) {
	if (!layer.is_valid()) {
		return;
	}

	if (_network.empty()) {
		int n = _output_set->size().y;

		layer->set_input(MLPPMatrix::create_gaussian_noise(n, _k));
	} else {
		layer->set_input(_network.write[_network.size() - 1]->get_a());
	}

	_network.push_back(layer);
	layer->forward_pass();
}
Ref<MLPPHiddenLayer> MLPPWGAN::get_layer(const int index) {
	ERR_FAIL_INDEX_V(index, _network.size(), Ref<MLPPHiddenLayer>());

	return _network[index];
}
void MLPPWGAN::remove_layer(const int index) {
	ERR_FAIL_INDEX(index, _network.size());

	_network.remove(index);
}
int MLPPWGAN::get_layer_count() const {
	return _network.size();
}

void MLPPWGAN::add_output_layer(MLPPUtilities::WeightDistributionType weight_init, MLPPReg::RegularizationType reg, real_t lambda, real_t alpha) {
	ERR_FAIL_COND(_network.empty());

	if (!_output_layer.is_valid()) {
		_output_layer.instance();
	}

	_output_layer->set_n_hidden(_network[_network.size() - 1]->get_n_hidden());
	_output_layer->set_activation(MLPPActivation::ACTIVATION_FUNCTION_LINEAR);
	_output_layer->set_cost(MLPPCost::COST_TYPE_WASSERSTEIN_LOSS);
	_output_layer->set_input(_network.write[_network.size() - 1]->get_a());
	_output_layer->set_weight_init(weight_init);
	_output_layer->set_lambda(lambda);
	_output_layer->set_alpha(alpha);
}

MLPPWGAN::MLPPWGAN(int p_k, const Ref<MLPPMatrix> &p_output_set) {
	_output_set = p_output_set;
	_k = p_k;

	_y_hat.instance();
}

MLPPWGAN::MLPPWGAN() {
	_k = 0;

	_y_hat.instance();
}

MLPPWGAN::~MLPPWGAN() {
}

Ref<MLPPMatrix> MLPPWGAN::model_set_test_generator(const Ref<MLPPMatrix> &X) {
	if (!_network.empty()) {
		_network.write[0]->set_input(X);
		_network.write[0]->forward_pass();

		for (int i = 1; i <= _network.size() / 2; ++i) {
			_network.write[i]->set_input(_network.write[i - 1]->get_a());
			_network.write[i]->forward_pass();
		}
	}

	return _network.write[_network.size() / 2]->get_a();
}

Ref<MLPPVector> MLPPWGAN::model_set_test_discriminator(const Ref<MLPPMatrix> &X) {
	if (!_network.empty()) {
		for (int i = _network.size() / 2 + 1; i < _network.size(); i++) {
			if (i == _network.size() / 2 + 1) {
				_network.write[i]->set_input(X);
			} else {
				_network.write[i]->set_input(_network.write[i - 1]->get_a());
			}
			_network.write[i]->forward_pass();
		}

		_output_layer->set_input(_network.write[_network.size() - 1]->get_a());
	}

	_output_layer->forward_pass();

	return _output_layer->get_a();
}

real_t MLPPWGAN::cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;

	real_t total_reg_term = 0;

	for (int i = 0; i < _network.size() - 1; ++i) {
		Ref<MLPPHiddenLayer> layer = _network[i];

		total_reg_term += regularization.reg_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg());
	}

	total_reg_term += regularization.reg_termv(_output_layer->get_weights(), _output_layer->get_lambda(), _output_layer->get_alpha(), _output_layer->get_reg());

	return mlpp_cost.run_cost_norm_vector(_output_layer->get_cost(), y_hat, y) + total_reg_term;
}

void MLPPWGAN::forward_pass() {
	int n = _output_set->size().y;

	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[0];

		layer->set_input(MLPPMatrix::create_gaussian_noise(n, _k));
		layer->forward_pass();

		for (int i = 1; i < _network.size(); i++) {
			layer = _network[i];

			layer->set_input(_network.write[i - 1]->get_a());
			layer->forward_pass();
		}

		_output_layer->set_input(_network.write[_network.size() - 1]->get_a());
	} else { // Should never happen, though.
		_output_layer->set_input(MLPPMatrix::create_gaussian_noise(n, _k));
	}

	_output_layer->forward_pass();

	_y_hat->set_from_mlpp_vector(_output_layer->get_a());
}

void MLPPWGAN::update_discriminator_parameters(const Vector<Ref<MLPPMatrix>> &hidden_layer_updations, const Ref<MLPPVector> &output_layer_updation, real_t learning_rate) {
	int n = _output_set->size().y;

	_output_layer->set_weights(_output_layer->get_weights()->subn(output_layer_updation));
	_output_layer->set_bias(_output_layer->get_bias() - learning_rate * _output_layer->get_delta()->sum_elements() / n);

	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[_network.size() - 1];

		Ref<MLPPMatrix> slice = hidden_layer_updations[0];

		layer->set_weights(layer->get_weights()->subn(slice));
		layer->set_bias(layer->get_bias()->subtract_matrix_rowsn(layer->get_delta()->scalar_multiplyn(learning_rate / n)));

		for (int i = _network.size() - 2; i > _network.size() / 2; i--) {
			layer = _network[i];

			slice = hidden_layer_updations[(_network.size() - 2) - i + 1];

			layer->set_weights(layer->get_weights()->subn(slice));
			layer->set_bias(layer->get_bias()->subtract_matrix_rowsn(layer->get_delta()->scalar_multiplyn(learning_rate / n)));
		}
	}
}

void MLPPWGAN::update_generator_parameters(const Vector<Ref<MLPPMatrix>> &hidden_layer_updations, real_t learning_rate) {
	if (!_network.empty()) {
		int n = _output_set->size().y;

		Ref<MLPPMatrix> slice;

		for (int i = _network.size() / 2; i >= 0; i--) {
			Ref<MLPPHiddenLayer> layer = _network[i];

			slice = hidden_layer_updations[(_network.size() - 2) - i + 1];

			//std::cout << network[i].weights.size() << "x" << network[i].weights[0].size() << std::endl;
			//std::cout << hiddenLayerUpdations[(network.size() - 2) - i + 1].size() << "x" << hiddenLayerUpdations[(network.size() - 2) - i + 1][0].size() << std::endl;
			layer->set_weights(layer->get_weights()->subn(slice));
			layer->set_bias(layer->get_bias()->subtract_matrix_rowsn(layer->get_delta()->scalar_multiplyn(learning_rate / n)));
		}
	}
}

MLPPWGAN::DiscriminatorGradientResult MLPPWGAN::compute_discriminator_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set) {
	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPReg regularization;

	DiscriminatorGradientResult data;

	_output_layer->set_delta(mlpp_cost.run_cost_deriv_vector(_output_layer->get_cost(), y_hat, output_set)->hadamard_productn(avn.run_activation_deriv_vector(_output_layer->get_activation(), _output_layer->get_z())));

	data.output_w_grad = _output_layer->get_input()->transposen()->mult_vec(_output_layer->get_delta());
	data.output_w_grad->add(regularization.reg_deriv_termv(_output_layer->get_weights(), _output_layer->get_lambda(), _output_layer->get_alpha(), _output_layer->get_reg()));

	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[_network.size() - 1];

		layer->set_delta(_output_layer->get_delta()->outer_product(_output_layer->get_weights())->hadamard_productn(avn.run_activation_deriv_matrix(layer->get_activation(), layer->get_z())));

		Ref<MLPPMatrix> hidden_layer_w_grad = layer->get_input()->transposen()->multn(layer->get_delta());

		data.cumulative_hidden_layer_w_grad.push_back(hidden_layer_w_grad->addn(regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

		//std::cout << "HIDDENLAYER FIRST:" << hiddenLayerWGrad.size() << "x" << hiddenLayerWGrad[0].size() << std::endl;
		//std::cout << "WEIGHTS SECOND:" << layer.weights.size() << "x" << layer.weights[0].size() << std::endl;

		for (int i = _network.size() - 2; i > _network.size() / 2; i--) {
			layer = _network[i];
			Ref<MLPPHiddenLayer> next_layer = _network[i + 1];

			layer->set_delta(next_layer->get_delta()->multn(next_layer->get_weights()->transposen())->hadamard_productn(avn.run_activation_deriv_matrix(layer->get_activation(), layer->get_z())));

			hidden_layer_w_grad = layer->get_input()->transposen()->multn(layer->get_delta());
			data.cumulative_hidden_layer_w_grad.push_back(hidden_layer_w_grad->addn(regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}

	return data;
}

Vector<Ref<MLPPMatrix>> MLPPWGAN::compute_generator_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set) {
	class MLPPCost cost;
	MLPPActivation avn;
	MLPPReg regularization;

	Vector<Ref<MLPPMatrix>> cumulative_hidden_layer_w_grad; // Tensor containing ALL hidden grads.

	Ref<MLPPVector> cost_deriv_vector = cost.run_cost_deriv_vector(_output_layer->get_cost(), y_hat, output_set);
	Ref<MLPPVector> activation_deriv_vector = avn.run_activation_deriv_vector(_output_layer->get_activation(), _output_layer->get_z());

	_output_layer->set_delta(cost_deriv_vector->hadamard_productn(activation_deriv_vector));

	Ref<MLPPVector> output_w_grad = _output_layer->get_input()->transposen()->mult_vec(_output_layer->get_delta());
	output_w_grad->add(regularization.reg_deriv_termv(_output_layer->get_weights(), _output_layer->get_lambda(), _output_layer->get_alpha(), _output_layer->get_reg()));

	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[_network.size() - 1];

		Ref<MLPPMatrix> activation_deriv_matrix = avn.run_activation_deriv_matrix(layer->get_activation(), layer->get_z());
		layer->set_delta(_output_layer->get_delta()->outer_product(_output_layer->get_weights())->hadamard_productn(activation_deriv_matrix));

		Ref<MLPPMatrix> hidden_layer_w_grad = layer->get_input()->transposen()->multn(layer->get_delta());

		cumulative_hidden_layer_w_grad.push_back(hidden_layer_w_grad->addn(regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

		for (int i = _network.size() - 2; i >= 0; i--) {
			layer = _network[i];
			Ref<MLPPHiddenLayer> next_layer = _network[i + 1];

			activation_deriv_matrix = avn.run_activation_deriv_matrix(layer->get_activation(), layer->get_z());

			layer->set_delta(next_layer->get_delta()->multn(next_layer->get_weights()->transposen())->hadamard_productn(activation_deriv_matrix));
			hidden_layer_w_grad = layer->get_input()->transposen()->multn(layer->get_delta());

			cumulative_hidden_layer_w_grad.push_back(hidden_layer_w_grad->addn(regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}

	return cumulative_hidden_layer_w_grad;
}

void MLPPWGAN::handle_ui(int epoch, real_t cost_prev, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set) {
	MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, output_set));

	PLOG_MSG("Layer " + itos(_network.size() + 1) + ":");

	MLPPUtilities::print_ui_vb(_output_layer->get_weights(), _output_layer->get_bias());

	if (!_network.empty()) {
		for (int i = _network.size() - 1; i >= 0; i--) {
			Ref<MLPPHiddenLayer> layer = _network[i];

			PLOG_MSG("Layer " + itos(i + 1) + ":");

			MLPPUtilities::print_ui_mb(layer->get_weights(), layer->get_bias());
		}
	}
}

void MLPPWGAN::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPWGAN::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "val"), &MLPPWGAN::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_k"), &MLPPWGAN::get_k);
	ClassDB::bind_method(D_METHOD("set_k", "val"), &MLPPWGAN::set_k);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "k"), "set_k", "get_k");

	ClassDB::bind_method(D_METHOD("generate_example", "n"), &MLPPWGAN::generate_example);
	ClassDB::bind_method(D_METHOD("gradient_descent", "learning_rate", "max_epoch", "ui"), &MLPPWGAN::gradient_descent, false);
	ClassDB::bind_method(D_METHOD("score"), &MLPPWGAN::score);
	ClassDB::bind_method(D_METHOD("save", "file_name"), &MLPPWGAN::save);

	ClassDB::bind_method(D_METHOD("create_layer", "activation", "weight_init", "reg", "lambda", "alpha"), &MLPPWGAN::create_layer, MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT, MLPPReg::REGULARIZATION_TYPE_NONE, 0.5, 0.5);
	ClassDB::bind_method(D_METHOD("add_layer", "layer"), &MLPPWGAN::add_layer);
	ClassDB::bind_method(D_METHOD("get_layer", "index"), &MLPPWGAN::get_layer);
	ClassDB::bind_method(D_METHOD("remove_layer", "index"), &MLPPWGAN::remove_layer);
	ClassDB::bind_method(D_METHOD("get_layer_count"), &MLPPWGAN::score);

	ClassDB::bind_method(D_METHOD("add_output_layer", "weight_init", "reg", "lambda", "alpha"), &MLPPWGAN::add_output_layer, MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT, MLPPReg::REGULARIZATION_TYPE_NONE, 0.5, 0.5);
}
