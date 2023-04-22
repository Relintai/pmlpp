//
//  GAN.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "gan.h"
#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include "core/log/logger.h"

#include <cmath>
#include <iostream>

/*
Ref<MLPPMatrix> MLPPGAN::get_input_set() {
	return _input_set;
}
void MLPPGAN::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;
}

Ref<MLPPVector> MLPPGAN::get_output_set() {
	return _output_set;
}
void MLPPGAN::set_output_set(const Ref<MLPPVector> &val) {
	_output_set = val;
}

int MLPPGAN::get_k() {
	return _k;
}
void MLPPGAN::set_k(const int val) {
	_k = val;
}
*/

Ref<MLPPMatrix> MLPPGAN::generate_example(int n) {
	MLPPLinAlg alg;

	return model_set_test_generator(alg.gaussian_noise(n, _k));
}

void MLPPGAN::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	MLPPCost mlpp_cost;
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	while (true) {
		cost_prev = cost(_y_hat, alg.onevecnv(_n));

		// Training of the discriminator.

		Ref<MLPPMatrix> generator_input_set = alg.gaussian_noise(_n, _k);
		Ref<MLPPMatrix> discriminator_input_set = model_set_test_generator(generator_input_set);
		discriminator_input_set->add_rows_mlpp_matrix(_output_set); // Fake + real inputs.

		Ref<MLPPVector> y_hat = model_set_test_discriminator(discriminator_input_set);
		Ref<MLPPVector> output_set = alg.zerovecnv(_n);
		Ref<MLPPVector> output_set_real = alg.onevecnv(_n);
		output_set->add_mlpp_vector(output_set_real); // Fake + real output scores.

		ComputeDiscriminatorGradientsResult dgrads = compute_discriminator_gradients(y_hat, _output_set);

		dgrads.cumulative_hidden_layer_w_grad = alg.scalar_multiplynvt(learning_rate / _n, dgrads.cumulative_hidden_layer_w_grad);
		dgrads.output_w_grad = alg.scalar_multiplynv(learning_rate / _n, dgrads.output_w_grad);
		update_discriminator_parameters(dgrads.cumulative_hidden_layer_w_grad, dgrads.output_w_grad, learning_rate);

		// Training of the generator.
		generator_input_set = alg.gaussian_noise(_n, _k);
		discriminator_input_set = model_set_test_generator(generator_input_set);
		y_hat = model_set_test_discriminator(discriminator_input_set);
		_output_set = alg.onevecnv(_n);

		Vector<Ref<MLPPMatrix>> cumulative_generator_hidden_layer_w_grad = compute_generator_gradients(y_hat, _output_set);
		cumulative_generator_hidden_layer_w_grad = alg.scalar_multiplynvt(learning_rate / _n, cumulative_generator_hidden_layer_w_grad);
		update_generator_parameters(cumulative_generator_hidden_layer_w_grad, learning_rate);

		forward_pass();

		if (ui) {
			print_ui(epoch, cost_prev, _y_hat, alg.onevecnv(_n));
		}

		epoch++;

		if (epoch > max_epoch) {
			break;
		}
	}
}

real_t MLPPGAN::score() {
	MLPPLinAlg alg;
	MLPPUtilities util;

	forward_pass();

	return util.performance_vec(_y_hat, alg.onevecnv(_n));
}

void MLPPGAN::save(const String &file_name) {
	MLPPUtilities util;
	/*
	if (!_network.empty()) {
		util.saveParameters(file_name, _network[0].weights, _network[0].bias, false, 1);
		for (uint32_t i = 1; i < _network.size(); i++) {
			util.saveParameters(file_name, _network[i].weights, _network[i].bias, true, i + 1);
		}
		util.saveParameters(file_name, _output_layer->weights, _output_layer->bias, true, _network.size() + 1);
	} else {
		util.saveParameters(file_name, _output_layer->weights, _output_layer->bias, false, _network.size() + 1);
	}
	*/
}

void MLPPGAN::add_layer(int n_hidden, MLPPActivation::ActivationFunction activation, MLPPUtilities::WeightDistributionType weight_init, MLPPReg::RegularizationType reg, real_t lambda, real_t alpha) {
	MLPPLinAlg alg;
	if (_network.empty()) {
		Ref<MLPPHiddenLayer> layer = Ref<MLPPHiddenLayer>(memnew(MLPPHiddenLayer(n_hidden, activation, alg.gaussian_noise(_n, _k), weight_init, reg, lambda, alpha)));

		_network.push_back(layer);

		_network.write[0]->forward_pass();
	} else {
		Ref<MLPPHiddenLayer> layer = Ref<MLPPHiddenLayer>(memnew(MLPPHiddenLayer(n_hidden, activation, _network.write[_network.size() - 1]->get_a(), weight_init, reg, lambda, alpha)));

		_network.push_back(layer);

		_network.write[_network.size() - 1]->forward_pass();
	}
}

void MLPPGAN::add_output_layer(MLPPUtilities::WeightDistributionType weight_init, MLPPReg::RegularizationType reg, real_t lambda, real_t alpha) {
	MLPPLinAlg alg;

	if (!_network.empty()) {
		_output_layer = Ref<MLPPOutputLayer>(memnew(MLPPOutputLayer(_network.write[_network.size() - 1]->get_n_hidden(), MLPPActivation::ACTIVATION_FUNCTION_SIGMOID, MLPPCost::COST_TYPE_LOGISTIC_LOSS, _network.write[_network.size() - 1]->get_a(), weight_init, reg, lambda, alpha)));
	} else {
		_output_layer = Ref<MLPPOutputLayer>(memnew(MLPPOutputLayer(_k, MLPPActivation::ACTIVATION_FUNCTION_SIGMOID, MLPPCost::COST_TYPE_LOGISTIC_LOSS, alg.gaussian_noise(_n, _k), weight_init, reg, lambda, alpha)));
	}
}

MLPPGAN::MLPPGAN(real_t k, const Ref<MLPPMatrix> &output_set) {
	_output_set = output_set;
	_n = _output_set->size().y;
	_k = k;
}

MLPPGAN::MLPPGAN() {
}

MLPPGAN::~MLPPGAN() {
}

Ref<MLPPMatrix> MLPPGAN::model_set_test_generator(const Ref<MLPPMatrix> &X) {
	if (!_network.empty()) {
		_network.write[0]->set_input(X);
		_network.write[0]->forward_pass();

		for (int i = 1; i <= _network.size() / 2; i++) {
			_network.write[i]->set_input(_network.write[i - 1]->get_a());
			_network.write[i]->forward_pass();
		}
	}

	return _network.write[_network.size() / 2]->get_a();
}

Ref<MLPPVector> MLPPGAN::model_set_test_discriminator(const Ref<MLPPMatrix> &X) {
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

real_t MLPPGAN::cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;
	real_t total_reg_term = 0;

	if (!_network.empty()) {
		for (int i = 0; i < _network.size() - 1; i++) {
			total_reg_term += regularization.reg_termm(_network.write[i]->get_weights(), _network.write[i]->get_lambda(), _network.write[i]->get_alpha(), _network.write[i]->get_reg());
		}
	}

	return mlpp_cost.run_cost_norm_vector(_output_layer->get_cost(), y_hat, y) + total_reg_term + regularization.reg_termv(_output_layer->get_weights(), _output_layer->get_lambda(), _output_layer->get_alpha(), _output_layer->get_reg());
}

void MLPPGAN::forward_pass() {
	MLPPLinAlg alg;

	if (!_network.empty()) {
		_network.write[0]->set_input(alg.gaussian_noise(_n, _k));
		_network.write[0]->forward_pass();

		for (int i = 1; i < _network.size(); i++) {
			_network.write[i]->set_input(_network.write[i - 1]->get_a());
			_network.write[i]->forward_pass();
		}
		_output_layer->set_input(_network.write[_network.size() - 1]->get_a());
	} else { // Should never happen, though.
		_output_layer->set_input(alg.gaussian_noise(_n, _k));
	}

	_output_layer->forward_pass();
	_y_hat = _output_layer->get_a();
}

void MLPPGAN::update_discriminator_parameters(const Vector<Ref<MLPPMatrix>> &hidden_layer_updations, const Ref<MLPPVector> &output_layer_updation, real_t learning_rate) {
	MLPPLinAlg alg;

	_output_layer->set_weights(alg.subtractionnv(_output_layer->get_weights(), output_layer_updation));
	real_t output_layer_bias = _output_layer->get_bias();
	output_layer_bias -= learning_rate * alg.sum_elementsv(_output_layer->get_delta()) / _n;
	_output_layer->set_bias(output_layer_bias);

	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[_network.size() - 1];

		layer->set_weights(alg.subtractionnm(layer->get_weights(), hidden_layer_updations[0]));
		layer->set_bias(alg.subtract_matrix_rowsnv(layer->get_bias(), alg.scalar_multiplynm(learning_rate / _n, layer->get_delta())));

		for (int i = _network.size() - 2; i > _network.size() / 2; i--) {
			layer = _network[i];

			layer->set_weights(alg.subtractionnm(layer->get_weights(), hidden_layer_updations[(_network.size() - 2) - i + 1]));
			layer->set_bias(alg.subtract_matrix_rowsnv(layer->get_bias(), alg.scalar_multiplynm(learning_rate / _n, layer->get_delta())));
		}
	}
}

void MLPPGAN::update_generator_parameters(const Vector<Ref<MLPPMatrix>> &hidden_layer_updations, real_t learning_rate) {
	MLPPLinAlg alg;

	if (!_network.empty()) {
		for (int i = _network.size() / 2; i >= 0; i--) {
			Ref<MLPPHiddenLayer> layer = _network[i];

			//std::cout << network[i].weights.size() << "x" << network[i].weights[0].size() << std::endl;
			//std::cout << hidden_layer_updations[(network.size() - 2) - i + 1].size() << "x" << hidden_layer_updations[(network.size() - 2) - i + 1][0].size() << std::endl;
			layer->set_weights(alg.subtractionnm(layer->get_weights(), hidden_layer_updations[(_network.size() - 2) - i + 1]));
			layer->set_bias(alg.subtract_matrix_rowsnv(layer->get_bias(), alg.scalar_multiplynm(learning_rate / _n, layer->get_delta())));
		}
	}
}

MLPPGAN::ComputeDiscriminatorGradientsResult MLPPGAN::compute_discriminator_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set) {
	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	ComputeDiscriminatorGradientsResult res;

	Ref<MLPPVector> cost_deriv = mlpp_cost.run_cost_deriv_vector(_output_layer->get_cost(), y_hat, _output_set);
	Ref<MLPPVector> activ_deriv = avn.run_activation_deriv_vector(_output_layer->get_activation(), _output_layer->get_z());

	_output_layer->set_delta(alg.hadamard_productnv(cost_deriv, activ_deriv));

	res.output_w_grad = alg.mat_vec_multnv(alg.transposenm(_output_layer->get_input()), _output_layer->get_delta());
	res.output_w_grad = alg.additionnv(res.output_w_grad, regularization.reg_deriv_termv(_output_layer->get_weights(), _output_layer->get_lambda(), _output_layer->get_alpha(), _output_layer->get_reg()));

	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[_network.size() - 1];

		Ref<MLPPVector> hidden_layer_activ_deriv = avn.run_activation_deriv_vector(layer->get_activation(), layer->get_z());

		layer->set_delta(alg.hadamard_productnm(alg.outer_product(_output_layer->get_delta(), _output_layer->get_weights()), hidden_layer_activ_deriv));
		Ref<MLPPMatrix> hidden_layer_w_grad = alg.matmultnm(alg.transposenm(layer->get_input()), layer->get_delta());

		res.cumulative_hidden_layer_w_grad.push_back(alg.additionnm(hidden_layer_w_grad, regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

		for (int i = static_cast<int>(_network.size()) - 2; i > static_cast<int>(_network.size()) / 2; i--) {
			layer = _network[i];
			Ref<MLPPHiddenLayer> next_layer = _network[i + 1];

			hidden_layer_activ_deriv = avn.run_activation_deriv_vector(layer->get_activation(), layer->get_z());

			layer->set_delta(alg.hadamard_productnm(alg.matmultnm(next_layer->get_delta(), alg.transposenm(next_layer->get_weights())), hidden_layer_activ_deriv));
			hidden_layer_w_grad = alg.matmultnm(alg.transposenm(layer->get_input()), layer->get_delta());

			res.cumulative_hidden_layer_w_grad.push_back(alg.additionnm(hidden_layer_w_grad, regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}

	return res;
}

Vector<Ref<MLPPMatrix>> MLPPGAN::compute_generator_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set) {
	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	Vector<Ref<MLPPMatrix>> cumulative_hidden_layer_w_grad; // Tensor containing ALL hidden grads.

	Ref<MLPPVector> cost_deriv = mlpp_cost.run_cost_deriv_vector(_output_layer->get_cost(), y_hat, _output_set);
	Ref<MLPPVector> activ_deriv = avn.run_activation_deriv_vector(_output_layer->get_activation(), _output_layer->get_z());

	_output_layer->set_delta(alg.hadamard_productnv(cost_deriv, activ_deriv));

	Ref<MLPPVector> output_w_grad = alg.mat_vec_multnv(alg.transposenm(_output_layer->get_input()), _output_layer->get_delta());
	output_w_grad = alg.additionnv(output_w_grad, regularization.reg_deriv_termv(_output_layer->get_weights(), _output_layer->get_lambda(), _output_layer->get_alpha(), _output_layer->get_reg()));

	if (!_network.empty()) {
		Ref<MLPPHiddenLayer> layer = _network[_network.size() - 1];

		Ref<MLPPVector> hidden_layer_activ_deriv = avn.run_activation_deriv_vector(layer->get_activation(), layer->get_z());

		layer->set_delta(alg.hadamard_productnv(alg.outer_product(_output_layer->get_delta(), _output_layer->get_weights()), hidden_layer_activ_deriv));

		Ref<MLPPMatrix> hidden_layer_w_grad = alg.matmultnm(alg.transposenm(layer->get_input()), layer->get_delta());
		cumulative_hidden_layer_w_grad.push_back(alg.additionnm(hidden_layer_w_grad, regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

		for (int i = _network.size() - 2; i >= 0; i--) {
			layer = _network[i];
			Ref<MLPPHiddenLayer> next_layer = _network[i + 1];

			hidden_layer_activ_deriv = avn.run_activation_deriv_vector(layer->get_activation(), layer->get_z());

			layer->set_delta(alg.hadamard_productnm(alg.matmultnm(next_layer->get_delta(), alg.transposenm(next_layer->get_weights())), hidden_layer_activ_deriv));

			hidden_layer_w_grad = alg.matmultnm(alg.transposenm(layer->get_input()), layer->get_delta());
			cumulative_hidden_layer_w_grad.push_back(alg.additionnm(hidden_layer_w_grad, regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}

	return cumulative_hidden_layer_w_grad;
}

void MLPPGAN::print_ui(int epoch, real_t cost_prev, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set) {
	MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, _output_set));

	PLOG_MSG("Layer " + itos(_network.size() + 1) + ": ");
	MLPPUtilities::print_ui_vb(_output_layer->get_weights(), _output_layer->get_bias());

	if (!_network.empty()) {
		for (int i = _network.size() - 1; i >= 0; i--) {
			Ref<MLPPHiddenLayer> layer = _network[i];

			PLOG_MSG("Layer " + itos(i + 1) + ": ");
			MLPPUtilities::print_ui_mb(layer->get_weights(), layer->get_bias());
		}
	}
}

void MLPPGAN::_bind_methods() {
	/*
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPGAN::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "value"), &MLPPGAN::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPGAN::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "value"), &MLPPGAN::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_k"), &MLPPGAN::get_k);
	ClassDB::bind_method(D_METHOD("set_k", "value"), &MLPPGAN::set_k);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "k"), "set_k", "get_k");

	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPGAN::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPGAN::model_test);
	ClassDB::bind_method(D_METHOD("score"), &MLPPGAN::score);
	*/
}
