//
//  WGAN.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "wgan.h"
#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include "core/object/method_bind_ext.gen.inc"

#include <cmath>
#include <iostream>

Ref<MLPPMatrix> MLPPWGAN::get_output_set() {
	return output_set;
}
void MLPPWGAN::set_output_set(const Ref<MLPPMatrix> &val) {
	output_set = val;
}

int MLPPWGAN::get_k() const {
	return k;
}
void MLPPWGAN::set_k(const int val) {
	k = val;
}

Ref<MLPPMatrix> MLPPWGAN::generate_example(int n) {
	MLPPLinAlg alg;

	return model_set_test_generator(alg.gaussian_noise(n, k));
}

void MLPPWGAN::gradient_descent(real_t learning_rate, int max_epoch, bool ui) {
	//MLPPCost mlpp_cost;
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;

	forward_pass();

	const int CRITIC_INTERATIONS = 5; // Wasserstein GAN specific parameter.

	while (true) {
		cost_prev = cost(y_hat, alg.onevecv(n));

		Ref<MLPPMatrix> generator_input_set;
		Ref<MLPPMatrix> discriminator_input_set;

		Ref<MLPPVector> ly_hat;
		Ref<MLPPVector> loutput_set;

		// Training of the discriminator.
		for (int i = 0; i < CRITIC_INTERATIONS; i++) {
			generator_input_set = alg.gaussian_noise(n, k);
			discriminator_input_set->set_from_mlpp_matrix(model_set_test_generator(generator_input_set));
			discriminator_input_set->add_rows_mlpp_matrix(output_set); // Fake + real inputs.

			ly_hat = model_set_test_discriminator(discriminator_input_set);
			loutput_set = alg.scalar_multiplym(-1, alg.onevecv(n)); // WGAN changes y_i = 1 and y_i = 0 to y_i = 1 and y_i = -1
			Ref<MLPPVector> output_set_real = alg.onevecv(n);
			loutput_set->add_mlpp_vector(output_set_real); // Fake + real output scores.

			DiscriminatorGradientResult discriminator_gradient_results = compute_discriminator_gradients(ly_hat, loutput_set);
			Vector<Ref<MLPPMatrix>> cumulative_discriminator_hidden_layer_w_grad = discriminator_gradient_results.cumulative_hidden_layer_w_grad;
			Ref<MLPPVector> output_discriminator_w_grad = discriminator_gradient_results.output_w_grad;

			cumulative_discriminator_hidden_layer_w_grad = alg.scalar_multiply_vm(learning_rate / n, cumulative_discriminator_hidden_layer_w_grad);
			output_discriminator_w_grad = alg.scalar_multiplynv(learning_rate / n, output_discriminator_w_grad);
			update_discriminator_parameters(cumulative_discriminator_hidden_layer_w_grad, output_discriminator_w_grad, learning_rate);
		}

		// Training of the generator.
		generator_input_set = alg.gaussian_noise(n, k);
		discriminator_input_set = model_set_test_generator(generator_input_set);
		ly_hat = model_set_test_discriminator(discriminator_input_set);
		loutput_set = alg.onevecv(n);

		Vector<Ref<MLPPMatrix>> cumulative_generator_hidden_layer_w_grad = compute_generator_gradients(y_hat, loutput_set);
		cumulative_generator_hidden_layer_w_grad = alg.scalar_multiply_vm(learning_rate / n, cumulative_generator_hidden_layer_w_grad);
		update_generator_parameters(cumulative_generator_hidden_layer_w_grad, learning_rate);

		forward_pass();

		if (ui) {
			handle_ui(epoch, cost_prev, y_hat, alg.onevecv(n));
		}

		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
}

real_t MLPPWGAN::score() {
	MLPPLinAlg alg;
	MLPPUtilities util;
	forward_pass();
	return util.performance_vec(y_hat, alg.onevecv(n));
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

void MLPPWGAN::add_layer(int n_hidden, MLPPActivation::ActivationFunction activation, MLPPUtilities::WeightDistributionType weight_init, MLPPReg::RegularizationType reg, real_t lambda, real_t alpha) {
	MLPPLinAlg alg;

	Ref<MLPPHiddenLayer> layer;
	layer.instance();

	layer->set_n_hidden(n_hidden);
	layer->set_activation(activation);
	layer->set_weight_init(weight_init);
	layer->set_reg(reg);
	layer->set_lambda(lambda);
	layer->set_alpha(alpha);

	if (network.empty()) {
		layer->set_input(alg.gaussian_noise(n, k));
	} else {
		layer->set_input(network.write[network.size() - 1]->get_a());
	}

	network.push_back(layer);
	layer->forward_pass();
}

void MLPPWGAN::add_output_layer(MLPPUtilities::WeightDistributionType weight_init, MLPPReg::RegularizationType reg, real_t lambda, real_t alpha) {
	ERR_FAIL_COND(network.empty());

	if (!output_layer.is_valid()) {
		output_layer.instance();
	}

	output_layer->set_n_hidden(network[network.size() - 1]->get_n_hidden());
	output_layer->set_activation(MLPPActivation::ACTIVATION_FUNCTION_LINEAR);
	output_layer->set_cost(MLPPCost::COST_TYPE_WASSERSTEIN_LOSS);
	output_layer->set_input(network.write[network.size() - 1]->get_a());
	output_layer->set_weight_init(weight_init);
	output_layer->set_lambda(lambda);
	output_layer->set_alpha(alpha);
}

MLPPWGAN::MLPPWGAN(real_t p_k, const Ref<MLPPMatrix> &p_output_set) {
	output_set = p_output_set;
	n = p_output_set->size().y;
	k = p_k;
}

MLPPWGAN::MLPPWGAN() {
	n = 0;
	k = 0;
}

MLPPWGAN::~MLPPWGAN() {
}

Ref<MLPPMatrix> MLPPWGAN::model_set_test_generator(const Ref<MLPPMatrix> &X) {
	if (!network.empty()) {
		network.write[0]->set_input(X);
		network.write[0]->forward_pass();

		for (int i = 1; i <= network.size() / 2; ++i) {
			network.write[i]->set_input(network.write[i - 1]->get_a());
			network.write[i]->forward_pass();
		}
	}

	return network.write[network.size() / 2]->get_a();
}

Ref<MLPPVector> MLPPWGAN::model_set_test_discriminator(const Ref<MLPPMatrix> &X) {
	if (!network.empty()) {
		for (int i = network.size() / 2 + 1; i < network.size(); i++) {
			if (i == network.size() / 2 + 1) {
				network.write[i]->set_input(X);
			} else {
				network.write[i]->set_input(network.write[i - 1]->get_a());
			}
			network.write[i]->forward_pass();
		}

		output_layer->set_input(network.write[network.size() - 1]->get_a());
	}

	output_layer->forward_pass();

	return output_layer->get_a();
}

real_t MLPPWGAN::cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPReg regularization;
	MLPPCost mlpp_cost;

	real_t total_reg_term = 0;

	for (int i = 0; i < network.size() - 1; ++i) {
		Ref<MLPPHiddenLayer> layer = network[i];

		total_reg_term += regularization.reg_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg());
	}

	total_reg_term += regularization.reg_termm(output_layer->get_weights(), output_layer->get_lambda(), output_layer->get_alpha(), output_layer->get_reg());

	return mlpp_cost.run_cost_norm_vector(output_layer->get_cost(), y_hat, y) + total_reg_term;
}

void MLPPWGAN::forward_pass() {
	MLPPLinAlg alg;

	if (!network.empty()) {
		Ref<MLPPHiddenLayer> layer = network[0];

		layer->set_input(alg.gaussian_noise(n, k));
		layer->forward_pass();

		for (int i = 1; i < network.size(); i++) {
			layer = network[i];

			layer->set_input(network.write[i - 1]->get_a());
			layer->forward_pass();
		}

		output_layer->set_input(network.write[network.size() - 1]->get_a());
	} else { // Should never happen, though.
		output_layer->set_input(alg.gaussian_noise(n, k));
	}

	output_layer->forward_pass();

	y_hat->set_from_mlpp_vector(output_layer->get_a());
}

void MLPPWGAN::update_discriminator_parameters(Vector<Ref<MLPPMatrix>> hidden_layer_updations, const Ref<MLPPVector> &output_layer_updation, real_t learning_rate) {
	MLPPLinAlg alg;

	output_layer->set_weights(alg.subtractionnv(output_layer->get_weights(), output_layer_updation));
	output_layer->set_bias(output_layer->get_bias() - learning_rate * alg.sum_elementsv(output_layer->get_delta()) / n);

	if (!network.empty()) {
		Ref<MLPPHiddenLayer> layer = network[network.size() - 1];

		layer->set_weights(alg.subtractionm(layer->get_weights(), hidden_layer_updations[0]));
		layer->set_bias(alg.subtract_matrix_rows(layer->get_bias(), alg.scalar_multiplym(learning_rate / n, layer->get_delta())));

		for (int i = network.size() - 2; i > network.size() / 2; i--) {
			layer = network[i];

			layer->set_weights(alg.subtractionm(layer->get_weights(), hidden_layer_updations[(network.size() - 2) - i + 1]));
			layer->set_bias(alg.subtract_matrix_rows(layer->get_bias(), alg.scalar_multiplym(learning_rate / n, layer->get_delta())));
		}
	}
}

void MLPPWGAN::update_generator_parameters(Vector<Ref<MLPPMatrix>> hidden_layer_updations, real_t learning_rate) {
	MLPPLinAlg alg;

	if (!network.empty()) {
		for (int i = network.size() / 2; i >= 0; i--) {
			Ref<MLPPHiddenLayer> layer = network[i];

			//std::cout << network[i].weights.size() << "x" << network[i].weights[0].size() << std::endl;
			//std::cout << hiddenLayerUpdations[(network.size() - 2) - i + 1].size() << "x" << hiddenLayerUpdations[(network.size() - 2) - i + 1][0].size() << std::endl;
			layer->set_weights(alg.subtractionm(layer->get_weights(), hidden_layer_updations[(network.size() - 2) - i + 1]));
			layer->set_bias(alg.subtract_matrix_rows(layer->get_bias(), alg.scalar_multiplym(learning_rate / n, layer->get_delta())));
		}
	}
}

MLPPWGAN::DiscriminatorGradientResult MLPPWGAN::compute_discriminator_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set) {
	MLPPCost mlpp_cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	DiscriminatorGradientResult data;

	output_layer->set_delta(alg.hadamard_productnv(mlpp_cost.run_cost_deriv_vector(output_layer->get_cost(), y_hat, output_set), avn.run_activation_deriv_vector(output_layer->get_activation(), output_layer->get_z())));

	data.output_w_grad = alg.mat_vec_multv(alg.transposem(output_layer->get_input()), output_layer->get_delta());
	data.output_w_grad = alg.additionnv(data.output_w_grad, regularization.reg_deriv_termv(output_layer->get_weights(), output_layer->get_lambda(), output_layer->get_alpha(), output_layer->get_reg()));

	if (!network.empty()) {
		Ref<MLPPHiddenLayer> layer = network[network.size() - 1];

		layer->set_delta(alg.hadamard_productm(alg.outer_product(output_layer->get_delta(), output_layer->get_weights()), avn.run_activation_deriv_vector(layer->get_activation(), layer->get_z())));
		Ref<MLPPMatrix> hidden_layer_w_grad = alg.matmultm(alg.transposem(layer->get_input()), layer->get_delta());

		data.cumulative_hidden_layer_w_grad.push_back(alg.additionm(hidden_layer_w_grad, regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

		//std::cout << "HIDDENLAYER FIRST:" << hiddenLayerWGrad.size() << "x" << hiddenLayerWGrad[0].size() << std::endl;
		//std::cout << "WEIGHTS SECOND:" << layer.weights.size() << "x" << layer.weights[0].size() << std::endl;

		for (int i = network.size() - 2; i > network.size() / 2; i--) {
			layer = network[i];
			Ref<MLPPHiddenLayer> next_layer = network[i + 1];

			layer->set_delta(alg.hadamard_productm(alg.matmultm(next_layer->get_delta(), alg.transposem(next_layer->get_weights())), avn.run_activation_deriv_matrix(layer->get_activation(), layer->get_z())));

			hidden_layer_w_grad = alg.matmultm(alg.transposem(layer->get_input()), layer->get_delta());

			data.cumulative_hidden_layer_w_grad.push_back(alg.additionm(hidden_layer_w_grad, regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}

	return data;
}

Vector<Ref<MLPPMatrix>> MLPPWGAN::compute_generator_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set) {
	class MLPPCost cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	Vector<Ref<MLPPMatrix>> cumulative_hidden_layer_w_grad; // Tensor containing ALL hidden grads.

	Ref<MLPPVector> cost_deriv_vector = cost.run_cost_deriv_vector(output_layer->get_cost(), y_hat, output_set);
	Ref<MLPPVector> activation_deriv_vector = avn.run_activation_deriv_vector(output_layer->get_activation(), output_layer->get_z());

	output_layer->set_delta(alg.hadamard_productnv(cost_deriv_vector, activation_deriv_vector));

	Ref<MLPPVector> output_w_grad = alg.mat_vec_multv(alg.transposem(output_layer->get_input()), output_layer->get_delta());
	output_w_grad = alg.additionnv(output_w_grad, regularization.reg_deriv_termm(output_layer->get_weights(), output_layer->get_lambda(), output_layer->get_alpha(), output_layer->get_reg()));

	if (!network.empty()) {
		Ref<MLPPHiddenLayer> layer = network[network.size() - 1];

		activation_deriv_vector = avn.run_activation_deriv_vector(layer->get_activation(), output_layer->get_z());
		layer->set_delta(alg.hadamard_productnv(alg.outer_product(output_layer->get_delta(), output_layer->get_weights()), activation_deriv_vector));

		Ref<MLPPMatrix> hidden_layer_w_grad = alg.matmultm(alg.transposem(layer->get_input()), layer->get_delta());
		cumulative_hidden_layer_w_grad.push_back(alg.additionm(hidden_layer_w_grad, regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

		for (int i = network.size() - 2; i >= 0; i--) {
			layer = network[i];
			Ref<MLPPHiddenLayer> next_layer = network[i + 1];

			activation_deriv_vector = avn.run_activation_deriv_vector(layer->get_activation(), layer->get_z());

			layer->set_delta(alg.hadamard_productm(alg.matmultm(next_layer->get_delta(), alg.transposem(next_layer->get_weights())), activation_deriv_vector));
			hidden_layer_w_grad = alg.matmultm(alg.transposem(layer->get_input()), layer->get_delta());
			cumulative_hidden_layer_w_grad.push_back(alg.additionm(hidden_layer_w_grad, regularization.reg_deriv_termm(layer->get_weights(), layer->get_lambda(), layer->get_alpha(), layer->get_reg()))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}

	return cumulative_hidden_layer_w_grad;
}

void MLPPWGAN::handle_ui(int epoch, real_t cost_prev, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set) {
	MLPPUtilities::cost_info(epoch, cost_prev, cost(y_hat, output_set));
	std::cout << "Layer " << network.size() + 1 << ": " << std::endl;

	MLPPUtilities::print_ui_vb(output_layer->get_weights(), output_layer->get_bias());

	if (!network.empty()) {
		for (int i = network.size() - 1; i >= 0; i--) {
			Ref<MLPPHiddenLayer> layer = network[i];

			std::cout << "Layer " << i + 1 << ": " << std::endl;

			MLPPUtilities::print_ui_vib(layer->get_weights(), layer->get_bias(), 0);
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

	ClassDB::bind_method(D_METHOD("add_layer", "activation", "weight_init", "reg", "lambda", "alpha"), &MLPPWGAN::add_layer, MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT, MLPPReg::REGULARIZATION_TYPE_NONE, 0.5, 0.5);
	ClassDB::bind_method(D_METHOD("add_output_layer", "weight_init", "reg", "lambda", "alpha"), &MLPPWGAN::add_output_layer, MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT, MLPPReg::REGULARIZATION_TYPE_NONE, 0.5, 0.5);
}

// ========  OLD  ==========

MLPPWGANOld::MLPPWGANOld(real_t k, std::vector<std::vector<real_t>> outputSet) :
		outputSet(outputSet), n(outputSet.size()), k(k) {
}

MLPPWGANOld::~MLPPWGANOld() {
	delete outputLayer;
}

std::vector<std::vector<real_t>> MLPPWGANOld::generateExample(int n) {
	MLPPLinAlg alg;
	return modelSetTestGenerator(alg.gaussianNoise(n, k));
}

void MLPPWGANOld::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	class MLPPCost cost;
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();

	const int CRITIC_INTERATIONS = 5; // Wasserstein GAN specific parameter.

	while (true) {
		cost_prev = Cost(y_hat, alg.onevec(n));

		std::vector<std::vector<real_t>> generatorInputSet;
		std::vector<std::vector<real_t>> discriminatorInputSet;

		std::vector<real_t> y_hat;
		std::vector<real_t> outputSet;

		// Training of the discriminator.
		for (int i = 0; i < CRITIC_INTERATIONS; i++) {
			generatorInputSet = alg.gaussianNoise(n, k);
			discriminatorInputSet = modelSetTestGenerator(generatorInputSet);
			discriminatorInputSet.insert(discriminatorInputSet.end(), MLPPWGANOld::outputSet.begin(), MLPPWGANOld::outputSet.end()); // Fake + real inputs.

			y_hat = modelSetTestDiscriminator(discriminatorInputSet);
			outputSet = alg.scalarMultiply(-1, alg.onevec(n)); // WGAN changes y_i = 1 and y_i = 0 to y_i = 1 and y_i = -1
			std::vector<real_t> outputSetReal = alg.onevec(n);
			outputSet.insert(outputSet.end(), outputSetReal.begin(), outputSetReal.end()); // Fake + real output scores.

			auto discriminator_gradient_results = computeDiscriminatorGradients(y_hat, outputSet);
			auto cumulativeDiscriminatorHiddenLayerWGrad = std::get<0>(discriminator_gradient_results);
			auto outputDiscriminatorWGrad = std::get<1>(discriminator_gradient_results);

			cumulativeDiscriminatorHiddenLayerWGrad = alg.scalarMultiply(learning_rate / n, cumulativeDiscriminatorHiddenLayerWGrad);
			outputDiscriminatorWGrad = alg.scalarMultiply(learning_rate / n, outputDiscriminatorWGrad);
			updateDiscriminatorParameters(cumulativeDiscriminatorHiddenLayerWGrad, outputDiscriminatorWGrad, learning_rate);
		}

		// Training of the generator.
		generatorInputSet = alg.gaussianNoise(n, k);
		discriminatorInputSet = modelSetTestGenerator(generatorInputSet);
		y_hat = modelSetTestDiscriminator(discriminatorInputSet);
		outputSet = alg.onevec(n);

		std::vector<std::vector<std::vector<real_t>>> cumulativeGeneratorHiddenLayerWGrad = computeGeneratorGradients(y_hat, outputSet);
		cumulativeGeneratorHiddenLayerWGrad = alg.scalarMultiply(learning_rate / n, cumulativeGeneratorHiddenLayerWGrad);
		updateGeneratorParameters(cumulativeGeneratorHiddenLayerWGrad, learning_rate);

		forwardPass();

		if (UI) {
			MLPPWGANOld::UI(epoch, cost_prev, MLPPWGANOld::y_hat, alg.onevec(n));
		}

		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
}

real_t MLPPWGANOld::score() {
	MLPPLinAlg alg;
	MLPPUtilities util;
	forwardPass();
	return util.performance(y_hat, alg.onevec(n));
}

void MLPPWGANOld::save(std::string fileName) {
	MLPPUtilities util;
	if (!network.empty()) {
		util.saveParameters(fileName, network[0].weights, network[0].bias, 0, 1);
		for (uint32_t i = 1; i < network.size(); i++) {
			util.saveParameters(fileName, network[i].weights, network[i].bias, 1, i + 1);
		}
		util.saveParameters(fileName, outputLayer->weights, outputLayer->bias, 1, network.size() + 1);
	} else {
		util.saveParameters(fileName, outputLayer->weights, outputLayer->bias, 0, network.size() + 1);
	}
}

void MLPPWGANOld::addLayer(int n_hidden, std::string activation, std::string weightInit, std::string reg, real_t lambda, real_t alpha) {
	MLPPLinAlg alg;
	if (network.empty()) {
		network.push_back(MLPPOldHiddenLayer(n_hidden, activation, alg.gaussianNoise(n, k), weightInit, reg, lambda, alpha));
		network[0].forwardPass();
	} else {
		network.push_back(MLPPOldHiddenLayer(n_hidden, activation, network[network.size() - 1].a, weightInit, reg, lambda, alpha));
		network[network.size() - 1].forwardPass();
	}
}

void MLPPWGANOld::addOutputLayer(std::string weightInit, std::string reg, real_t lambda, real_t alpha) {
	MLPPLinAlg alg;
	if (!network.empty()) {
		outputLayer = new MLPPOldOutputLayer(network[network.size() - 1].n_hidden, "Linear", "WassersteinLoss", network[network.size() - 1].a, weightInit, "WeightClipping", -0.01, 0.01);
	} else { // Should never happen.
		outputLayer = new MLPPOldOutputLayer(k, "Linear", "WassersteinLoss", alg.gaussianNoise(n, k), weightInit, "WeightClipping", -0.01, 0.01);
	}
}

std::vector<std::vector<real_t>> MLPPWGANOld::modelSetTestGenerator(std::vector<std::vector<real_t>> X) {
	if (!network.empty()) {
		network[0].input = X;
		network[0].forwardPass();

		for (uint32_t i = 1; i <= network.size() / 2; i++) {
			network[i].input = network[i - 1].a;
			network[i].forwardPass();
		}
	}
	return network[network.size() / 2].a;
}

std::vector<real_t> MLPPWGANOld::modelSetTestDiscriminator(std::vector<std::vector<real_t>> X) {
	if (!network.empty()) {
		for (uint32_t i = network.size() / 2 + 1; i < network.size(); i++) {
			if (i == network.size() / 2 + 1) {
				network[i].input = X;
			} else {
				network[i].input = network[i - 1].a;
			}
			network[i].forwardPass();
		}
		outputLayer->input = network[network.size() - 1].a;
	}
	outputLayer->forwardPass();
	return outputLayer->a;
}

real_t MLPPWGANOld::Cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPReg regularization;
	class MLPPCost cost;
	real_t totalRegTerm = 0;

	auto cost_function = outputLayer->cost_map[outputLayer->cost];
	if (!network.empty()) {
		for (uint32_t i = 0; i < network.size() - 1; i++) {
			totalRegTerm += regularization.regTerm(network[i].weights, network[i].lambda, network[i].alpha, network[i].reg);
		}
	}
	return (cost.*cost_function)(y_hat, y) + totalRegTerm + regularization.regTerm(outputLayer->weights, outputLayer->lambda, outputLayer->alpha, outputLayer->reg);
}

void MLPPWGANOld::forwardPass() {
	MLPPLinAlg alg;
	if (!network.empty()) {
		network[0].input = alg.gaussianNoise(n, k);
		network[0].forwardPass();

		for (uint32_t i = 1; i < network.size(); i++) {
			network[i].input = network[i - 1].a;
			network[i].forwardPass();
		}
		outputLayer->input = network[network.size() - 1].a;
	} else { // Should never happen, though.
		outputLayer->input = alg.gaussianNoise(n, k);
	}
	outputLayer->forwardPass();
	y_hat = outputLayer->a;
}

void MLPPWGANOld::updateDiscriminatorParameters(std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations, std::vector<real_t> outputLayerUpdation, real_t learning_rate) {
	MLPPLinAlg alg;

	outputLayer->weights = alg.subtraction(outputLayer->weights, outputLayerUpdation);
	outputLayer->bias -= learning_rate * alg.sum_elements(outputLayer->delta) / n;

	if (!network.empty()) {
		network[network.size() - 1].weights = alg.subtraction(network[network.size() - 1].weights, hiddenLayerUpdations[0]);
		network[network.size() - 1].bias = alg.subtractMatrixRows(network[network.size() - 1].bias, alg.scalarMultiply(learning_rate / n, network[network.size() - 1].delta));

		for (uint32_t i = network.size() - 2; i > network.size() / 2; i--) {
			network[i].weights = alg.subtraction(network[i].weights, hiddenLayerUpdations[(network.size() - 2) - i + 1]);
			network[i].bias = alg.subtractMatrixRows(network[i].bias, alg.scalarMultiply(learning_rate / n, network[i].delta));
		}
	}
}

void MLPPWGANOld::updateGeneratorParameters(std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations, real_t learning_rate) {
	MLPPLinAlg alg;

	if (!network.empty()) {
		for (int ii = network.size() / 2; ii >= 0; ii--) {
			uint32_t i = static_cast<uint32_t>(ii);

			//std::cout << network[i].weights.size() << "x" << network[i].weights[0].size() << std::endl;
			//std::cout << hiddenLayerUpdations[(network.size() - 2) - i + 1].size() << "x" << hiddenLayerUpdations[(network.size() - 2) - i + 1][0].size() << std::endl;
			network[i].weights = alg.subtraction(network[i].weights, hiddenLayerUpdations[(network.size() - 2) - i + 1]);
			network[i].bias = alg.subtractMatrixRows(network[i].bias, alg.scalarMultiply(learning_rate / n, network[i].delta));
		}
	}
}

std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<real_t>> MLPPWGANOld::computeDiscriminatorGradients(std::vector<real_t> y_hat, std::vector<real_t> outputSet) {
	class MLPPCost cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	std::vector<std::vector<std::vector<real_t>>> cumulativeHiddenLayerWGrad; // Tensor containing ALL hidden grads.

	auto costDeriv = outputLayer->costDeriv_map[outputLayer->cost];
	auto outputAvn = outputLayer->activation_map[outputLayer->activation];
	outputLayer->delta = alg.hadamard_product((cost.*costDeriv)(y_hat, outputSet), (avn.*outputAvn)(outputLayer->z, 1));
	std::vector<real_t> outputWGrad = alg.mat_vec_mult(alg.transpose(outputLayer->input), outputLayer->delta);
	outputWGrad = alg.addition(outputWGrad, regularization.regDerivTerm(outputLayer->weights, outputLayer->lambda, outputLayer->alpha, outputLayer->reg));

	if (!network.empty()) {
		auto hiddenLayerAvn = network[network.size() - 1].activation_map[network[network.size() - 1].activation];

		network[network.size() - 1].delta = alg.hadamard_product(alg.outerProduct(outputLayer->delta, outputLayer->weights), (avn.*hiddenLayerAvn)(network[network.size() - 1].z, 1));
		std::vector<std::vector<real_t>> hiddenLayerWGrad = alg.matmult(alg.transpose(network[network.size() - 1].input), network[network.size() - 1].delta);

		cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(network[network.size() - 1].weights, network[network.size() - 1].lambda, network[network.size() - 1].alpha, network[network.size() - 1].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

		//std::cout << "HIDDENLAYER FIRST:" << hiddenLayerWGrad.size() << "x" << hiddenLayerWGrad[0].size() << std::endl;
		//std::cout << "WEIGHTS SECOND:" << network[network.size() - 1].weights.size() << "x" << network[network.size() - 1].weights[0].size() << std::endl;

		for (uint32_t i = network.size() - 2; i > network.size() / 2; i--) {
			auto hiddenLayerAvnl = network[i].activation_map[network[i].activation];
			network[i].delta = alg.hadamard_product(alg.matmult(network[i + 1].delta, alg.transpose(network[i + 1].weights)), (avn.*hiddenLayerAvnl)(network[i].z, 1));
			std::vector<std::vector<real_t>> hiddenLayerWGradl = alg.matmult(alg.transpose(network[i].input), network[i].delta);

			cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGradl, regularization.regDerivTerm(network[i].weights, network[i].lambda, network[i].alpha, network[i].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}
	return { cumulativeHiddenLayerWGrad, outputWGrad };
}

std::vector<std::vector<std::vector<real_t>>> MLPPWGANOld::computeGeneratorGradients(std::vector<real_t> y_hat, std::vector<real_t> outputSet) {
	class MLPPCost cost;
	MLPPActivation avn;
	MLPPLinAlg alg;
	MLPPReg regularization;

	std::vector<std::vector<std::vector<real_t>>> cumulativeHiddenLayerWGrad; // Tensor containing ALL hidden grads.

	auto costDeriv = outputLayer->costDeriv_map[outputLayer->cost];
	auto outputAvn = outputLayer->activation_map[outputLayer->activation];
	outputLayer->delta = alg.hadamard_product((cost.*costDeriv)(y_hat, outputSet), (avn.*outputAvn)(outputLayer->z, 1));
	std::vector<real_t> outputWGrad = alg.mat_vec_mult(alg.transpose(outputLayer->input), outputLayer->delta);
	outputWGrad = alg.addition(outputWGrad, regularization.regDerivTerm(outputLayer->weights, outputLayer->lambda, outputLayer->alpha, outputLayer->reg));
	if (!network.empty()) {
		auto hiddenLayerAvn = network[network.size() - 1].activation_map[network[network.size() - 1].activation];
		network[network.size() - 1].delta = alg.hadamard_product(alg.outerProduct(outputLayer->delta, outputLayer->weights), (avn.*hiddenLayerAvn)(network[network.size() - 1].z, 1));
		std::vector<std::vector<real_t>> hiddenLayerWGrad = alg.matmult(alg.transpose(network[network.size() - 1].input), network[network.size() - 1].delta);
		cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(network[network.size() - 1].weights, network[network.size() - 1].lambda, network[network.size() - 1].alpha, network[network.size() - 1].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.

		for (int ii = network.size() - 2; ii >= 0; ii--) {
			uint32_t i = static_cast<uint32_t>(ii);
			auto hiddenLayerAvnl = network[i].activation_map[network[i].activation];
			network[i].delta = alg.hadamard_product(alg.matmult(network[i + 1].delta, alg.transpose(network[i + 1].weights)), (avn.*hiddenLayerAvnl)(network[i].z, 1));
			std::vector<std::vector<real_t>> hiddenLayerWGradl = alg.matmult(alg.transpose(network[i].input), network[i].delta);
			cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGradl, regularization.regDerivTerm(network[i].weights, network[i].lambda, network[i].alpha, network[i].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}
	return cumulativeHiddenLayerWGrad;
}

void MLPPWGANOld::UI(int epoch, real_t cost_prev, std::vector<real_t> y_hat, std::vector<real_t> outputSet) {
	MLPPUtilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
	std::cout << "Layer " << network.size() + 1 << ": " << std::endl;
	MLPPUtilities::UI(outputLayer->weights, outputLayer->bias);
	if (!network.empty()) {
		for (int ii = network.size() - 1; ii >= 0; ii--) {
			uint32_t i = static_cast<uint32_t>(ii);

			std::cout << "Layer " << i + 1 << ": " << std::endl;
			MLPPUtilities::UI(network[i].weights, network[i].bias);
		}
	}
}
