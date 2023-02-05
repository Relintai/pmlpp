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

#include <cmath>
#include <iostream>

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
		for (uint32_t i = network.size() / 2; i >= 0; i--) {
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

		for (uint32_t i = network.size() - 2; i >= 0; i--) {
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
		for (uint32_t i = network.size() - 1; i >= 0; i--) {
			std::cout << "Layer " << i + 1 << ": " << std::endl;
			MLPPUtilities::UI(network[i].weights, network[i].bias);
		}
	}
}
