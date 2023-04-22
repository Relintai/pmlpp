//
//  GAN.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "gan_old.h"
#include "../activation/activation_old.h"
#include "../cost/cost_old.h"
#include "../lin_alg/lin_alg_old.h"
#include "../regularization/reg_old.h"
#include "../utilities/utilities.h"

#include <cmath>
#include <iostream>

MLPPGANOld::MLPPGANOld(real_t k, std::vector<std::vector<real_t>> outputSet) :
		outputSet(outputSet), n(outputSet.size()), k(k) {
}

MLPPGANOld::~MLPPGANOld() {
	delete outputLayer;
}

std::vector<std::vector<real_t>> MLPPGANOld::generateExample(int n) {
	MLPPLinAlgOld alg;
	return modelSetTestGenerator(alg.gaussianNoise(n, k));
}

void MLPPGANOld::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	class MLPPCost cost;
	MLPPLinAlgOld alg;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();

	while (true) {
		cost_prev = Cost(y_hat, alg.onevec(n));

		// Training of the discriminator.

		std::vector<std::vector<real_t>> generatorInputSet = alg.gaussianNoise(n, k);
		std::vector<std::vector<real_t>> discriminatorInputSet = modelSetTestGenerator(generatorInputSet);
		discriminatorInputSet.insert(discriminatorInputSet.end(), outputSet.begin(), outputSet.end()); // Fake + real inputs.

		std::vector<real_t> y_hat = modelSetTestDiscriminator(discriminatorInputSet);
		std::vector<real_t> outputSet = alg.zerovec(n);
		std::vector<real_t> outputSetReal = alg.onevec(n);
		outputSet.insert(outputSet.end(), outputSetReal.begin(), outputSetReal.end()); // Fake + real output scores.

		auto dgrads = computeDiscriminatorGradients(y_hat, outputSet);
		auto cumulativeDiscriminatorHiddenLayerWGrad = std::get<0>(dgrads);
		auto outputDiscriminatorWGrad = std::get<1>(dgrads);

		cumulativeDiscriminatorHiddenLayerWGrad = alg.scalarMultiply(learning_rate / n, cumulativeDiscriminatorHiddenLayerWGrad);
		outputDiscriminatorWGrad = alg.scalarMultiply(learning_rate / n, outputDiscriminatorWGrad);
		updateDiscriminatorParameters(cumulativeDiscriminatorHiddenLayerWGrad, outputDiscriminatorWGrad, learning_rate);

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
			MLPPGANOld::UI(epoch, cost_prev, MLPPGANOld::y_hat, alg.onevec(n));
		}

		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
}

real_t MLPPGANOld::score() {
	MLPPLinAlgOld alg;
	MLPPUtilities util;
	forwardPass();
	return util.performance(y_hat, alg.onevec(n));
}

void MLPPGANOld::save(std::string fileName) {
	MLPPUtilities util;
	if (!network.empty()) {
		util.saveParameters(fileName, network[0].weights, network[0].bias, false, 1);
		for (uint32_t i = 1; i < network.size(); i++) {
			util.saveParameters(fileName, network[i].weights, network[i].bias, true, i + 1);
		}
		util.saveParameters(fileName, outputLayer->weights, outputLayer->bias, true, network.size() + 1);
	} else {
		util.saveParameters(fileName, outputLayer->weights, outputLayer->bias, false, network.size() + 1);
	}
}

void MLPPGANOld::addLayer(int n_hidden, std::string activation, std::string weightInit, std::string reg, real_t lambda, real_t alpha) {
	MLPPLinAlgOld alg;
	if (network.empty()) {
		network.push_back(MLPPOldHiddenLayer(n_hidden, activation, alg.gaussianNoise(n, k), weightInit, reg, lambda, alpha));
		network[0].forwardPass();
	} else {
		network.push_back(MLPPOldHiddenLayer(n_hidden, activation, network[network.size() - 1].a, weightInit, reg, lambda, alpha));
		network[network.size() - 1].forwardPass();
	}
}

void MLPPGANOld::addOutputLayer(std::string weightInit, std::string reg, real_t lambda, real_t alpha) {
	MLPPLinAlgOld alg;
	if (!network.empty()) {
		outputLayer = new MLPPOldOutputLayer(network[network.size() - 1].n_hidden, "Sigmoid", "LogLoss", network[network.size() - 1].a, weightInit, reg, lambda, alpha);
	} else {
		outputLayer = new MLPPOldOutputLayer(k, "Sigmoid", "LogLoss", alg.gaussianNoise(n, k), weightInit, reg, lambda, alpha);
	}
}

std::vector<std::vector<real_t>> MLPPGANOld::modelSetTestGenerator(std::vector<std::vector<real_t>> X) {
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

std::vector<real_t> MLPPGANOld::modelSetTestDiscriminator(std::vector<std::vector<real_t>> X) {
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

real_t MLPPGANOld::Cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPRegOld regularization;
	class MLPPCostOld cost;
	real_t totalRegTerm = 0;

	auto cost_function = outputLayer->cost_map[outputLayer->cost];
	if (!network.empty()) {
		for (uint32_t i = 0; i < network.size() - 1; i++) {
			totalRegTerm += regularization.regTerm(network[i].weights, network[i].lambda, network[i].alpha, network[i].reg);
		}
	}
	return (cost.*cost_function)(y_hat, y) + totalRegTerm + regularization.regTerm(outputLayer->weights, outputLayer->lambda, outputLayer->alpha, outputLayer->reg);
}

void MLPPGANOld::forwardPass() {
	MLPPLinAlgOld alg;
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

void MLPPGANOld::updateDiscriminatorParameters(std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations, std::vector<real_t> outputLayerUpdation, real_t learning_rate) {
	MLPPLinAlgOld alg;

	outputLayer->weights = alg.subtraction(outputLayer->weights, outputLayerUpdation);
	outputLayer->bias -= learning_rate * alg.sum_elements(outputLayer->delta) / n;

	if (!network.empty()) {
		network[network.size() - 1].weights = alg.subtraction(network[network.size() - 1].weights, hiddenLayerUpdations[0]);
		network[network.size() - 1].bias = alg.subtractMatrixRows(network[network.size() - 1].bias, alg.scalarMultiply(learning_rate / n, network[network.size() - 1].delta));

		for (int i = static_cast<int>(network.size()) - 2; i > static_cast<int>(network.size()) / 2; i--) {
			network[i].weights = alg.subtraction(network[i].weights, hiddenLayerUpdations[(network.size() - 2) - i + 1]);
			network[i].bias = alg.subtractMatrixRows(network[i].bias, alg.scalarMultiply(learning_rate / n, network[i].delta));
		}
	}
}

void MLPPGANOld::updateGeneratorParameters(std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations, real_t learning_rate) {
	MLPPLinAlgOld alg;

	if (!network.empty()) {
		for (int i = network.size() / 2; i >= 0; i--) {
			//std::cout << network[i].weights.size() << "x" << network[i].weights[0].size() << std::endl;
			//std::cout << hiddenLayerUpdations[(network.size() - 2) - i + 1].size() << "x" << hiddenLayerUpdations[(network.size() - 2) - i + 1][0].size() << std::endl;
			network[i].weights = alg.subtraction(network[i].weights, hiddenLayerUpdations[(network.size() - 2) - i + 1]);
			network[i].bias = alg.subtractMatrixRows(network[i].bias, alg.scalarMultiply(learning_rate / n, network[i].delta));
		}
	}
}

std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<real_t>> MLPPGANOld::computeDiscriminatorGradients(std::vector<real_t> y_hat, std::vector<real_t> outputSet) {
	class MLPPCostOld cost;
	MLPPActivationOld avn;
	MLPPLinAlgOld alg;
	MLPPRegOld regularization;

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

		for (int i = static_cast<int>(network.size()) - 2; i > static_cast<int>(network.size()) / 2; i--) {
			hiddenLayerAvn = network[i].activation_map[network[i].activation];
			network[i].delta = alg.hadamard_product(alg.matmult(network[i + 1].delta, alg.transpose(network[i + 1].weights)), (avn.*hiddenLayerAvn)(network[i].z, 1));
			hiddenLayerWGrad = alg.matmult(alg.transpose(network[i].input), network[i].delta);

			cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(network[i].weights, network[i].lambda, network[i].alpha, network[i].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}
	return { cumulativeHiddenLayerWGrad, outputWGrad };
}

std::vector<std::vector<std::vector<real_t>>> MLPPGANOld::computeGeneratorGradients(std::vector<real_t> y_hat, std::vector<real_t> outputSet) {
	class MLPPCostOld cost;
	MLPPActivationOld avn;
	MLPPLinAlgOld alg;
	MLPPRegOld regularization;

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

		for (int i = network.size() - 2; i >= 0; i--) {
			hiddenLayerAvn = network[i].activation_map[network[i].activation];
			network[i].delta = alg.hadamard_product(alg.matmult(network[i + 1].delta, alg.transpose(network[i + 1].weights)), (avn.*hiddenLayerAvn)(network[i].z, 1));
			hiddenLayerWGrad = alg.matmult(alg.transpose(network[i].input), network[i].delta);
			cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(network[i].weights, network[i].lambda, network[i].alpha, network[i].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}
	return cumulativeHiddenLayerWGrad;
}

void MLPPGANOld::UI(int epoch, real_t cost_prev, std::vector<real_t> y_hat, std::vector<real_t> outputSet) {
	MLPPUtilities::CostInfo(epoch, cost_prev, Cost(y_hat, outputSet));
	std::cout << "Layer " << network.size() + 1 << ": " << std::endl;
	MLPPUtilities::UI(outputLayer->weights, outputLayer->bias);
	if (!network.empty()) {
		for (int i = network.size() - 1; i >= 0; i--) {
			std::cout << "Layer " << i + 1 << ": " << std::endl;
			MLPPUtilities::UI(network[i].weights, network[i].bias);
		}
	}
}
