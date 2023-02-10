//
//  ANN.cpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "ann_old.h"
#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <cmath>
#include <iostream>
#include <random>

MLPPANNOld::MLPPANNOld(std::vector<std::vector<real_t>> p_inputSet, std::vector<real_t> p_outputSet) {
	inputSet = p_inputSet;
	outputSet = p_outputSet;

	n = inputSet.size();
	k = inputSet[0].size();
	lrScheduler = "None";
	decayConstant = 0;
	dropRate = 0;
}

MLPPANNOld::~MLPPANNOld() {
	delete outputLayer;
}

std::vector<real_t> MLPPANNOld::modelSetTest(std::vector<std::vector<real_t>> X) {
	if (!network.empty()) {
		network[0].input = X;
		network[0].forwardPass();

		for (uint32_t i = 1; i < network.size(); i++) {
			network[i].input = network[i - 1].a;
			network[i].forwardPass();
		}
		outputLayer->input = network[network.size() - 1].a;
	} else {
		outputLayer->input = X;
	}
	outputLayer->forwardPass();
	return outputLayer->a;
}

real_t MLPPANNOld::modelTest(std::vector<real_t> x) {
	if (!network.empty()) {
		network[0].Test(x);
		for (uint32_t i = 1; i < network.size(); i++) {
			network[i].Test(network[i - 1].a_test);
		}
		outputLayer->Test(network[network.size() - 1].a_test);
	} else {
		outputLayer->Test(x);
	}
	return outputLayer->a_test;
}

void MLPPANNOld::gradientDescent(real_t learning_rate, int max_epoch, bool UI) {
	class MLPPCost cost;
	MLPPLinAlg alg;
	real_t cost_prev = 0;
	int epoch = 1;
	forwardPass();
	real_t initial_learning_rate = learning_rate;

	alg.printMatrix(network[network.size() - 1].weights);
	while (true) {
		learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
		cost_prev = Cost(y_hat, outputSet);

		auto grads = computeGradients(y_hat, outputSet);
		auto cumulativeHiddenLayerWGrad = std::get<0>(grads);
		auto outputWGrad = std::get<1>(grads);

		cumulativeHiddenLayerWGrad = alg.scalarMultiply(learning_rate / n, cumulativeHiddenLayerWGrad);
		outputWGrad = alg.scalarMultiply(learning_rate / n, outputWGrad);
		updateParameters(cumulativeHiddenLayerWGrad, outputWGrad, learning_rate); // subject to change. may want bias to have this matrix too.

		std::cout << learning_rate << std::endl;

		forwardPass();

		if (UI) {
			MLPPANNOld::UI(epoch, cost_prev, y_hat, outputSet);
		}

		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
}

void MLPPANNOld::SGD(real_t learning_rate, int max_epoch, bool UI) {
	class MLPPCost cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	while (true) {
		learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);

		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(n - 1));
		int outputIndex = distribution(generator);

		std::vector<real_t> y_hat = modelSetTest({ inputSet[outputIndex] });
		cost_prev = Cost({ y_hat }, { outputSet[outputIndex] });

		auto grads = computeGradients(y_hat, { outputSet[outputIndex] });
		auto cumulativeHiddenLayerWGrad = std::get<0>(grads);
		auto outputWGrad = std::get<1>(grads);

		cumulativeHiddenLayerWGrad = alg.scalarMultiply(learning_rate / n, cumulativeHiddenLayerWGrad);
		outputWGrad = alg.scalarMultiply(learning_rate / n, outputWGrad);

		updateParameters(cumulativeHiddenLayerWGrad, outputWGrad, learning_rate); // subject to change. may want bias to have this matrix too.
		y_hat = modelSetTest({ inputSet[outputIndex] });

		if (UI) {
			MLPPANNOld::UI(epoch, cost_prev, y_hat, { outputSet[outputIndex] });
		}

		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPANNOld::MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI) {
	class MLPPCost cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	auto batches = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
	auto inputMiniBatches = std::get<0>(batches);
	auto outputMiniBatches = std::get<1>(batches);

	while (true) {
		learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = modelSetTest(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			auto grads = computeGradients(y_hat, outputMiniBatches[i]);
			auto cumulativeHiddenLayerWGrad = std::get<0>(grads);
			auto outputWGrad = std::get<1>(grads);

			cumulativeHiddenLayerWGrad = alg.scalarMultiply(learning_rate / n, cumulativeHiddenLayerWGrad);
			outputWGrad = alg.scalarMultiply(learning_rate / n, outputWGrad);

			updateParameters(cumulativeHiddenLayerWGrad, outputWGrad, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = modelSetTest(inputMiniBatches[i]);

			if (UI) {
				MLPPANNOld::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPANNOld::Momentum(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool NAG, bool UI) {
	class MLPPCost cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	auto batches = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
	auto inputMiniBatches = std::get<0>(batches);
	auto outputMiniBatches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> v_hidden;

	std::vector<real_t> v_output;
	while (true) {
		learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = modelSetTest(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			auto grads = computeGradients(y_hat, outputMiniBatches[i]);
			auto cumulativeHiddenLayerWGrad = std::get<0>(grads);
			auto outputWGrad = std::get<1>(grads);

			if (!network.empty() && v_hidden.empty()) { // Initing our tensor
				v_hidden = alg.resize(v_hidden, cumulativeHiddenLayerWGrad);
			}

			if (v_output.empty()) {
				v_output.resize(outputWGrad.size());
			}

			if (NAG) { // "Aposterori" calculation
				updateParameters(v_hidden, v_output, 0); // DON'T update bias.
			}

			v_hidden = alg.addition(alg.scalarMultiply(gamma, v_hidden), alg.scalarMultiply(learning_rate / n, cumulativeHiddenLayerWGrad));

			v_output = alg.addition(alg.scalarMultiply(gamma, v_output), alg.scalarMultiply(learning_rate / n, outputWGrad));

			updateParameters(v_hidden, v_output, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = modelSetTest(inputMiniBatches[i]);

			if (UI) {
				MLPPANNOld::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPANNOld::Adagrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t e, bool UI) {
	class MLPPCost cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	auto batches = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
	auto inputMiniBatches = std::get<0>(batches);
	auto outputMiniBatches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> v_hidden;

	std::vector<real_t> v_output;
	while (true) {
		learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = modelSetTest(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			auto grads = computeGradients(y_hat, outputMiniBatches[i]);
			auto cumulativeHiddenLayerWGrad = std::get<0>(grads);
			auto outputWGrad = std::get<1>(grads);

			if (!network.empty() && v_hidden.empty()) { // Initing our tensor
				v_hidden = alg.resize(v_hidden, cumulativeHiddenLayerWGrad);
			}

			if (v_output.empty()) {
				v_output.resize(outputWGrad.size());
			}

			v_hidden = alg.addition(v_hidden, alg.exponentiate(cumulativeHiddenLayerWGrad, 2));

			v_output = alg.addition(v_output, alg.exponentiate(outputWGrad, 2));

			std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations = alg.scalarMultiply(learning_rate / n, alg.elementWiseDivision(cumulativeHiddenLayerWGrad, alg.scalarAdd(e, alg.sqrt(v_hidden))));
			std::vector<real_t> outputLayerUpdation = alg.scalarMultiply(learning_rate / n, alg.elementWiseDivision(outputWGrad, alg.scalarAdd(e, alg.sqrt(v_output))));

			updateParameters(hiddenLayerUpdations, outputLayerUpdation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = modelSetTest(inputMiniBatches[i]);

			if (UI) {
				MLPPANNOld::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPANNOld::Adadelta(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t e, bool UI) {
	class MLPPCost cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	auto batches = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
	auto inputMiniBatches = std::get<0>(batches);
	auto outputMiniBatches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> v_hidden;

	std::vector<real_t> v_output;
	while (true) {
		learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = modelSetTest(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			auto grads = computeGradients(y_hat, outputMiniBatches[i]);
			auto cumulativeHiddenLayerWGrad = std::get<0>(grads);
			auto outputWGrad = std::get<1>(grads);

			if (!network.empty() && v_hidden.empty()) { // Initing our tensor
				v_hidden = alg.resize(v_hidden, cumulativeHiddenLayerWGrad);
			}

			if (v_output.empty()) {
				v_output.resize(outputWGrad.size());
			}

			v_hidden = alg.addition(alg.scalarMultiply(1 - b1, v_hidden), alg.scalarMultiply(b1, alg.exponentiate(cumulativeHiddenLayerWGrad, 2)));

			v_output = alg.addition(v_output, alg.exponentiate(outputWGrad, 2));

			std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations = alg.scalarMultiply(learning_rate / n, alg.elementWiseDivision(cumulativeHiddenLayerWGrad, alg.scalarAdd(e, alg.sqrt(v_hidden))));
			std::vector<real_t> outputLayerUpdation = alg.scalarMultiply(learning_rate / n, alg.elementWiseDivision(outputWGrad, alg.scalarAdd(e, alg.sqrt(v_output))));

			updateParameters(hiddenLayerUpdations, outputLayerUpdation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = modelSetTest(inputMiniBatches[i]);

			if (UI) {
				MLPPANNOld::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPANNOld::Adam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI) {
	class MLPPCost cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	auto batches = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
	auto inputMiniBatches = std::get<0>(batches);
	auto outputMiniBatches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> m_hidden;
	std::vector<std::vector<std::vector<real_t>>> v_hidden;

	std::vector<real_t> m_output;
	std::vector<real_t> v_output;
	while (true) {
		learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = modelSetTest(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			auto grads = computeGradients(y_hat, outputMiniBatches[i]);
			auto cumulativeHiddenLayerWGrad = std::get<0>(grads);
			auto outputWGrad = std::get<1>(grads);

			if (!network.empty() && m_hidden.empty() && v_hidden.empty()) { // Initing our tensor
				m_hidden = alg.resize(m_hidden, cumulativeHiddenLayerWGrad);
				v_hidden = alg.resize(v_hidden, cumulativeHiddenLayerWGrad);
			}

			if (m_output.empty() && v_output.empty()) {
				m_output.resize(outputWGrad.size());
				v_output.resize(outputWGrad.size());
			}

			m_hidden = alg.addition(alg.scalarMultiply(b1, m_hidden), alg.scalarMultiply(1 - b1, cumulativeHiddenLayerWGrad));
			v_hidden = alg.addition(alg.scalarMultiply(b2, v_hidden), alg.scalarMultiply(1 - b2, alg.exponentiate(cumulativeHiddenLayerWGrad, 2)));

			m_output = alg.addition(alg.scalarMultiply(b1, m_output), alg.scalarMultiply(1 - b1, outputWGrad));
			v_output = alg.addition(alg.scalarMultiply(b2, v_output), alg.scalarMultiply(1 - b2, alg.exponentiate(outputWGrad, 2)));

			std::vector<std::vector<std::vector<real_t>>> m_hidden_hat = alg.scalarMultiply(1 / (1 - std::pow(b1, epoch)), m_hidden);
			std::vector<std::vector<std::vector<real_t>>> v_hidden_hat = alg.scalarMultiply(1 / (1 - std::pow(b2, epoch)), v_hidden);

			std::vector<real_t> m_output_hat = alg.scalarMultiply(1 / (1 - std::pow(b1, epoch)), m_output);
			std::vector<real_t> v_output_hat = alg.scalarMultiply(1 / (1 - std::pow(b2, epoch)), v_output);

			std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations = alg.scalarMultiply(learning_rate / n, alg.elementWiseDivision(m_hidden_hat, alg.scalarAdd(e, alg.sqrt(v_hidden_hat))));
			std::vector<real_t> outputLayerUpdation = alg.scalarMultiply(learning_rate / n, alg.elementWiseDivision(m_output_hat, alg.scalarAdd(e, alg.sqrt(v_output_hat))));

			updateParameters(hiddenLayerUpdations, outputLayerUpdation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = modelSetTest(inputMiniBatches[i]);

			if (UI) {
				MLPPANNOld::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPANNOld::Adamax(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI) {
	class MLPPCost cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	auto batches = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
	auto inputMiniBatches = std::get<0>(batches);
	auto outputMiniBatches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> m_hidden;
	std::vector<std::vector<std::vector<real_t>>> u_hidden;

	std::vector<real_t> m_output;
	std::vector<real_t> u_output;
	while (true) {
		learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = modelSetTest(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			auto grads = computeGradients(y_hat, outputMiniBatches[i]);
			auto cumulativeHiddenLayerWGrad = std::get<0>(grads);
			auto outputWGrad = std::get<1>(grads);

			if (!network.empty() && m_hidden.empty() && u_hidden.empty()) { // Initing our tensor
				m_hidden = alg.resize(m_hidden, cumulativeHiddenLayerWGrad);
				u_hidden = alg.resize(u_hidden, cumulativeHiddenLayerWGrad);
			}

			if (m_output.empty() && u_output.empty()) {
				m_output.resize(outputWGrad.size());
				u_output.resize(outputWGrad.size());
			}

			m_hidden = alg.addition(alg.scalarMultiply(b1, m_hidden), alg.scalarMultiply(1 - b1, cumulativeHiddenLayerWGrad));
			u_hidden = alg.max(alg.scalarMultiply(b2, u_hidden), alg.abs(cumulativeHiddenLayerWGrad));

			m_output = alg.addition(alg.scalarMultiply(b1, m_output), alg.scalarMultiply(1 - b1, outputWGrad));
			u_output = alg.max(alg.scalarMultiply(b2, u_output), alg.abs(outputWGrad));

			std::vector<std::vector<std::vector<real_t>>> m_hidden_hat = alg.scalarMultiply(1 / (1 - std::pow(b1, epoch)), m_hidden);

			std::vector<real_t> m_output_hat = alg.scalarMultiply(1 / (1 - std::pow(b1, epoch)), m_output);

			std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations = alg.scalarMultiply(learning_rate / n, alg.elementWiseDivision(m_hidden_hat, alg.scalarAdd(e, u_hidden)));
			std::vector<real_t> outputLayerUpdation = alg.scalarMultiply(learning_rate / n, alg.elementWiseDivision(m_output_hat, alg.scalarAdd(e, u_output)));

			updateParameters(hiddenLayerUpdations, outputLayerUpdation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = modelSetTest(inputMiniBatches[i]);

			if (UI) {
				MLPPANNOld::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPANNOld::Nadam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI) {
	class MLPPCost cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	auto batches = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
	auto inputMiniBatches = std::get<0>(batches);
	auto outputMiniBatches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> m_hidden;
	std::vector<std::vector<std::vector<real_t>>> v_hidden;

	std::vector<real_t> m_output;
	std::vector<real_t> v_output;
	while (true) {
		learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = modelSetTest(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			auto grads = computeGradients(y_hat, outputMiniBatches[i]);
			auto cumulativeHiddenLayerWGrad = std::get<0>(grads);
			auto outputWGrad = std::get<1>(grads);

			if (!network.empty() && m_hidden.empty() && v_hidden.empty()) { // Initing our tensor
				m_hidden = alg.resize(m_hidden, cumulativeHiddenLayerWGrad);
				v_hidden = alg.resize(v_hidden, cumulativeHiddenLayerWGrad);
			}

			if (m_output.empty() && v_output.empty()) {
				m_output.resize(outputWGrad.size());
				v_output.resize(outputWGrad.size());
			}

			m_hidden = alg.addition(alg.scalarMultiply(b1, m_hidden), alg.scalarMultiply(1 - b1, cumulativeHiddenLayerWGrad));
			v_hidden = alg.addition(alg.scalarMultiply(b2, v_hidden), alg.scalarMultiply(1 - b2, alg.exponentiate(cumulativeHiddenLayerWGrad, 2)));

			m_output = alg.addition(alg.scalarMultiply(b1, m_output), alg.scalarMultiply(1 - b1, outputWGrad));
			v_output = alg.addition(alg.scalarMultiply(b2, v_output), alg.scalarMultiply(1 - b2, alg.exponentiate(outputWGrad, 2)));

			std::vector<std::vector<std::vector<real_t>>> m_hidden_hat = alg.scalarMultiply(1 / (1 - std::pow(b1, epoch)), m_hidden);
			std::vector<std::vector<std::vector<real_t>>> v_hidden_hat = alg.scalarMultiply(1 / (1 - std::pow(b2, epoch)), v_hidden);
			std::vector<std::vector<std::vector<real_t>>> m_hidden_final = alg.addition(alg.scalarMultiply(b1, m_hidden_hat), alg.scalarMultiply((1 - b1) / (1 - std::pow(b1, epoch)), cumulativeHiddenLayerWGrad));

			std::vector<real_t> m_output_hat = alg.scalarMultiply(1 / (1 - std::pow(b1, epoch)), m_output);
			std::vector<real_t> v_output_hat = alg.scalarMultiply(1 / (1 - std::pow(b2, epoch)), v_output);
			std::vector<real_t> m_output_final = alg.addition(alg.scalarMultiply(b1, m_output_hat), alg.scalarMultiply((1 - b1) / (1 - std::pow(b1, epoch)), outputWGrad));

			std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations = alg.scalarMultiply(learning_rate / n, alg.elementWiseDivision(m_hidden_final, alg.scalarAdd(e, alg.sqrt(v_hidden_hat))));
			std::vector<real_t> outputLayerUpdation = alg.scalarMultiply(learning_rate / n, alg.elementWiseDivision(m_output_final, alg.scalarAdd(e, alg.sqrt(v_output_hat))));

			updateParameters(hiddenLayerUpdations, outputLayerUpdation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = modelSetTest(inputMiniBatches[i]);

			if (UI) {
				MLPPANNOld::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

void MLPPANNOld::AMSGrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI) {
	class MLPPCost cost;
	MLPPLinAlg alg;

	real_t cost_prev = 0;
	int epoch = 1;
	real_t initial_learning_rate = learning_rate;

	// Creating the mini-batches
	int n_mini_batch = n / mini_batch_size;
	// always evaluate the result
	// always do forward pass only ONCE at end.

	auto batches = MLPPUtilities::createMiniBatches(inputSet, outputSet, n_mini_batch);
	auto inputMiniBatches = std::get<0>(batches);
	auto outputMiniBatches = std::get<1>(batches);

	// Initializing necessary components for Adam.
	std::vector<std::vector<std::vector<real_t>>> m_hidden;
	std::vector<std::vector<std::vector<real_t>>> v_hidden;

	std::vector<std::vector<std::vector<real_t>>> v_hidden_hat;

	std::vector<real_t> m_output;
	std::vector<real_t> v_output;

	std::vector<real_t> v_output_hat;
	while (true) {
		learning_rate = applyLearningRateScheduler(initial_learning_rate, decayConstant, epoch, dropRate);
		for (int i = 0; i < n_mini_batch; i++) {
			std::vector<real_t> y_hat = modelSetTest(inputMiniBatches[i]);
			cost_prev = Cost(y_hat, outputMiniBatches[i]);

			auto grads = computeGradients(y_hat, outputMiniBatches[i]);
			auto cumulativeHiddenLayerWGrad = std::get<0>(grads);
			auto outputWGrad = std::get<1>(grads);

			if (!network.empty() && m_hidden.empty() && v_hidden.empty()) { // Initing our tensor
				m_hidden = alg.resize(m_hidden, cumulativeHiddenLayerWGrad);
				v_hidden = alg.resize(v_hidden, cumulativeHiddenLayerWGrad);
				v_hidden_hat = alg.resize(v_hidden_hat, cumulativeHiddenLayerWGrad);
			}

			if (m_output.empty() && v_output.empty()) {
				m_output.resize(outputWGrad.size());
				v_output.resize(outputWGrad.size());
				v_output_hat.resize(outputWGrad.size());
			}

			m_hidden = alg.addition(alg.scalarMultiply(b1, m_hidden), alg.scalarMultiply(1 - b1, cumulativeHiddenLayerWGrad));
			v_hidden = alg.addition(alg.scalarMultiply(b2, v_hidden), alg.scalarMultiply(1 - b2, alg.exponentiate(cumulativeHiddenLayerWGrad, 2)));

			m_output = alg.addition(alg.scalarMultiply(b1, m_output), alg.scalarMultiply(1 - b1, outputWGrad));
			v_output = alg.addition(alg.scalarMultiply(b2, v_output), alg.scalarMultiply(1 - b2, alg.exponentiate(outputWGrad, 2)));

			v_hidden_hat = alg.max(v_hidden_hat, v_hidden);

			v_output_hat = alg.max(v_output_hat, v_output);

			std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations = alg.scalarMultiply(learning_rate / n, alg.elementWiseDivision(m_hidden, alg.scalarAdd(e, alg.sqrt(v_hidden_hat))));
			std::vector<real_t> outputLayerUpdation = alg.scalarMultiply(learning_rate / n, alg.elementWiseDivision(m_output, alg.scalarAdd(e, alg.sqrt(v_output_hat))));

			updateParameters(hiddenLayerUpdations, outputLayerUpdation, learning_rate); // subject to change. may want bias to have this matrix too.
			y_hat = modelSetTest(inputMiniBatches[i]);

			if (UI) {
				MLPPANNOld::UI(epoch, cost_prev, y_hat, outputMiniBatches[i]);
			}
		}
		epoch++;
		if (epoch > max_epoch) {
			break;
		}
	}
	forwardPass();
}

real_t MLPPANNOld::score() {
	MLPPUtilities util;
	forwardPass();
	return util.performance(y_hat, outputSet);
}

void MLPPANNOld::save(std::string fileName) {
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

void MLPPANNOld::setLearningRateScheduler(std::string type, real_t decayConstant) {
	lrScheduler = type;
	MLPPANNOld::decayConstant = decayConstant;
}

void MLPPANNOld::setLearningRateScheduler(std::string type, real_t decayConstant, real_t dropRate) {
	lrScheduler = type;
	MLPPANNOld::decayConstant = decayConstant;
	MLPPANNOld::dropRate = dropRate;
}

// https://en.wikipedia.org/wiki/Learning_rate
// Learning Rate Decay (C2W2L09) - Andrew Ng - Deep Learning Specialization
real_t MLPPANNOld::applyLearningRateScheduler(real_t learningRate, real_t decayConstant, real_t epoch, real_t dropRate) {
	if (lrScheduler == "Time") {
		return learningRate / (1 + decayConstant * epoch);
	} else if (lrScheduler == "Epoch") {
		return learningRate * (decayConstant / std::sqrt(epoch));
	} else if (lrScheduler == "Step") {
		return learningRate * std::pow(decayConstant, int((1 + epoch) / dropRate)); // Utilizing an explicit int conversion implicitly takes the floor.
	} else if (lrScheduler == "Exponential") {
		return learningRate * std::exp(-decayConstant * epoch);
	}
	return learningRate;
}

void MLPPANNOld::addLayer(int n_hidden, std::string activation, std::string weightInit, std::string reg, real_t lambda, real_t alpha) {
	if (network.empty()) {
		network.push_back(MLPPOldHiddenLayer(n_hidden, activation, inputSet, weightInit, reg, lambda, alpha));
		network[0].forwardPass();
	} else {
		network.push_back(MLPPOldHiddenLayer(n_hidden, activation, network[network.size() - 1].a, weightInit, reg, lambda, alpha));
		network[network.size() - 1].forwardPass();
	}
}

void MLPPANNOld::addOutputLayer(std::string activation, std::string loss, std::string weightInit, std::string reg, real_t lambda, real_t alpha) {
	if (!network.empty()) {
		outputLayer = new MLPPOldOutputLayer(network[network.size() - 1].n_hidden, activation, loss, network[network.size() - 1].a, weightInit, reg, lambda, alpha);
	} else {
		outputLayer = new MLPPOldOutputLayer(k, activation, loss, inputSet, weightInit, reg, lambda, alpha);
	}
}

real_t MLPPANNOld::Cost(std::vector<real_t> y_hat, std::vector<real_t> y) {
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

void MLPPANNOld::forwardPass() {
	if (!network.empty()) {
		network[0].input = inputSet;
		network[0].forwardPass();

		for (uint32_t i = 1; i < network.size(); i++) {
			network[i].input = network[i - 1].a;
			network[i].forwardPass();
		}
		outputLayer->input = network[network.size() - 1].a;
	} else {
		outputLayer->input = inputSet;
	}
	outputLayer->forwardPass();
	y_hat = outputLayer->a;
}

void MLPPANNOld::updateParameters(std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations, std::vector<real_t> outputLayerUpdation, real_t learning_rate) {
	MLPPLinAlg alg;

	outputLayer->weights = alg.subtraction(outputLayer->weights, outputLayerUpdation);
	outputLayer->bias -= learning_rate * alg.sum_elements(outputLayer->delta) / n;

	if (!network.empty()) {
		network[network.size() - 1].weights = alg.subtraction(network[network.size() - 1].weights, hiddenLayerUpdations[0]);
		network[network.size() - 1].bias = alg.subtractMatrixRows(network[network.size() - 1].bias, alg.scalarMultiply(learning_rate / n, network[network.size() - 1].delta));

		for (int i = network.size() - 2; i >= 0; i--) {
			network[i].weights = alg.subtraction(network[i].weights, hiddenLayerUpdations[(network.size() - 2) - i + 1]);
			network[i].bias = alg.subtractMatrixRows(network[i].bias, alg.scalarMultiply(learning_rate / n, network[i].delta));
		}
	}
}

std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<real_t>> MLPPANNOld::computeGradients(std::vector<real_t> y_hat, std::vector<real_t> outputSet) {
	// std::cout << "BEGIN" << std::endl;
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

		for (int i = network.size() - 2; i >= 0; i--) {
			hiddenLayerAvn = network[i].activation_map[network[i].activation];
			network[i].delta = alg.hadamard_product(alg.matmult(network[i + 1].delta, alg.transpose(network[i + 1].weights)), (avn.*hiddenLayerAvn)(network[i].z, 1));
			hiddenLayerWGrad = alg.matmult(alg.transpose(network[i].input), network[i].delta);
			cumulativeHiddenLayerWGrad.push_back(alg.addition(hiddenLayerWGrad, regularization.regDerivTerm(network[i].weights, network[i].lambda, network[i].alpha, network[i].reg))); // Adding to our cumulative hidden layer grads. Maintain reg terms as well.
		}
	}
	return { cumulativeHiddenLayerWGrad, outputWGrad };
}

void MLPPANNOld::UI(int epoch, real_t cost_prev, std::vector<real_t> y_hat, std::vector<real_t> outputSet) {
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
