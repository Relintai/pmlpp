#ifndef MLPP_ANN_H
#define MLPP_ANN_H

//
//  ANN.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/math/math_defs.h"

#include "../hidden_layer/hidden_layer.h"
#include "../output_layer/output_layer.h"

#include <string>
#include <tuple>
#include <vector>

class MLPPANN {
public:
	MLPPANN(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet);
	~MLPPANN();
	std::vector<real_t> modelSetTest(std::vector<std::vector<real_t>> X);
	real_t modelTest(std::vector<real_t> x);
	void gradientDescent(real_t learning_rate, int max_epoch, bool UI = false);
	void SGD(real_t learning_rate, int max_epoch, bool UI = false);
	void MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI = false);
	void Momentum(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool NAG, bool UI = false);
	void Adagrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t e, bool UI = false);
	void Adadelta(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t e, bool UI = false);
	void Adam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI = false);
	void Adamax(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI = false);
	void Nadam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI = false);
	void AMSGrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool UI = false);
	real_t score();
	void save(std::string fileName);

	void setLearningRateScheduler(std::string type, real_t decayConstant);
	void setLearningRateScheduler(std::string type, real_t decayConstant, real_t dropRate);

	void addLayer(int n_hidden, std::string activation, std::string weightInit = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);
	void addOutputLayer(std::string activation, std::string loss, std::string weightInit = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);

private:
	real_t applyLearningRateScheduler(real_t learningRate, real_t decayConstant, real_t epoch, real_t dropRate);

	real_t Cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	void forwardPass();
	void updateParameters(std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations, std::vector<real_t> outputLayerUpdation, real_t learning_rate);
	std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<real_t>> computeGradients(std::vector<real_t> y_hat, std::vector<real_t> outputSet);

	void UI(int epoch, real_t cost_prev, std::vector<real_t> y_hat, std::vector<real_t> outputSet);

	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;
	std::vector<real_t> y_hat;

	std::vector<MLPPOldHiddenLayer> network;
	MLPPOldOutputLayer *outputLayer;

	int n;
	int k;

	std::string lrScheduler;
	real_t decayConstant;
	real_t dropRate;
};

#endif /* ANN_hpp */