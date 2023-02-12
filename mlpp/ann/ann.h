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

#include "../hidden_layer/hidden_layer_old.h"
#include "../output_layer/output_layer_old.h"

#include <string>
#include <tuple>
#include <vector>

class MLPPANN {
public:
	std::vector<real_t> model_set_test(std::vector<std::vector<real_t>> X);
	real_t model_test(std::vector<real_t> x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);
	void momentum(real_t learning_rate, int max_epoch, int mini_batch_size, real_t gamma, bool nag, bool ui = false);
	void adagrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t e, bool ui = false);
	void adadelta(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t e, bool ui = false);
	void adam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui = false);
	void adamax(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui = false);
	void nadam(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui = false);
	void amsgrad(real_t learning_rate, int max_epoch, int mini_batch_size, real_t b1, real_t b2, real_t e, bool ui = false);

	real_t score();
	void save(std::string file_name);

	void set_learning_rate_scheduler(std::string type, real_t decay_constant);
	void set_learning_rate_scheduler_drop(std::string type, real_t decay_constant, real_t drop_rate);

	void add_layer(int n_hidden, std::string activation, std::string weight_init = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);
	void add_output_layer(std::string activation, std::string loss, std::string weight_init = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);

	MLPPANN(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet);

	MLPPANN();
	~MLPPANN();

private:
	real_t apply_learning_rate_scheduler(real_t learningRate, real_t decayConstant, real_t epoch, real_t dropRate);

	real_t cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	void forward_pass();
	void update_parameters(std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations, std::vector<real_t> outputLayerUpdation, real_t learning_rate);
	std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<real_t>> compute_gradients(std::vector<real_t> y_hat, std::vector<real_t> outputSet);

	void print_ui(int epoch, real_t cost_prev, std::vector<real_t> y_hat, std::vector<real_t> outputSet);

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