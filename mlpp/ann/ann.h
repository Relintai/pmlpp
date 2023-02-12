#ifndef MLPP_ANN_H
#define MLPP_ANN_H

//
//  ANN.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../hidden_layer/hidden_layer.h"
#include "../output_layer/output_layer.h"

#include "../hidden_layer/hidden_layer_old.h"
#include "../output_layer/output_layer_old.h"

#include <string>
#include <tuple>
#include <vector>

class MLPPANN : public Reference {
	GDCLASS(MLPPANN, Reference);

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

	MLPPANN(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set);

	MLPPANN();
	~MLPPANN();

protected:
	real_t apply_learning_rate_scheduler(real_t learning_rate, real_t decay_constant, real_t epoch, real_t drop_rate);

	real_t cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	void forward_pass();
	void update_parameters(std::vector<std::vector<std::vector<real_t>>> hidden_layer_updations, std::vector<real_t> output_layer_updation, real_t learning_rate);
	std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<real_t>> compute_gradients(std::vector<real_t> y_hat, std::vector<real_t> _output_set);

	void print_ui(int epoch, real_t cost_prev, std::vector<real_t> y_hat, std::vector<real_t> p_output_set);

	static void _bind_methods();

	std::vector<std::vector<real_t>> _input_set;
	std::vector<real_t> _output_set;
	std::vector<real_t> _y_hat;

	std::vector<MLPPOldHiddenLayer> _network;
	MLPPOldOutputLayer *_output_layer;

	int _n;
	int _k;

	std::string _lr_scheduler;
	real_t _decay_constant;
	real_t _drop_rate;
};

#endif /* ANN_hpp */