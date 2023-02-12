
#ifndef MLPP_GAN_H
#define MLPP_GAN_H

//
//  GAN.hpp
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

class MLPPGAN {
public:
	/*
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output_set();
	void set_output_set(const Ref<MLPPVector> &val);

	int get_k();
	void set_k(const int val);
	*/

	std::vector<std::vector<real_t>> generate_example(int n);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);

	real_t score();

	void save(std::string file_name);

	void add_layer(int n_hidden, std::string activation, std::string weight_init = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);
	void add_output_layer(std::string weight_init = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);

	MLPPGAN(real_t k, std::vector<std::vector<real_t>> output_set);

	MLPPGAN();
	~MLPPGAN();

protected:
	std::vector<std::vector<real_t>> model_set_test_generator(std::vector<std::vector<real_t>> X); // Evaluator for the generator of the gan.
	std::vector<real_t> model_set_test_discriminator(std::vector<std::vector<real_t>> X); // Evaluator for the discriminator of the gan.

	real_t cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	void forward_pass();

	void update_discriminator_parameters(std::vector<std::vector<std::vector<real_t>>> hidden_layer_updations, std::vector<real_t> output_layer_updation, real_t learning_rate);
	void update_generator_parameters(std::vector<std::vector<std::vector<real_t>>> hidden_layer_updations, real_t learning_rate);

	std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<real_t>> compute_discriminator_gradients(std::vector<real_t> y_hat, std::vector<real_t> output_set);
	std::vector<std::vector<std::vector<real_t>>> compute_generator_gradients(std::vector<real_t> y_hat, std::vector<real_t> output_set);

	void print_ui(int epoch, real_t cost_prev, std::vector<real_t> y_hat, std::vector<real_t> output_set);

	static void _bind_methods();

	std::vector<std::vector<real_t>> _output_set;
	std::vector<real_t> _y_hat;

	std::vector<MLPPOldHiddenLayer> _network;
	MLPPOldOutputLayer *_output_layer;

	int _n;
	int _k;
};

#endif /* GAN_hpp */