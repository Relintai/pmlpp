
#ifndef MLPP_WGAN_H
#define MLPP_WGAN_H

//
//  WGAN.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/containers/vector.h"
#include "core/math/math_defs.h"
#include "core/string/ustring.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../hidden_layer/hidden_layer.h"
#include "../output_layer/output_layer.h"

#include <string>
#include <tuple>
#include <vector>

class MLPPWGAN : public Reference {
	GDCLASS(MLPPWGAN, Reference);

public:
	std::vector<std::vector<real_t>> generate_example(int n);
	void gradient_descent(real_t learning_rate, int max_epoch, bool UI = false);
	real_t score();
	void save(std::string fileName);

	void add_layer(int n_hidden, std::string activation, std::string weightInit = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);
	void add_output_layer(std::string weightInit = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);

	MLPPWGAN(real_t k, std::vector<std::vector<real_t>> outputSet);

	MLPPWGAN();
	~MLPPWGAN();

protected:
	std::vector<std::vector<real_t>> model_set_test_generator(std::vector<std::vector<real_t>> X); // Evaluator for the generator of the WGAN.
	std::vector<real_t> model_set_test_discriminator(std::vector<std::vector<real_t>> X); // Evaluator for the discriminator of the WGAN.

	real_t cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	void forward_pass();
	void update_discriminator_parameters(std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations, std::vector<real_t> outputLayerUpdation, real_t learning_rate);
	void update_generator_parameters(std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations, real_t learning_rate);
	std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<real_t>> compute_discriminator_gradients(std::vector<real_t> y_hat, std::vector<real_t> outputSet);
	std::vector<std::vector<std::vector<real_t>>> compute_generator_gradients(std::vector<real_t> y_hat, std::vector<real_t> outputSet);

	void handle_ui(int epoch, real_t cost_prev, std::vector<real_t> y_hat, std::vector<real_t> outputSet);

	static void _bind_methods();

	std::vector<std::vector<real_t>> outputSet;
	std::vector<real_t> y_hat;

	std::vector<MLPPOldHiddenLayer> network;
	MLPPOldOutputLayer *outputLayer;

	int n;
	int k;
};

class MLPPWGANOld {
public:
	MLPPWGANOld(real_t k, std::vector<std::vector<real_t>> outputSet);
	~MLPPWGANOld();
	std::vector<std::vector<real_t>> generateExample(int n);
	void gradientDescent(real_t learning_rate, int max_epoch, bool UI = false);
	real_t score();
	void save(std::string fileName);

	void addLayer(int n_hidden, std::string activation, std::string weightInit = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);
	void addOutputLayer(std::string weightInit = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);

private:
	std::vector<std::vector<real_t>> modelSetTestGenerator(std::vector<std::vector<real_t>> X); // Evaluator for the generator of the WGAN.
	std::vector<real_t> modelSetTestDiscriminator(std::vector<std::vector<real_t>> X); // Evaluator for the discriminator of the WGAN.

	real_t Cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	void forwardPass();
	void updateDiscriminatorParameters(std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations, std::vector<real_t> outputLayerUpdation, real_t learning_rate);
	void updateGeneratorParameters(std::vector<std::vector<std::vector<real_t>>> hiddenLayerUpdations, real_t learning_rate);
	std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<real_t>> computeDiscriminatorGradients(std::vector<real_t> y_hat, std::vector<real_t> outputSet);
	std::vector<std::vector<std::vector<real_t>>> computeGeneratorGradients(std::vector<real_t> y_hat, std::vector<real_t> outputSet);

	void UI(int epoch, real_t cost_prev, std::vector<real_t> y_hat, std::vector<real_t> outputSet);

	std::vector<std::vector<real_t>> outputSet;
	std::vector<real_t> y_hat;

	std::vector<MLPPOldHiddenLayer> network;
	MLPPOldOutputLayer *outputLayer;

	int n;
	int k;
};

#endif /* WGAN_hpp */