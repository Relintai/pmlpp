
#ifndef MLPP_GAN_OLD_hpp
#define MLPP_GAN_OLD_hpp

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

class MLPPGANOld {
public:
	MLPPGANOld(real_t k, std::vector<std::vector<real_t>> outputSet);
	~MLPPGANOld();
	std::vector<std::vector<real_t>> generateExample(int n);
	void gradientDescent(real_t learning_rate, int max_epoch, bool UI = false);
	real_t score();
	void save(std::string fileName);

	void addLayer(int n_hidden, std::string activation, std::string weightInit = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);
	void addOutputLayer(std::string weightInit = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);

private:
	std::vector<std::vector<real_t>> modelSetTestGenerator(std::vector<std::vector<real_t>> X); // Evaluator for the generator of the gan.
	std::vector<real_t> modelSetTestDiscriminator(std::vector<std::vector<real_t>> X); // Evaluator for the discriminator of the gan.

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

#endif /* GAN_hpp */