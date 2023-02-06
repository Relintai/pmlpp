
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

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include <string>
#include <tuple>
#include <vector>

class MLPPWGAN : public Reference {
	GDCLASS(MLPPWGAN, Reference);

public:
	Ref<MLPPMatrix> get_output_set();
	void set_output_set(const Ref<MLPPMatrix> &val);

	int get_k() const;
	void set_k(const int val);

	Ref<MLPPMatrix> generate_example(int n);
	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	real_t score();
	void save(const String &file_name);

	void add_layer(int n_hidden, MLPPActivation::ActivationFunction activation, MLPPUtilities::WeightDistributionType weight_init = MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT, MLPPReg::RegularizationType reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t lambda = 0.5, real_t alpha = 0.5);
	void add_output_layer(MLPPUtilities::WeightDistributionType weight_init = MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT, MLPPReg::RegularizationType reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t lambda = 0.5, real_t alpha = 0.5);

	MLPPWGAN(real_t k, const Ref<MLPPMatrix> &output_set);

	MLPPWGAN();
	~MLPPWGAN();

protected:
	Ref<MLPPMatrix> model_set_test_generator(const Ref<MLPPMatrix> &X); // Evaluator for the generator of the WGAN.
	Ref<MLPPVector> model_set_test_discriminator(const Ref<MLPPMatrix> &X); // Evaluator for the discriminator of the WGAN.

	real_t cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);

	void forward_pass();
	void update_discriminator_parameters(Vector<Ref<MLPPMatrix>> hidden_layer_updations, const Ref<MLPPVector> &output_layer_updation, real_t learning_rate);
	void update_generator_parameters(Vector<Ref<MLPPMatrix>> hidden_layer_updations, real_t learning_rate);

	struct DiscriminatorGradientResult {
		Vector<Ref<MLPPMatrix>> cumulative_hidden_layer_w_grad; // Tensor containing ALL hidden grads.
		Ref<MLPPVector> output_w_grad;
	};

	DiscriminatorGradientResult compute_discriminator_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set);
	Vector<Ref<MLPPMatrix>> compute_generator_gradients(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set);

	void handle_ui(int epoch, real_t cost_prev, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set);

	static void _bind_methods();

	Ref<MLPPMatrix> output_set;
	Ref<MLPPVector> y_hat;

	Vector<Ref<MLPPHiddenLayer>> network;
	Ref<MLPPOutputLayer> output_layer;

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