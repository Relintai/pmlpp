
#ifndef MLPP_HIDDEN_LAYER_H
#define MLPP_HIDDEN_LAYER_H

//
//  HiddenLayer.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/containers/hash_map.h"
#include "core/math/math_defs.h"
#include "core/string/ustring.h"

#include "core/object/reference.h"

#include "../activation/activation.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <map>
#include <string>
#include <vector>

class MLPPHiddenLayer : public Reference {
	GDCLASS(MLPPHiddenLayer, Reference);

public:
	int get_n_hidden();
	void set_n_hidden(const int val);

	MLPPActivation::ActivationFunction get_activation();
	void set_activation(const MLPPActivation::ActivationFunction val);

	Ref<MLPPMatrix> get_input();
	void set_input(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> get_weights();
	void set_weights(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_bias();
	void set_bias(const Ref<MLPPVector> &val);

	Ref<MLPPMatrix> get_z();
	void set_z(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> get_a();
	void set_a(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_z_test();
	void set_z_test(const Ref<MLPPVector> &val);

	Ref<MLPPVector> get_a_test();
	void set_a_test(const Ref<MLPPVector> &val);

	Ref<MLPPMatrix> get_delta();
	void set_delta(const Ref<MLPPMatrix> &val);

	MLPPReg::RegularizationType get_reg();
	void set_reg(const MLPPReg::RegularizationType val);

	real_t get_lambda();
	void set_lambda(const real_t val);

	real_t get_alpha();
	void set_alpha(const real_t val);

	MLPPUtilities::WeightDistributionType get_weight_init();
	void set_weight_init(const MLPPUtilities::WeightDistributionType val);

	void forward_pass();
	void test(const Ref<MLPPVector> &x);

	MLPPHiddenLayer(int p_n_hidden, MLPPActivation::ActivationFunction p_activation, Ref<MLPPMatrix> p_input, MLPPUtilities::WeightDistributionType p_weight_init, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha);

	MLPPHiddenLayer();
	~MLPPHiddenLayer();

protected:
	static void _bind_methods();

	int n_hidden;
	MLPPActivation::ActivationFunction activation;

	Ref<MLPPMatrix> input;

	Ref<MLPPMatrix> weights;
	Ref<MLPPVector> bias;

	Ref<MLPPMatrix> z;
	Ref<MLPPMatrix> a;

	Ref<MLPPVector> z_test;
	Ref<MLPPVector> a_test;

	Ref<MLPPMatrix> delta;

	// Regularization Params
	MLPPReg::RegularizationType reg;
	real_t lambda; /* Regularization Parameter */
	real_t alpha; /* This is the controlling param for Elastic Net*/

	MLPPUtilities::WeightDistributionType weight_init;
};

class MLPPOldHiddenLayer {
public:
	MLPPOldHiddenLayer(int n_hidden, std::string activation, std::vector<std::vector<real_t>> input, std::string weightInit, std::string reg, real_t lambda, real_t alpha);

	int n_hidden;
	std::string activation;

	std::vector<std::vector<real_t>> input;

	std::vector<std::vector<real_t>> weights;
	std::vector<real_t> bias;

	std::vector<std::vector<real_t>> z;
	std::vector<std::vector<real_t>> a;

	std::map<std::string, std::vector<std::vector<real_t>> (MLPPActivation::*)(std::vector<std::vector<real_t>>, bool)> activation_map;
	std::map<std::string, std::vector<real_t> (MLPPActivation::*)(std::vector<real_t>, bool)> activationTest_map;

	std::vector<real_t> z_test;
	std::vector<real_t> a_test;

	std::vector<std::vector<real_t>> delta;

	// Regularization Params
	std::string reg;
	real_t lambda; /* Regularization Parameter */
	real_t alpha; /* This is the controlling param for Elastic Net*/

	std::string weightInit;

	void forwardPass();
	void Test(std::vector<real_t> x);
};

#endif /* HiddenLayer_hpp */