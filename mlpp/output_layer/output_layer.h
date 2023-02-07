
#ifndef MLPP_OUTPUT_LAYER_H
#define MLPP_OUTPUT_LAYER_H

//
//  OutputLayer.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/math/math_defs.h"
#include "core/string/ustring.h"

#include "core/object/reference.h"

#include "../activation/activation.h"
#include "../cost/cost.h"
#include "../regularization/reg.h"
#include "../utilities/utilities.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"
class MLPPOutputLayer : public Reference {
	GDCLASS(MLPPOutputLayer, Reference);

public:
	int get_n_hidden();
	void set_n_hidden(const int val);

	MLPPActivation::ActivationFunction get_activation();
	void set_activation(const MLPPActivation::ActivationFunction val);

	MLPPCost::CostTypes get_cost();
	void set_cost(const MLPPCost::CostTypes val);

	Ref<MLPPMatrix> get_input();
	void set_input(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_weights();
	void set_weights(const Ref<MLPPVector> &val);

	real_t get_bias();
	void set_bias(const real_t val);

	Ref<MLPPVector> get_z();
	void set_z(const Ref<MLPPVector> &val);

	Ref<MLPPVector> get_a();
	void set_a(const Ref<MLPPVector> &val);

	Ref<MLPPVector> get_z_test();
	void set_z_test(const Ref<MLPPVector> &val);

	Ref<MLPPVector> get_a_test();
	void set_a_test(const Ref<MLPPVector> &val);

	Ref<MLPPVector> get_delta();
	void set_delta(const Ref<MLPPVector> &val);

	MLPPReg::RegularizationType get_reg();
	void set_reg(const MLPPReg::RegularizationType val);

	real_t get_lambda();
	void set_lambda(const real_t val);

	real_t get_alpha();
	void set_alpha(const real_t val);

	MLPPUtilities::WeightDistributionType get_weight_init();
	void set_weight_init(const MLPPUtilities::WeightDistributionType val);

	bool is_initialized();
	void initialize();

	void forward_pass();
	void test(const Ref<MLPPVector> &x);

	MLPPOutputLayer(int p_n_hidden, MLPPActivation::ActivationFunction p_activation, Ref<MLPPMatrix> p_input, MLPPUtilities::WeightDistributionType p_weight_init, MLPPReg::RegularizationType p_reg, real_t p_lambda, real_t p_alpha);

	MLPPOutputLayer();
	~MLPPOutputLayer();

protected:
	static void _bind_methods();

	int n_hidden;
	MLPPActivation::ActivationFunction activation;
	MLPPCost::CostTypes cost;

	Ref<MLPPMatrix> input;

	Ref<MLPPVector> weights;
	real_t bias;

	Ref<MLPPVector> z;
	Ref<MLPPVector> a;

	Ref<MLPPVector> z_test;
	Ref<MLPPVector> a_test;

	Ref<MLPPVector> delta;

	// Regularization Params
	MLPPReg::RegularizationType reg;
	real_t lambda; /* Regularization Parameter */
	real_t alpha; /* This is the controlling param for Elastic Net*/

	MLPPUtilities::WeightDistributionType weight_init;

	bool _initialized;
};

#endif /* OutputLayer_hpp */
