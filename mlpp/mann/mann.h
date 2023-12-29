
#ifndef MLPP_MANN_H
#define MLPP_MANN_H


#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../regularization/reg.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../hidden_layer/hidden_layer.h"
#include "../multi_output_layer/multi_output_layer.h"

class MLPPMANN : public Reference {
	GDCLASS(MLPPMANN, Reference);

public:
	/*
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> get_output_set();
	void set_output_set(const Ref<MLPPMatrix> &val);
	*/

	Ref<MLPPMatrix> model_set_test(const Ref<MLPPMatrix> &X);
	Ref<MLPPVector> model_test(const Ref<MLPPVector> &x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	real_t score();

	void save(const String &file_name);

	void add_layer(int n_hidden, MLPPActivation::ActivationFunction activation, MLPPUtilities::WeightDistributionType weight_init = MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT, MLPPReg::RegularizationType reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t lambda = 0.5, real_t alpha = 0.5);
	void add_output_layer(MLPPActivation::ActivationFunction activation, MLPPCost::CostTypes loss, MLPPUtilities::WeightDistributionType weight_init = MLPPUtilities::WEIGHT_DISTRIBUTION_TYPE_DEFAULT, MLPPReg::RegularizationType reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t lambda = 0.5, real_t alpha = 0.5);

	bool is_initialized();
	void initialize();

	MLPPMANN(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPMatrix> &p_output_set);

	MLPPMANN();
	~MLPPMANN();

private:
	real_t cost(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	void forward_pass();

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPMatrix> _output_set;
	Ref<MLPPMatrix> _y_hat;

	Vector<Ref<MLPPHiddenLayer>> _network;
	Ref<MLPPMultiOutputLayer> _output_layer;

	int _n;
	int _k;
	int _n_output;

	bool _initialized;
};

#endif /* MANN_hpp */