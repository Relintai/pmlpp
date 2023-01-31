
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

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <map>
#include <string>
#include <vector>

class MLPPHiddenLayer : public Reference {
	GDCLASS(MLPPHiddenLayer, Reference);

public:
	int n_hidden;
	int activation;

	Ref<MLPPMatrix> input;

	Ref<MLPPMatrix> weights;
	Ref<MLPPVector> bias;

	Ref<MLPPMatrix> z;
	Ref<MLPPMatrix> a;

	HashMap<int, Ref<MLPPMatrix> (MLPPActivation::*)(const Ref<MLPPMatrix> &, bool)> activation_map;
	HashMap<int, Ref<MLPPVector> (MLPPActivation::*)(const Ref<MLPPVector> &, bool)> activation_test_map;

	Ref<MLPPVector> z_test;
	Ref<MLPPVector> a_test;

	Ref<MLPPMatrix> delta;

	// Regularization Params
	String reg;
	real_t lambda; /* Regularization Parameter */
	real_t alpha; /* This is the controlling param for Elastic Net*/

	String weight_init;

	void forward_pass();
	void test(const Ref<MLPPVector> &x);

	MLPPHiddenLayer(int p_n_hidden, int p_activation, Ref<MLPPMatrix> p_input, String p_weight_init, String p_reg, real_t p_lambda, real_t p_alpha);

	MLPPHiddenLayer();
	~MLPPHiddenLayer();
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