
#ifndef MLPP_MULTI_OUTPUT_LAYER_OLD_H
#define MLPP_MULTI_OUTPUT_LAYER_OLD_H

//
//  MultiOutputLayer.hpp
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

#include <map>
#include <string>
#include <vector>

class MLPPOldMultiOutputLayer {
public:
	MLPPOldMultiOutputLayer(int n_output, int n_hidden, std::string activation, std::string cost, std::vector<std::vector<real_t>> input, std::string weightInit, std::string reg, real_t lambda, real_t alpha);

	int n_output;
	int n_hidden;
	std::string activation;
	std::string cost;

	std::vector<std::vector<real_t>> input;

	std::vector<std::vector<real_t>> weights;
	std::vector<real_t> bias;

	std::vector<std::vector<real_t>> z;
	std::vector<std::vector<real_t>> a;

	std::map<std::string, std::vector<std::vector<real_t>> (MLPPActivation::*)(std::vector<std::vector<real_t>>, bool)> activation_map;
	std::map<std::string, std::vector<real_t> (MLPPActivation::*)(std::vector<real_t>, bool)> activationTest_map;
	std::map<std::string, real_t (MLPPCost::*)(std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>)> cost_map;
	std::map<std::string, std::vector<std::vector<real_t>> (MLPPCost::*)(std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>)> costDeriv_map;

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

#endif /* MultiOutputLayer_hpp */
