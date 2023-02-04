
#ifndef MLPP_MANN_H
#define MLPP_MANN_H

//
//  MANN.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/math/math_defs.h"

#include "../hidden_layer/hidden_layer.h"
#include "../multi_output_layer/multi_output_layer.h"

#include <string>
#include <vector>



class MLPPMANN {
public:
	MLPPMANN(std::vector<std::vector<real_t>> inputSet, std::vector<std::vector<real_t>> outputSet);
	~MLPPMANN();
	std::vector<std::vector<real_t>> modelSetTest(std::vector<std::vector<real_t>> X);
	std::vector<real_t> modelTest(std::vector<real_t> x);
	void gradientDescent(real_t learning_rate, int max_epoch, bool UI = 1);
	real_t score();
	void save(std::string fileName);

	void addLayer(int n_hidden, std::string activation, std::string weightInit = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);
	void addOutputLayer(std::string activation, std::string loss, std::string weightInit = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);

private:
	real_t Cost(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);
	void forwardPass();

	std::vector<std::vector<real_t>> inputSet;
	std::vector<std::vector<real_t>> outputSet;
	std::vector<std::vector<real_t>> y_hat;

	std::vector<MLPPOldHiddenLayer> network;
	MLPPOldMultiOutputLayer *outputLayer;

	int n;
	int k;
	int n_output;
};


#endif /* MANN_hpp */