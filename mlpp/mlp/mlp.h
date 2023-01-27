
#ifndef MLPP_MLP_H
#define MLPP_MLP_H

//
//  MLP.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/math/math_defs.h"

#include <map>
#include <string>
#include <vector>



class MLPPMLP {
public:
	MLPPMLP(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, int n_hidden, std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);
	std::vector<real_t> modelSetTest(std::vector<std::vector<real_t>> X);
	real_t modelTest(std::vector<real_t> x);
	void gradientDescent(real_t learning_rate, int max_epoch, bool UI = 1);
	void SGD(real_t learning_rate, int max_epoch, bool UI = 1);
	void MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI = 1);
	real_t score();
	void save(std::string fileName);

private:
	real_t Cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	std::vector<real_t> Evaluate(std::vector<std::vector<real_t>> X);
	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> propagate(std::vector<std::vector<real_t>> X);
	real_t Evaluate(std::vector<real_t> x);
	std::tuple<std::vector<real_t>, std::vector<real_t>> propagate(std::vector<real_t> x);
	void forwardPass();

	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;
	std::vector<real_t> y_hat;

	std::vector<std::vector<real_t>> weights1;
	std::vector<real_t> weights2;

	std::vector<real_t> bias1;
	real_t bias2;

	std::vector<std::vector<real_t>> z2;
	std::vector<std::vector<real_t>> a2;

	int n;
	int k;
	int n_hidden;

	// Regularization Params
	std::string reg;
	real_t lambda; /* Regularization Parameter */
	real_t alpha; /* This is the controlling param for Elastic Net*/
};


#endif /* MLP_hpp */
