
#ifndef MLPP_SVC_H
#define MLPP_SVC_H

//
//  SVC.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

// https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2
// Illustratd a practical definition of the Hinge Loss function and its gradient when optimizing with SGD.

#include "core/math/math_defs.h"

#include <string>
#include <vector>



class MLPPSVC {
public:
	MLPPSVC(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, real_t C);
	std::vector<real_t> modelSetTest(std::vector<std::vector<real_t>> X);
	real_t modelTest(std::vector<real_t> x);
	void gradientDescent(real_t learning_rate, int max_epoch, bool UI = false);
	void SGD(real_t learning_rate, int max_epoch, bool UI = false);
	void MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI = false);
	real_t score();
	void save(std::string fileName);

private:
	real_t Cost(std::vector<real_t> y_hat, std::vector<real_t> y, std::vector<real_t> weights, real_t C);

	std::vector<real_t> Evaluate(std::vector<std::vector<real_t>> X);
	std::vector<real_t> propagate(std::vector<std::vector<real_t>> X);
	real_t Evaluate(std::vector<real_t> x);
	real_t propagate(std::vector<real_t> x);
	void forwardPass();

	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;
	std::vector<real_t> z;
	std::vector<real_t> y_hat;
	std::vector<real_t> weights;
	real_t bias;

	real_t C;
	int n;
	int k;

	// UI Portion
	void UI(int epoch, real_t cost_prev);
};


#endif /* SVC_hpp */
