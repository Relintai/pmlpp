
#ifndef MLPP_DUAL_SVC_H
#define MLPP_DUAL_SVC_H

//
//  DualSVC.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//
// http://disp.ee.ntu.edu.tw/~pujols/Support%20Vector%20Machine.pdf
// http://ciml.info/dl/v0_99/ciml-v0_99-ch11.pdf
// Were excellent for the practical intution behind the dual formulation.

#include "core/math/math_defs.h"

#include <string>
#include <vector>



class MLPPDualSVC {
public:
	MLPPDualSVC(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, real_t C, std::string kernel = "Linear");
	MLPPDualSVC(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, real_t C, std::string kernel, real_t p, real_t c);

	std::vector<real_t> modelSetTest(std::vector<std::vector<real_t>> X);
	real_t modelTest(std::vector<real_t> x);
	void gradientDescent(real_t learning_rate, int max_epoch, bool UI = 1);
	void SGD(real_t learning_rate, int max_epoch, bool UI = 1);
	void MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI = 1);
	real_t score();
	void save(std::string fileName);

private:
	void init();

	real_t Cost(std::vector<real_t> alpha, std::vector<std::vector<real_t>> X, std::vector<real_t> y);

	std::vector<real_t> Evaluate(std::vector<std::vector<real_t>> X);
	std::vector<real_t> propagate(std::vector<std::vector<real_t>> X);
	real_t Evaluate(std::vector<real_t> x);
	real_t propagate(std::vector<real_t> x);
	void forwardPass();

	void alphaProjection();

	real_t kernelFunction(std::vector<real_t> v, std::vector<real_t> u, std::string kernel);
	std::vector<std::vector<real_t>> kernelFunction(std::vector<std::vector<real_t>> U, std::vector<std::vector<real_t>> V, std::string kernel);

	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;
	std::vector<real_t> z;
	std::vector<real_t> y_hat;
	real_t bias;

	std::vector<real_t> alpha;
	std::vector<std::vector<real_t>> K;

	real_t C;
	int n;
	int k;

	std::string kernel;
	real_t p; // Poly
	real_t c; // Poly

	// UI Portion
	void UI(int epoch, real_t cost_prev);
};


#endif /* DualSVC_hpp */
