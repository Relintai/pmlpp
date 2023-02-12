
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
	std::vector<real_t> model_set_test(std::vector<std::vector<real_t>> X);
	real_t model_test(std::vector<real_t> x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	//void SGD(real_t learning_rate, int max_epoch, bool ui = false);
	//void MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();
	void save(std::string file_name);

	MLPPDualSVC(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, real_t p_C, std::string p_kernel = "Linear");

	MLPPDualSVC();
	~MLPPDualSVC();

private:
	void init();

	real_t cost(std::vector<real_t> alpha, std::vector<std::vector<real_t>> X, std::vector<real_t> y);

	real_t evaluatev(std::vector<real_t> x);
	real_t propagatev(std::vector<real_t> x);

	std::vector<real_t> evaluatem(std::vector<std::vector<real_t>> X);
	std::vector<real_t> propagatem(std::vector<std::vector<real_t>> X);

	void forward_pass();

	void alpha_projection();

	real_t kernel_functionv(std::vector<real_t> v, std::vector<real_t> u, std::string kernel);
	std::vector<std::vector<real_t>> kernel_functionm(std::vector<std::vector<real_t>> U, std::vector<std::vector<real_t>> V, std::string kernel);

	std::vector<std::vector<real_t>> _input_set;
	std::vector<real_t> _output_set;
	std::vector<real_t> _z;
	std::vector<real_t> _y_hat;
	real_t _bias;

	std::vector<real_t> _alpha;
	std::vector<std::vector<real_t>> _K;

	real_t _C;
	int _n;
	int _k;

	std::string _kernel;
	real_t _p; // Poly
	real_t _c; // Poly
};

#endif /* DualSVC_hpp */
