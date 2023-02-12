
#ifndef MLPP_BERNOULLI_NB_H
#define MLPP_BERNOULLI_NB_H

//
//  BernoulliNB.hpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include <map>
#include <vector>

class MLPPBernoulliNB : public Reference {
	GDCLASS(MLPPBernoulliNB, Reference);

public:
	std::vector<real_t> model_set_test(std::vector<std::vector<real_t>> X);
	real_t model_test(std::vector<real_t> x);

	real_t score();

	MLPPBernoulliNB(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set);

	MLPPBernoulliNB();
	~MLPPBernoulliNB();

protected:
	void compute_vocab();
	void compute_theta();
	void evaluate();

	static void _bind_methods();

	// Model Params
	real_t _prior_1;
	real_t _prior_0;

	std::vector<std::map<real_t, int>> _theta;
	std::vector<real_t> _vocab;
	int _class_num;

	// Datasets
	std::vector<std::vector<real_t>> _input_set;
	std::vector<real_t> _output_set;
	std::vector<real_t> _y_hat;
};

#endif /* BernoulliNB_hpp */