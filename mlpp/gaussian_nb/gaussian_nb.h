
#ifndef MLPP_GAUSSIAN_NB_H
#define MLPP_GAUSSIAN_NB_H

//
//  GaussianNB.hpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <vector>

class MLPPGaussianNB : public Reference {
	GDCLASS(MLPPGaussianNB, Reference);

public:
	/*
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output_set();
	void set_output_set(const Ref<MLPPVector> &val);

	int get_class_num();
	void set_class_num(const int val);
	*/

	std::vector<real_t> model_set_test(std::vector<std::vector<real_t>> X);
	real_t model_test(std::vector<real_t> x);

	real_t score();

	bool is_initialized();
	void initialize();

	MLPPGaussianNB(std::vector<std::vector<real_t>> p_input_set, std::vector<real_t> p_output_set, int p_class_num);

	MLPPGaussianNB();
	~MLPPGaussianNB();

protected:
	void evaluate();

	static void _bind_methods();

	int _class_num;

	std::vector<real_t> _priors;
	std::vector<real_t> _mu;
	std::vector<real_t> _sigma;

	std::vector<std::vector<real_t>> _input_set;
	std::vector<real_t> _output_set;

	std::vector<real_t> _y_hat;

	bool _initialized;
};

#endif /* GaussianNB_hpp */
