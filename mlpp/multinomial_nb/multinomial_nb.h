
#ifndef MLPP_MULTINOMIAL_NB_H
#define MLPP_MULTINOMIAL_NB_H

//
//  MultinomialNB.hpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <map>
#include <vector>

class MLPPMultinomialNB : public Reference {
	GDCLASS(MLPPMultinomialNB, Reference);

public:
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output_set();
	void set_output_set(const Ref<MLPPMatrix> &val);

	real_t get_class_num();
	void set_class_num(const real_t val);

	std::vector<real_t> model_set_test(std::vector<std::vector<real_t>> X);
	real_t model_test(std::vector<real_t> x);

	real_t score();

	bool is_initialized();
	void initialize();

	MLPPMultinomialNB(std::vector<std::vector<real_t>> _input_set, std::vector<real_t> _output_set, int class_num);

	MLPPMultinomialNB();
	~MLPPMultinomialNB();

protected:
	void compute_theta();
	void evaluate();

	static void _bind_methods();

	// Model Params
	std::vector<real_t> _priors;

	std::vector<std::map<real_t, int>> _theta;
	std::vector<real_t> _vocab;
	int _class_num;

	// Datasets
	std::vector<std::vector<real_t>> _input_set;
	std::vector<real_t> _output_set;
	std::vector<real_t> _y_hat;

	bool _initialized;
};

#endif /* MultinomialNB_hpp */
