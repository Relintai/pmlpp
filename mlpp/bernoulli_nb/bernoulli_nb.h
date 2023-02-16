
#ifndef MLPP_BERNOULLI_NB_H
#define MLPP_BERNOULLI_NB_H

//
//  BernoulliNB.hpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include "core/containers/hash_map.h"
#include "core/containers/vector.h"
#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPBernoulliNB : public Reference {
	GDCLASS(MLPPBernoulliNB, Reference);

public:
	Ref<MLPPVector> model_set_test(const Ref<MLPPMatrix> &X);
	real_t model_test(const Ref<MLPPVector> &x);

	real_t score();

	MLPPBernoulliNB(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set);

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

	Vector<HashMap<real_t, int>> _theta;
	Ref<MLPPVector> _vocab;
	int _class_num;

	// Datasets
	Ref<MLPPMatrix> _input_set;
	Ref<MLPPVector> _output_set;
	Ref<MLPPVector> _y_hat;
};

#endif /* BernoulliNB_hpp */