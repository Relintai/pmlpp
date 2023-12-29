
#ifndef MLPP_MULTINOMIAL_NB_H
#define MLPP_MULTINOMIAL_NB_H

#include "core/containers/hash_map.h"
#include "core/containers/vector.h"
#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPMultinomialNB : public Reference {
	GDCLASS(MLPPMultinomialNB, Reference);

public:
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output_set();
	void set_output_set(const Ref<MLPPVector> &val);

	real_t get_class_num();
	void set_class_num(const real_t val);

	Ref<MLPPVector> model_set_test(const Ref<MLPPMatrix> &X);
	real_t model_test(const Ref<MLPPVector> &x);

	real_t score();

	bool is_initialized();
	void initialize();

	MLPPMultinomialNB(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, int class_num);

	MLPPMultinomialNB();
	~MLPPMultinomialNB();

protected:
	void compute_theta();
	void evaluate();

	static void _bind_methods();

	// Model Params
	Ref<MLPPVector> _priors;

	Vector<HashMap<real_t, int>> _theta;
	Ref<MLPPVector> _vocab;
	int _class_num;

	// Datasets
	Ref<MLPPMatrix> _input_set;
	Ref<MLPPVector> _output_set;
	Ref<MLPPVector> _y_hat;

	bool _initialized;
};

#endif /* MultinomialNB_hpp */
