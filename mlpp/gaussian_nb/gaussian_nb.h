
#ifndef MLPP_GAUSSIAN_NB_H
#define MLPP_GAUSSIAN_NB_H



#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

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

	Ref<MLPPVector> model_set_test(const Ref<MLPPMatrix> &X);
	real_t model_test(const Ref<MLPPVector> &x);

	real_t score();

	bool is_initialized();
	void initialize();

	MLPPGaussianNB(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, int p_class_num);

	MLPPGaussianNB();
	~MLPPGaussianNB();

protected:
	void evaluate();

	static void _bind_methods();

	int _class_num;

	Ref<MLPPVector> _priors;
	Ref<MLPPVector> _mu;
	Ref<MLPPVector> _sigma;

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPVector> _output_set;

	Ref<MLPPVector> _y_hat;

	bool _initialized;
};

#endif /* GaussianNB_hpp */
