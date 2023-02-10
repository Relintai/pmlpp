
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

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../regularization/reg.h"

class MLPPSVC : public Reference {
	GDCLASS(MLPPSVC, Reference);

public:
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output_set();
	void set_output_set(const Ref<MLPPMatrix> &val);

	real_t get_c();
	void set_c(const real_t val);

	Ref<MLPPVector> model_set_test(const Ref<MLPPMatrix> &X);
	real_t model_test(const Ref<MLPPVector> &x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	void save(const String &file_name);

	bool is_initialized();
	void initialize();

	MLPPSVC(const Ref<MLPPMatrix> &input_set, const Ref<MLPPVector> &output_set, real_t c);

	MLPPSVC();
	~MLPPSVC();

protected:
	real_t cost(const Ref<MLPPVector> &z, const Ref<MLPPVector> &y, const Ref<MLPPVector> &weights, real_t c);

	Ref<MLPPVector> evaluatem(const Ref<MLPPMatrix> &X);
	Ref<MLPPVector> propagatem(const Ref<MLPPMatrix> &X);

	real_t evaluatev(const Ref<MLPPVector> &x);
	real_t propagatev(const Ref<MLPPVector> &x);

	void forward_pass();

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPVector> _output_set;

	Ref<MLPPVector> _z;
	Ref<MLPPVector> _y_hat;
	Ref<MLPPVector> _weights;
	real_t _bias;

	real_t _c;
	int _n;
	int _k;

	bool _initialized;
};

#endif /* SVC_hpp */
