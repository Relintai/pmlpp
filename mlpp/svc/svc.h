
#ifndef MLPP_SVC_H
#define MLPP_SVC_H



// https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2
// Illustratd a practical definition of the Hinge Loss function and its gradient when optimizing with SGD.

#include "core/math/math_defs.h"

#include "core/object/resource.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../regularization/reg.h"

class MLPPSVC : public Resource {
	GDCLASS(MLPPSVC, Resource);

public:
	Ref<MLPPMatrix> get_input_set() const;
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output_set() const;
	void set_output_set(const Ref<MLPPMatrix> &val);

	real_t get_c() const;
	void set_c(const real_t val);

	Ref<MLPPVector> data_z_get() const;
	void data_z_set(const Ref<MLPPVector> &val);

	Ref<MLPPVector> data_y_hat_get() const;
	void data_y_hat_set(const Ref<MLPPVector> &val);

	Ref<MLPPVector> data_weights_get() const;
	void data_weights_set(const Ref<MLPPVector> &val);

	real_t data_bias_get() const;
	void data_bias_set(const real_t val);

	Ref<MLPPVector> model_set_test(const Ref<MLPPMatrix> &X);
	real_t model_test(const Ref<MLPPVector> &x);

	void train_gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void train_sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void train_mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	bool needs_init() const;
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
	real_t _c;

	Ref<MLPPVector> _z;
	Ref<MLPPVector> _y_hat;
	Ref<MLPPVector> _weights;
	real_t _bias;
};

#endif /* SVC_hpp */
