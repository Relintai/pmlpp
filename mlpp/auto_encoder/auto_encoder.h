
#ifndef MLPP_AUTO_ENCODER_H
#define MLPP_AUTO_ENCODER_H

//
//  AutoEncoder.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../regularization/reg.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPAutoEncoder : public Reference {
	GDCLASS(MLPPAutoEncoder, Reference);

public:
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	int get_n_hidden();
	void set_n_hidden(const int val);

	Ref<MLPPMatrix> model_set_test(const Ref<MLPPMatrix> &X);
	Ref<MLPPVector> model_test(const Ref<MLPPVector> &x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	void save(const String &file_name);

	MLPPAutoEncoder(const Ref<MLPPMatrix> &p_input_set, int p_n_hidden);

	MLPPAutoEncoder();
	~MLPPAutoEncoder();

protected:
	real_t cost(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	Ref<MLPPVector> evaluatev(const Ref<MLPPVector> &x);

	struct PropagateVResult {
		Ref<MLPPVector> z2;
		Ref<MLPPVector> a2;
	};

	PropagateVResult propagatev(const Ref<MLPPVector> &x);

	Ref<MLPPMatrix> evaluatem(const Ref<MLPPMatrix> &X);

	struct PropagateMResult {
		Ref<MLPPMatrix> z2;
		Ref<MLPPMatrix> a2;
	};

	PropagateMResult propagatem(const Ref<MLPPMatrix> &X);

	void forward_pass();

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPMatrix> _y_hat;

	Ref<MLPPMatrix> _weights1;
	Ref<MLPPMatrix> _weights2;

	Ref<MLPPVector> _bias1;
	Ref<MLPPVector> _bias2;

	Ref<MLPPMatrix> _z2;
	Ref<MLPPMatrix> _a2;

	int _n;
	int _k;
	int _n_hidden;

	bool _initialized;
};

#endif /* AutoEncoder_hpp */
