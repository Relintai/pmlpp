
#ifndef MLPP_DUAL_SVC_H
#define MLPP_DUAL_SVC_H


// http://disp.ee.ntu.edu.tw/~pujols/Support%20Vector%20Machine.pdf
// http://ciml.info/dl/v0_99/ciml-v0_99-ch11.pdf
// Were excellent for the practical intution behind the dual formulation.

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPDualSVC : public Reference {
	GDCLASS(MLPPDualSVC, Reference);

public:
	enum KernelMethod {
		KERNEL_METHOD_LINEAR = 0,
	};

public:
	Ref<MLPPVector> model_set_test(const Ref<MLPPMatrix> &X);
	real_t model_test(const Ref<MLPPVector> &x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	//void SGD(real_t learning_rate, int max_epoch, bool ui = false);
	//void MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();
	void save(const String &file_name);

	MLPPDualSVC(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, real_t p_C, KernelMethod p_kernel = KERNEL_METHOD_LINEAR);

	MLPPDualSVC();
	~MLPPDualSVC();

protected:
	void init();

	real_t cost(const Ref<MLPPVector> &alpha, const Ref<MLPPMatrix> &X, const Ref<MLPPVector> &y);

	real_t evaluatev(const Ref<MLPPVector> &x);
	real_t propagatev(const Ref<MLPPVector> &x);

	Ref<MLPPVector> evaluatem(const Ref<MLPPMatrix> &X);
	Ref<MLPPVector> propagatem(const Ref<MLPPMatrix> &X);

	void forward_pass();

	void alpha_projection();

	real_t kernel_functionv(const Ref<MLPPVector> &v, const Ref<MLPPVector> &u, KernelMethod kernel);
	Ref<MLPPMatrix> kernel_functionm(const Ref<MLPPMatrix> &U, const Ref<MLPPMatrix> &V, KernelMethod kernel);

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPVector> _output_set;
	Ref<MLPPVector> _z;
	Ref<MLPPVector> _y_hat;
	real_t _bias;

	Ref<MLPPVector> _alpha;
	Ref<MLPPMatrix> _K;

	real_t _C;
	int _n;
	int _k;

	KernelMethod _kernel;
	real_t _p; // Poly
	real_t _c; // Poly
};

#endif /* DualSVC_hpp */
