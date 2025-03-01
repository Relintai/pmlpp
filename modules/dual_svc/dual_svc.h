#ifndef MLPP_DUAL_SVC_H
#define MLPP_DUAL_SVC_H

/*************************************************************************/
/*  dual_svc.h                                                           */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2023-present Péter Magyar.                              */
/* Copyright (c) 2022-2023 Marc Melikyan                                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

// http://disp.ee.ntu.edu.tw/~pujols/Support%20Vector%20Machine.pdf
// http://ciml.info/dl/v0_99/ciml-v0_99-ch11.pdf
// Were excellent for the practical intution behind the dual formulation.

#ifdef USING_SFW
#include "sfw.h"
#else
#include "core/math/math_defs.h"

#include "core/object/reference.h"
#endif

#include "../core/mlpp_matrix.h"
#include "../core/mlpp_vector.h"

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
