#ifndef MLPP_PROBIT_REG_H
#define MLPP_PROBIT_REG_H

/*************************************************************************/
/*  probit_reg.h                                                         */
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

#ifdef USING_SFW
#include "sfw.h"
#else
#include "core/math/math_defs.h"

#include "core/object/resource.h"
#endif

#include "../core/mlpp_matrix.h"
#include "../core/mlpp_vector.h"

#include "../core/reg.h"

class MLPPProbitReg : public Resource {
	GDCLASS(MLPPProbitReg, Resource);

public:
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output_set();
	void set_output_set(const Ref<MLPPVector> &val);

	MLPPReg::RegularizationType get_reg();
	void set_reg(const MLPPReg::RegularizationType val);

	real_t get_lambda();
	void set_lambda(const real_t val);

	real_t get_alpha();
	void set_alpha(const real_t val);

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

	void train_gradient_descent(real_t learning_rate, int max_epoch = 0, bool ui = false);
	void train_mle(real_t learning_rate, int max_epoch = 0, bool ui = false);
	void train_sgd(real_t learning_rate, int max_epoch = 0, bool ui = false);
	void train_mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	bool needs_init() const;
	void initialize();

	MLPPProbitReg(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, MLPPReg::RegularizationType p_reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t p_lambda = 0.5, real_t p_alpha = 0.5);

	MLPPProbitReg();
	~MLPPProbitReg();

protected:
	real_t cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);

	Ref<MLPPVector> evaluatem(const Ref<MLPPMatrix> &X);
	Ref<MLPPVector> propagatem(const Ref<MLPPMatrix> &X);

	real_t evaluatev(const Ref<MLPPVector> &x);
	real_t propagatev(const Ref<MLPPVector> &x);

	void forward_pass();

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPVector> _output_set;
	// Regularization Params
	MLPPReg::RegularizationType _reg;
	real_t _lambda;
	real_t _alpha; /* This is the controlling param for Elastic Net*/

	Ref<MLPPVector> _z;
	Ref<MLPPVector> _y_hat;
	Ref<MLPPVector> _weights;
	real_t _bias;
};

#endif /* ProbitReg_hpp */
