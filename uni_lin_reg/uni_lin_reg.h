#ifndef MLPP_UNI_LIN_REG_H
#define MLPP_UNI_LIN_REG_H

/*************************************************************************/
/*  uni_lin_reg.h                                                        */
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

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPUniLinReg : public Resource {
	GDCLASS(MLPPUniLinReg, Resource);

public:
	Ref<MLPPVector> get_input_set() const;
	void set_input_set(const Ref<MLPPVector> &val);

	Ref<MLPPVector> get_output_set() const;
	void set_output_set(const Ref<MLPPVector> &val);

	real_t get_b0() const;
	void set_b0(const real_t val);

	real_t get_b1() const;
	void set_b1(const real_t val);

	void train();

	Ref<MLPPVector> model_set_test(const Ref<MLPPVector> &x);
	real_t model_test(real_t x);

	MLPPUniLinReg(const Ref<MLPPVector> &p_input_set, const Ref<MLPPVector> &p_output_set);

	MLPPUniLinReg();
	~MLPPUniLinReg();

protected:
	static void _bind_methods();

	Ref<MLPPVector> _input_set;
	Ref<MLPPVector> _output_set;

	real_t _b0;
	real_t _b1;
};

#endif /* UniLinReg_hpp */
