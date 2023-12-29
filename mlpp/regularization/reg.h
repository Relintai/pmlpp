#ifndef MLPP_REG_H
#define MLPP_REG_H

/*************************************************************************/
/*  reg.h                                                                */
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

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <string>
#include <vector>

class MLPPReg : public Reference {
	GDCLASS(MLPPReg, Reference);

public:
	enum RegularizationType {
		REGULARIZATION_TYPE_NONE = 0,
		REGULARIZATION_TYPE_RIDGE,
		REGULARIZATION_TYPE_LASSO,
		REGULARIZATION_TYPE_ELASTIC_NET,
		REGULARIZATION_TYPE_WEIGHT_CLIPPING,
	};

	real_t reg_termv(const Ref<MLPPVector> &weights, real_t lambda, real_t alpha, RegularizationType reg);
	real_t reg_termm(const Ref<MLPPMatrix> &weights, real_t lambda, real_t alpha, RegularizationType reg);

	Ref<MLPPVector> reg_weightsv(const Ref<MLPPVector> &weights, real_t lambda, real_t alpha, RegularizationType reg);
	Ref<MLPPMatrix> reg_weightsm(const Ref<MLPPMatrix> &weights, real_t lambda, real_t alpha, RegularizationType reg);

	Ref<MLPPVector> reg_deriv_termv(const Ref<MLPPVector> &weights, real_t lambda, real_t alpha, RegularizationType reg);
	Ref<MLPPMatrix> reg_deriv_termm(const Ref<MLPPMatrix> &weights, real_t lambda, real_t alpha, RegularizationType reg);

	MLPPReg();
	~MLPPReg();

protected:
	static void _bind_methods();

private:
	real_t reg_deriv_termvr(const Ref<MLPPVector> &weights, real_t lambda, real_t alpha, RegularizationType reg, int j);
	real_t reg_deriv_termmr(const Ref<MLPPMatrix> &weights, real_t lambda, real_t alpha, RegularizationType reg, int i, int j);
};

VARIANT_ENUM_CAST(MLPPReg::RegularizationType);

#endif /* Reg_hpp */
