/*************************************************************************/
/*  transforms.cpp                                                       */
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

#include "transforms.h"

#include "../lin_alg/lin_alg.h"

#ifdef USING_SFW
#include "sfw.h"
#else
#include "core/math/math_funcs.h"
#endif

// DCT ii.
// https://www.mathworks.com/help/images/discrete-cosine-transform.html
Ref<MLPPMatrix> MLPPTransforms::discrete_cosine_transform(const Ref<MLPPMatrix> &p_A) {
	Ref<MLPPMatrix> A = p_A->scalar_addn(-128); // Center around 0.

	Size2i size = A->size();

	Ref<MLPPMatrix> B;
	B.instance();
	B->resize(size);

	real_t M = size.y;
	real_t inv_sqrt_M = 1 / Math::sqrt(M);
	real_t s2M = Math::sqrt(real_t(2) / real_t(M));

	for (int i = 0; i < size.y; i++) {
		for (int j = 0; j < size.x; j++) {
			real_t sum = 0;

			real_t alphaI;
			if (i == 0) {
				alphaI = inv_sqrt_M;
			} else {
				alphaI = s2M;
			}

			real_t alphaJ;
			if (j == 0) {
				alphaJ = inv_sqrt_M;
			} else {
				alphaJ = s2M;
			}

			for (int k = 0; k < size.y; k++) {
				for (int f = 0; f < size.x; f++) {
					sum += A->element_get(k, f) * Math::cos((Math_PI * i * (2 * k + 1)) / (2 * M)) * Math::cos((Math_PI * j * (2 * f + 1)) / (2 * M));
				}
			}

			B->element_set(i, j, sum * alphaI * alphaJ);
		}
	}
	return B;
}

void MLPPTransforms::_bind_methods() {
}
