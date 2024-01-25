#ifndef MLPP_OUTLIER_FINDER_H
#define MLPP_OUTLIER_FINDER_H

/*************************************************************************/
/*  outlier_finder.h                                                     */
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

#include "core/object/reference.h"
#endif

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPOutlierFinder : public Reference {
	GDCLASS(MLPPOutlierFinder, Reference);

public:
	real_t get_threshold();
	void set_threshold(real_t val);

	Vector<Vector<real_t>> model_set_test(const Ref<MLPPMatrix> &input_set);
	Array model_set_test_bind(const Ref<MLPPMatrix> &input_set);

	PoolVector2iArray model_set_test_indices(const Ref<MLPPMatrix> &input_set);

	PoolRealArray model_test(const Ref<MLPPVector> &input_set);

	MLPPOutlierFinder(real_t threshold);

	MLPPOutlierFinder();
	~MLPPOutlierFinder();

protected:
	static void _bind_methods();

	real_t _threshold;
};

#endif /* OutlierFinder_hpp */
