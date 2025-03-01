#ifndef MLPP_CONVOLUTIONS_H
#define MLPP_CONVOLUTIONS_H

/*************************************************************************/
/*  convolutions.h                                                       */
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
#include "core/containers/vector.h"
#include "core/string/ustring.h"

#include "core/math/math_defs.h"

#include "core/object/reference.h"
#endif

#include "../core/mlpp_matrix.h"
#include "../core/mlpp_tensor3.h"
#include "../core/mlpp_vector.h"



class MLPPConvolutions : public Reference {
	GDCLASS(MLPPConvolutions, Reference);

public:
	enum PoolType {
		POOL_TYPE_AVERAGE = 0,
		POOL_TYPE_MIN,
		POOL_TYPE_MAX,
	};

	Ref<MLPPMatrix> convolve_2d(const Ref<MLPPMatrix> &input, const Ref<MLPPMatrix> &filter, const int S, const int P = 0);
	Ref<MLPPTensor3> convolve_3d(const Ref<MLPPTensor3> &input, const Ref<MLPPTensor3> &filter, const int S, const int P = 0);

	Ref<MLPPMatrix> pool_2d(const Ref<MLPPMatrix> &input, const int F, const int S, const PoolType type);
	Ref<MLPPTensor3> pool_3d(const Ref<MLPPTensor3> &input, const int F, const int S, const PoolType type);

	real_t global_pool_2d(const Ref<MLPPMatrix> &input, const PoolType type);
	Ref<MLPPVector> global_pool_3d(const Ref<MLPPTensor3> &input, const PoolType type);

	real_t gaussian_2d(const real_t x, const real_t y, const real_t std);
	Ref<MLPPMatrix> gaussian_filter_2d(const int size, const real_t std);

	Ref<MLPPMatrix> dx(const Ref<MLPPMatrix> &input);
	Ref<MLPPMatrix> dy(const Ref<MLPPMatrix> &input);

	Ref<MLPPMatrix> grad_magnitude(const Ref<MLPPMatrix> &input);
	Ref<MLPPMatrix> grad_orientation(const Ref<MLPPMatrix> &input);

	Ref<MLPPTensor3> compute_m(const Ref<MLPPMatrix> &input);
	Vector<Ref<MLPPMatrix>> compute_mv(const Ref<MLPPMatrix> &input);

	//TODO better data srtucture for this. Maybe IntMatrix?
	Vector<Vector<CharType>> harris_corner_detection(const Ref<MLPPMatrix> &input);

	Ref<MLPPMatrix> get_prewitt_horizontal() const;
	Ref<MLPPMatrix> get_prewitt_vertical() const;
	Ref<MLPPMatrix> get_sobel_horizontal() const;
	Ref<MLPPMatrix> get_sobel_vertical() const;
	Ref<MLPPMatrix> get_scharr_horizontal() const;
	Ref<MLPPMatrix> get_scharr_vertical() const;
	Ref<MLPPMatrix> get_roberts_horizontal() const;
	Ref<MLPPMatrix> get_roberts_vertical() const;

	MLPPConvolutions();

protected:
	static void _bind_methods();

	Ref<MLPPMatrix> _prewitt_horizontal;
	Ref<MLPPMatrix> _prewitt_vertical;
	Ref<MLPPMatrix> _sobel_horizontal;
	Ref<MLPPMatrix> _sobel_vertical;
	Ref<MLPPMatrix> _scharr_horizontal;
	Ref<MLPPMatrix> _scharr_vertical;
	Ref<MLPPMatrix> _roberts_horizontal;
	Ref<MLPPMatrix> _roberts_vertical;
};

#endif // Convolutions_hpp