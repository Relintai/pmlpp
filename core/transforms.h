#ifndef MLPP_TRANSFORMS_H
#define MLPP_TRANSFORMS_H

/*************************************************************************/
/*  transforms.h                                                         */
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

//
//  Transforms.hpp
//
//

#ifdef USING_SFW
#include "sfw.h"
#else
#include "core/math/math_defs.h"

#include "core/object/reference.h"
#endif

#include "../core/mlpp_matrix.h"

class MLPPTransforms : public Reference {
	GDCLASS(MLPPTransforms, Reference);

public:
	Ref<MLPPMatrix> discrete_cosine_transform(const Ref<MLPPMatrix> &p_A);

protected:
	static void _bind_methods();
};

#endif /* Transforms_hpp */
