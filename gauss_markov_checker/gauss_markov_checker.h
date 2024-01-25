#ifndef MLPP_GAUSS_MARKOV_CHECKER_H
#define MLPP_GAUSS_MARKOV_CHECKER_H

/*************************************************************************/
/*  gauss_markov_checker.h                                               */
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

#include <string>
#include <vector>

class MLPPGaussMarkovChecker : public Reference {
	GDCLASS(MLPPGaussMarkovChecker, Reference);

public:
	/*
	void checkGMConditions(std::vector<real_t> eps);

	// Independent, 3 Gauss-Markov Conditions
	bool arithmeticMean(std::vector<real_t> eps); // 1) Arithmetic Mean of 0.
	bool homoscedasticity(std::vector<real_t> eps); // 2) Homoscedasticity
	bool exogeneity(std::vector<real_t> eps); // 3) Cov of any 2 non-equal eps values = 0.
	*/

protected:
	static void _bind_methods();
};

#endif /* GaussMarkovChecker_hpp */
