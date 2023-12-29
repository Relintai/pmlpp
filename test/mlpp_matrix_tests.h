#ifndef MLPP_MATRIX_TESTS_H
#define MLPP_MATRIX_TESTS_H

/*************************************************************************/
/*  mlpp_matrix_tests.h                                                  */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2022-present Péter Magyar.                              */
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

#include "core/containers/vector.h"

#include "core/object/reference.h"

#include "core/string/ustring.h"

#include "mlpp_tests.h"

// TODO port this class to use the test module once it's working
// Also don't forget to remove it's bindings

class MLPPMatrix;
class MLPPVector;

class MLPPMatrixTests : public MLPPTests {
	GDCLASS(MLPPMatrixTests, MLPPTests);

public:
	void run_tests();

	void test_mlpp_matrix();

	void test_row_add();
	void test_row_add_pool_vector();
	void test_row_add_mlpp_vector();
	void test_rows_add_mlpp_matrix();

	void test_row_remove();
	void test_row_remove_unordered();

	void test_mlpp_matrix_mul();

	MLPPMatrixTests();
	~MLPPMatrixTests();

protected:
	static void _bind_methods();
};

#endif
