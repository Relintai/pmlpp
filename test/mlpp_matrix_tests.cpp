/*************************************************************************/
/*  mlpp_matrix_tests.cpp                                                */
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

#include "mlpp_matrix_tests.h"

#include "core/log/logger.h"

#include "../mlpp/lin_alg/mlpp_matrix.h"

void MLPPMatrixTests::run_tests() {
	PLOG_MSG("RUNNIG MLPPMatrixTests!");

	PLOG_TRACE("test_mlpp_matrix()");
	test_mlpp_matrix();

	PLOG_TRACE("test_row_add()");
	test_row_add();
	PLOG_TRACE("test_row_add_pool_vector()");
	test_row_add_pool_vector();
	PLOG_TRACE("test_row_add_mlpp_vector()");
	test_row_add_mlpp_vector();
	PLOG_TRACE("test_rows_add_mlpp_matrix()");
	test_rows_add_mlpp_matrix();

	PLOG_TRACE("test_row_remove()");
	test_row_remove();
	PLOG_TRACE("test_row_remove_unordered()");
	test_row_remove_unordered();

	PLOG_TRACE("test_mlpp_matrix_mul()");
	test_mlpp_matrix_mul();
}

void MLPPMatrixTests::test_mlpp_matrix() {
	const real_t A[] = {
		1, 0, 0, 0, //
		0, 1, 0, 0, //
		0, 0, 1, 0, //
		0, 0, 0, 1, //
	};

	Ref<MLPPMatrix> rmat(memnew(MLPPMatrix(A, 4, 4)));

	Ref<MLPPMatrix> rmat2;
	rmat2.instance();
	rmat2->set_from_ptr(A, 4, 4);

	is_approx_equals_mat(rmat, rmat2, "set_from_ptr test.");

	rmat2->set_from_ptr(A, 4, 4);

	is_approx_equals_mat(rmat, rmat2, "re-set_from_ptr test.");
}

void MLPPMatrixTests::test_row_add() {
	const real_t A[] = {
		1, 2, 3, 4, //
	};

	const real_t B[] = {
		1, 2, 3, 4, //
		1, 2, 3, 4, //
	};

	const real_t C[] = {
		1, 2, 3, 4, //
		1, 2, 3, 4, //
		1, 2, 3, 4, //
	};

	Vector<real_t> rv;
	rv.push_back(1);
	rv.push_back(2);
	rv.push_back(3);
	rv.push_back(4);

	Ref<MLPPMatrix> rmata(memnew(MLPPMatrix(A, 1, 4)));
	Ref<MLPPMatrix> rmatb(memnew(MLPPMatrix(B, 2, 4)));
	Ref<MLPPMatrix> rmatc(memnew(MLPPMatrix(C, 3, 4)));

	Ref<MLPPMatrix> rmat;
	rmat.instance();

	rmat->row_add(rv);
	is_approx_equals_mat(rmata, rmat, "rmat->row_add(rv);");

	rmat->row_add(rv);
	is_approx_equals_mat(rmatb, rmat, "rmat->row_add(rv);");

	rmat->row_add(rv);
	is_approx_equals_mat(rmatc, rmat, "rmat->row_add(rv);");
}
void MLPPMatrixTests::test_row_add_pool_vector() {
	const real_t A[] = {
		1, 2, 3, 4, //
	};

	const real_t B[] = {
		1, 2, 3, 4, //
		1, 2, 3, 4, //
	};

	const real_t C[] = {
		1, 2, 3, 4, //
		1, 2, 3, 4, //
		1, 2, 3, 4, //
	};

	PoolVector<real_t> rv;
	rv.push_back(1);
	rv.push_back(2);
	rv.push_back(3);
	rv.push_back(4);

	Ref<MLPPMatrix> rmata(memnew(MLPPMatrix(A, 1, 4)));
	Ref<MLPPMatrix> rmatb(memnew(MLPPMatrix(B, 2, 4)));
	Ref<MLPPMatrix> rmatc(memnew(MLPPMatrix(C, 3, 4)));

	Ref<MLPPMatrix> rmat;
	rmat.instance();

	rmat->row_add_pool_vector(rv);
	is_approx_equals_mat(rmata, rmat, "rmat->row_add_pool_vector(rv);");

	rmat->row_add_pool_vector(rv);
	is_approx_equals_mat(rmatb, rmat, "rmat->row_add_pool_vector(rv);");

	rmat->row_add_pool_vector(rv);
	is_approx_equals_mat(rmatc, rmat, "rmat->row_add_pool_vector(rv);");
}
void MLPPMatrixTests::test_row_add_mlpp_vector() {
	const real_t A[] = {
		1, 2, 3, 4, //
	};

	const real_t B[] = {
		1, 2, 3, 4, //
		1, 2, 3, 4, //
	};

	const real_t C[] = {
		1, 2, 3, 4, //
		1, 2, 3, 4, //
		1, 2, 3, 4, //
	};

	Ref<MLPPVector> rv;
	rv.instance();
	rv->push_back(1);
	rv->push_back(2);
	rv->push_back(3);
	rv->push_back(4);

	Ref<MLPPMatrix> rmata(memnew(MLPPMatrix(A, 1, 4)));
	Ref<MLPPMatrix> rmatb(memnew(MLPPMatrix(B, 2, 4)));
	Ref<MLPPMatrix> rmatc(memnew(MLPPMatrix(C, 3, 4)));

	Ref<MLPPMatrix> rmat;
	rmat.instance();

	rmat->row_add_mlpp_vector(rv);
	is_approx_equals_mat(rmata, rmat, "rmat->row_add_mlpp_vector(rv);");

	rmat->row_add_mlpp_vector(rv);
	is_approx_equals_mat(rmatb, rmat, "rmat->row_add_mlpp_vector(rv);");

	rmat->row_add_mlpp_vector(rv);
	is_approx_equals_mat(rmatc, rmat, "rmat->row_add_mlpp_vector(rv);");
}
void MLPPMatrixTests::test_rows_add_mlpp_matrix() {
	const real_t A[] = {
		1, 2, 3, 4 //
	};

	const real_t B[] = {
		1, 2, 3, 4, //
		1, 2, 3, 4, //
	};

	const real_t C[] = {
		1, 2, 3, 4, //
		1, 2, 3, 4, //
		1, 2, 3, 4, //
	};

	//const real_t r[] = {
	//	1, 2, 3, 4
	//};

	PoolVector<real_t> rvp;
	rvp.push_back(1);
	rvp.push_back(2);
	rvp.push_back(3);
	rvp.push_back(4);

	Ref<MLPPMatrix> rv;
	rv.instance();
	rv->row_add_pool_vector(rvp);

	Ref<MLPPMatrix> rmata(memnew(MLPPMatrix(A, 1, 4)));
	Ref<MLPPMatrix> rmatb(memnew(MLPPMatrix(B, 2, 4)));
	Ref<MLPPMatrix> rmatc(memnew(MLPPMatrix(C, 3, 4)));

	Ref<MLPPMatrix> rmat;
	rmat.instance();

	rmat->rows_add_mlpp_matrix(rv);
	is_approx_equals_mat(rmata, rmat, "rmat->rows_add_mlpp_matrix(rv);");

	rmat->rows_add_mlpp_matrix(rv);
	is_approx_equals_mat(rmatb, rmat, "rmat->rows_add_mlpp_matrix(rv);");

	rmat->rows_add_mlpp_matrix(rv);
	is_approx_equals_mat(rmatc, rmat, "rmat->rows_add_mlpp_matrix(rv);");
}

void MLPPMatrixTests::test_row_remove() {
	const real_t A[] = {
		1, 2, 3, 4, //
		5, 6, 7, 8, //
		9, 10, 11, 12, //
	};

	const real_t B[] = {
		1, 2, 3, 4, //
		5, 6, 7, 8, //
	};

	const real_t C[] = {
		1, 2, 3, 4, //
	};

	const real_t D[] = {
		1, 2, 3, 4, //
		5, 6, 7, 8, //
		13, 14, 15, 16, //
		9, 10, 11, 12, //
	};

	Ref<MLPPMatrix> rmata(memnew(MLPPMatrix(A, 3, 4)));
	Ref<MLPPMatrix> rmatb(memnew(MLPPMatrix(B, 2, 4)));
	Ref<MLPPMatrix> rmatc(memnew(MLPPMatrix(C, 1, 4)));

	Ref<MLPPMatrix> rmat;
	rmat.instance();
	rmat->set_from_ptr(D, 4, 4);

	rmat->row_remove(2);
	is_approx_equals_mat(rmat, rmata, "rmat->row_remove(2);");

	rmat->row_remove(2);
	is_approx_equals_mat(rmat, rmatb, "rmat->row_remove(2);");

	rmat->row_remove(1);
	is_approx_equals_mat(rmat, rmatc, "rmat->row_remove(1);");
}
void MLPPMatrixTests::test_row_remove_unordered() {
	const real_t A[] = {
		1, 2, 3, 4, //
		13, 14, 15, 16, //
		9, 10, 11, 12, //
	};

	const real_t B[] = {
		9, 10, 11, 12, //
		13, 14, 15, 16, //
	};

	const real_t C[] = {
		9, 10, 11, 12, //
	};

	const real_t D[] = {
		1, 2, 3, 4, //
		5, 6, 7, 8, //
		9, 10, 11, 12, //
		13, 14, 15, 16, //
	};

	Ref<MLPPMatrix> rmata(memnew(MLPPMatrix(A, 3, 4)));
	Ref<MLPPMatrix> rmatb(memnew(MLPPMatrix(B, 2, 4)));
	Ref<MLPPMatrix> rmatc(memnew(MLPPMatrix(C, 1, 4)));
	Ref<MLPPMatrix> rmat(memnew(MLPPMatrix(D, 4, 4)));

	rmat->row_remove_unordered(1);
	is_approx_equals_mat(rmat, rmata, "rmat->row_remove_unordered(1);");

	rmat->row_remove_unordered(0);
	is_approx_equals_mat(rmat, rmatb, "rmat->row_remove(0);");

	rmat->row_remove_unordered(1);
	is_approx_equals_mat(rmat, rmatc, "rmat->row_remove_unordered(1);");
}

void MLPPMatrixTests::test_mlpp_matrix_mul() {
	const real_t A[] = {
		1, 2, //
		3, 4, //
		5, 6, //
		7, 8, //
	};

	const real_t B[] = {
		1, 2, 3, 4, //
		5, 6, 7, 8, //
	};

	const real_t C[] = {
		11, 14, 17, 20, //
		23, 30, 37, 44, //
		35, 46, 57, 68, //
		47, 62, 77, 92, //
	};

	Ref<MLPPMatrix> rmata(memnew(MLPPMatrix(A, 4, 2)));
	Ref<MLPPMatrix> rmatb(memnew(MLPPMatrix(B, 2, 4)));
	Ref<MLPPMatrix> rmatc(memnew(MLPPMatrix(C, 4, 4)));

	Ref<MLPPMatrix> rmatr1 = rmata->multn(rmatb);
	is_approx_equals_mat(rmatr1, rmatc, "Ref<MLPPMatrix> rmatr1 = rmata->multn(rmatb);");

	Ref<MLPPMatrix> rmatr2;
	rmatr2.instance();
	rmatr2->multb(rmata, rmatb);
	is_approx_equals_mat(rmatr2, rmatc, "rmatr2->multb(rmata, rmatb);");

	rmata->mult(rmatb);
	is_approx_equals_mat(rmata, rmatc, "rmata->mult(rmatb);");
}

MLPPMatrixTests::MLPPMatrixTests() {
}

MLPPMatrixTests::~MLPPMatrixTests() {
}

void MLPPMatrixTests::_bind_methods() {
	ClassDB::bind_method(D_METHOD("run_tests"), &MLPPMatrixTests::run_tests);

	ClassDB::bind_method(D_METHOD("test_mlpp_matrix"), &MLPPMatrixTests::test_mlpp_matrix);
	ClassDB::bind_method(D_METHOD("test_mlpp_matrix_mul"), &MLPPMatrixTests::test_mlpp_matrix_mul);
}
