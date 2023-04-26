
#include "mlpp_matrix_tests.h"

//TODO remove
#include <vector>

#include "../mlpp/lin_alg/mlpp_matrix.h"

void MLPPMatrixTests::test_mlpp_matrix() {
	std::vector<std::vector<real_t>> A = {
		{ 1, 0, 0, 0 },
		{ 0, 1, 0, 0 },
		{ 0, 0, 1, 0 },
		{ 0, 0, 0, 1 }
	};

	Ref<MLPPMatrix> rmat;
	rmat.instance();
	rmat->set_from_std_vectors(A);

	Ref<MLPPMatrix> rmat2;
	rmat2.instance();
	rmat2->set_from_std_vectors(A);

	is_approx_equals_mat(rmat, rmat2, "set_from_std_vectors test.");

	rmat2->set_from_std_vectors(A);

	is_approx_equals_mat(rmat, rmat2, "re-set_from_std_vectors test.");
}

void MLPPMatrixTests::test_mlpp_matrix_mul() {
	std::vector<std::vector<real_t>> A = {
		{ 1, 2 },
		{ 3, 4 },
		{ 5, 6 },
		{ 7, 8 }
	};

	std::vector<std::vector<real_t>> B = {
		{ 1, 2, 3, 4 },
		{ 5, 6, 7, 8 }
	};

	std::vector<std::vector<real_t>> C = {
		{ 11, 14, 17, 20 },
		{ 23, 30, 37, 44 },
		{ 35, 46, 57, 68 },
		{ 47, 62, 77, 92 }
	};

	Ref<MLPPMatrix> rmata;
	rmata.instance();
	rmata->set_from_std_vectors(A);

	Ref<MLPPMatrix> rmatb;
	rmatb.instance();
	rmatb->set_from_std_vectors(B);

	Ref<MLPPMatrix> rmatc;
	rmatc.instance();
	rmatc->set_from_std_vectors(C);

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
	ClassDB::bind_method(D_METHOD("test_mlpp_matrix"), &MLPPMatrixTests::test_mlpp_matrix);
	ClassDB::bind_method(D_METHOD("test_mlpp_matrix_mul"), &MLPPMatrixTests::test_mlpp_matrix_mul);
}
