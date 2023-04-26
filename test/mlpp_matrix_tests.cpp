
#include "mlpp_matrix_tests.h"

#include "core/log/logger.h"

//TODO remove
#include <vector>

#include "../mlpp/lin_alg/mlpp_matrix.h"

void MLPPMatrixTests::run_tests() {
	PLOG_MSG("RUNNIG MLPPMatrixTests!");

	PLOG_MSG("test_mlpp_matrix()");
	test_mlpp_matrix();

	PLOG_MSG("test_add_row()");
	test_add_row();
	PLOG_MSG("test_add_row_pool_vector()");
	test_add_row_pool_vector();
	PLOG_MSG("test_add_row_mlpp_vector()");
	test_add_row_mlpp_vector();
	PLOG_MSG("test_add_rows_mlpp_matrix()");
	test_add_rows_mlpp_matrix();

	PLOG_MSG("test_remove_row()");
	test_remove_row();
	PLOG_MSG("test_remove_row_unordered()");
	test_remove_row_unordered();

	PLOG_MSG("test_mlpp_matrix_mul()");
	test_mlpp_matrix_mul();
}

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

void MLPPMatrixTests::test_add_row() {
	std::vector<std::vector<real_t>> A = {
		{ 1, 2, 3, 4 },
	};

	std::vector<std::vector<real_t>> B = {
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 4 },
	};

	std::vector<std::vector<real_t>> C = {
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 4 },
	};

	Vector<real_t> rv;
	rv.push_back(1);
	rv.push_back(2);
	rv.push_back(3);
	rv.push_back(4);

	Ref<MLPPMatrix> rmata;
	rmata.instance();
	rmata->set_from_std_vectors(A);

	Ref<MLPPMatrix> rmatb;
	rmatb.instance();
	rmatb->set_from_std_vectors(B);

	Ref<MLPPMatrix> rmatc;
	rmatc.instance();
	rmatc->set_from_std_vectors(C);

	Ref<MLPPMatrix> rmat;
	rmat.instance();

	rmat->add_row(rv);
	is_approx_equals_mat(rmata, rmat, "rmat->add_row(rv);");

	rmat->add_row(rv);
	is_approx_equals_mat(rmatb, rmat, "rmat->add_row(rv);");

	rmat->add_row(rv);
	is_approx_equals_mat(rmatc, rmat, "rmat->add_row(rv);");
}
void MLPPMatrixTests::test_add_row_pool_vector() {
	std::vector<std::vector<real_t>> A = {
		{ 1, 2, 3, 4 },
	};

	std::vector<std::vector<real_t>> B = {
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 4 },
	};

	std::vector<std::vector<real_t>> C = {
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 4 },
	};

	PoolVector<real_t> rv;
	rv.push_back(1);
	rv.push_back(2);
	rv.push_back(3);
	rv.push_back(4);

	Ref<MLPPMatrix> rmata;
	rmata.instance();
	rmata->set_from_std_vectors(A);

	Ref<MLPPMatrix> rmatb;
	rmatb.instance();
	rmatb->set_from_std_vectors(B);

	Ref<MLPPMatrix> rmatc;
	rmatc.instance();
	rmatc->set_from_std_vectors(C);

	Ref<MLPPMatrix> rmat;
	rmat.instance();

	rmat->add_row_pool_vector(rv);
	is_approx_equals_mat(rmata, rmat, "rmat->add_row_pool_vector(rv);");

	rmat->add_row_pool_vector(rv);
	is_approx_equals_mat(rmatb, rmat, "rmat->add_row_pool_vector(rv);");

	rmat->add_row_pool_vector(rv);
	is_approx_equals_mat(rmatc, rmat, "rmat->add_row_pool_vector(rv);");
}
void MLPPMatrixTests::test_add_row_mlpp_vector() {
	std::vector<std::vector<real_t>> A = {
		{ 1, 2, 3, 4 },
	};

	std::vector<std::vector<real_t>> B = {
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 4 },
	};

	std::vector<std::vector<real_t>> C = {
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 4 },
	};

	Ref<MLPPVector> rv;
	rv.instance();
	rv->push_back(1);
	rv->push_back(2);
	rv->push_back(3);
	rv->push_back(4);

	Ref<MLPPMatrix> rmata;
	rmata.instance();
	rmata->set_from_std_vectors(A);

	Ref<MLPPMatrix> rmatb;
	rmatb.instance();
	rmatb->set_from_std_vectors(B);

	Ref<MLPPMatrix> rmatc;
	rmatc.instance();
	rmatc->set_from_std_vectors(C);

	Ref<MLPPMatrix> rmat;
	rmat.instance();

	rmat->add_row_mlpp_vector(rv);
	is_approx_equals_mat(rmata, rmat, "rmat->add_row_mlpp_vector(rv);");

	rmat->add_row_mlpp_vector(rv);
	is_approx_equals_mat(rmatb, rmat, "rmat->add_row_mlpp_vector(rv);");

	rmat->add_row_mlpp_vector(rv);
	is_approx_equals_mat(rmatc, rmat, "rmat->add_row_mlpp_vector(rv);");
}
void MLPPMatrixTests::test_add_rows_mlpp_matrix() {
	std::vector<std::vector<real_t>> A = {
		{ 1, 2, 3, 4 },
	};

	std::vector<std::vector<real_t>> B = {
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 4 },
	};

	std::vector<std::vector<real_t>> C = {
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 4 },
		{ 1, 2, 3, 4 },
	};

	std::vector<real_t> r = { 1, 2, 3, 4 };

	PoolVector<real_t> rvp;
	rvp.push_back(1);
	rvp.push_back(2);
	rvp.push_back(3);
	rvp.push_back(4);

	Ref<MLPPMatrix> rv;
	rv.instance();
	rv->add_row_pool_vector(rvp);

	Ref<MLPPMatrix> rmata;
	rmata.instance();
	rmata->set_from_std_vectors(A);

	Ref<MLPPMatrix> rmatb;
	rmatb.instance();
	rmatb->set_from_std_vectors(B);

	Ref<MLPPMatrix> rmatc;
	rmatc.instance();
	rmatc->set_from_std_vectors(C);

	Ref<MLPPMatrix> rmat;
	rmat.instance();

	rmat->add_rows_mlpp_matrix(rv);
	is_approx_equals_mat(rmata, rmat, "rmat->add_rows_mlpp_matrix(rv);");

	rmat->add_rows_mlpp_matrix(rv);
	is_approx_equals_mat(rmatb, rmat, "rmat->add_rows_mlpp_matrix(rv);");

	rmat->add_rows_mlpp_matrix(rv);
	is_approx_equals_mat(rmatc, rmat, "rmat->add_rows_mlpp_matrix(rv);");
}

void MLPPMatrixTests::test_remove_row() {
	std::vector<std::vector<real_t>> A = {
		{ 1, 2, 3, 4 },
		{ 5, 6, 7, 8 },
		{ 9, 10, 11, 12 },
	};

	std::vector<std::vector<real_t>> B = {
		{ 1, 2, 3, 4 },
		{ 5, 6, 7, 8 },
	};

	std::vector<std::vector<real_t>> C = {
		{ 1, 2, 3, 4 },
	};

	std::vector<std::vector<real_t>> D = {
		{ 1, 2, 3, 4 },
		{ 5, 6, 7, 8 },
		{ 13, 14, 15, 16 },
		{ 9, 10, 11, 12 },
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

	Ref<MLPPMatrix> rmat;
	rmat.instance();
	rmat->set_from_std_vectors(D);

	rmat->remove_row(2);
	is_approx_equals_mat(rmat, rmata, "rmat->remove_row(2);");

	rmat->remove_row(2);
	is_approx_equals_mat(rmat, rmatb, "rmat->remove_row(2);");

	rmat->remove_row(1);
	is_approx_equals_mat(rmat, rmatc, "rmat->remove_row(1);");
}
void MLPPMatrixTests::test_remove_row_unordered() {
	std::vector<std::vector<real_t>> A = {
		{ 1, 2, 3, 4 },
		{ 13, 14, 15, 16 },
		{ 9, 10, 11, 12 },
	};

	std::vector<std::vector<real_t>> B = {
		{ 9, 10, 11, 12 },
		{ 13, 14, 15, 16 },
	};

	std::vector<std::vector<real_t>> C = {
		{ 9, 10, 11, 12 },
	};

	std::vector<std::vector<real_t>> D = {
		{ 1, 2, 3, 4 },
		{ 5, 6, 7, 8 },
		{ 9, 10, 11, 12 },
		{ 13, 14, 15, 16 },
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

	Ref<MLPPMatrix> rmat;
	rmat.instance();
	rmat->set_from_std_vectors(D);

	rmat->remove_row_unordered(1);
	is_approx_equals_mat(rmat, rmata, "rmat->remove_row_unordered(1);");

	rmat->remove_row_unordered(0);
	is_approx_equals_mat(rmat, rmatb, "rmat->remove_row(0);");

	rmat->remove_row_unordered(1);
	is_approx_equals_mat(rmat, rmatc, "rmat->remove_row_unordered(1);");
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
	ClassDB::bind_method(D_METHOD("run_tests"), &MLPPMatrixTests::run_tests);

	ClassDB::bind_method(D_METHOD("test_mlpp_matrix"), &MLPPMatrixTests::test_mlpp_matrix);
	ClassDB::bind_method(D_METHOD("test_mlpp_matrix_mul"), &MLPPMatrixTests::test_mlpp_matrix_mul);
}
