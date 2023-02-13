
#ifndef MLPP_LIN_ALG_OLD_H
#define MLPP_LIN_ALG_OLD_H

//
//  LinAlg.hpp
//
//  Created by Marc Melikyan on 1/8/21.
//

//TODO Methods here should probably use error macros in a way where they get disabled in non-tools(?) (maybe release?) builds

#include "core/math/math_defs.h"

#include <tuple>
#include <vector>

class MLPPLinAlgOld {
public:
	// MATRIX FUNCTIONS

	std::vector<std::vector<real_t>> gramMatrix(std::vector<std::vector<real_t>> A);

	bool linearIndependenceChecker(std::vector<std::vector<real_t>> A);

	std::vector<std::vector<real_t>> gaussianNoise(int n, int m);

	std::vector<std::vector<real_t>> addition(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);
	std::vector<std::vector<real_t>> subtraction(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);
	std::vector<std::vector<real_t>> matmult(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);

	std::vector<std::vector<real_t>> hadamard_product(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);
	std::vector<std::vector<real_t>> kronecker_product(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);
	std::vector<std::vector<real_t>> elementWiseDivision(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);

	std::vector<std::vector<real_t>> transpose(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> scalarMultiply(real_t scalar, std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> scalarAdd(real_t scalar, std::vector<std::vector<real_t>> A);

	std::vector<std::vector<real_t>> log(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> log10(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> exp(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> erf(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> exponentiate(std::vector<std::vector<real_t>> A, real_t p);
	std::vector<std::vector<real_t>> sqrt(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> cbrt(std::vector<std::vector<real_t>> A);

	std::vector<std::vector<real_t>> matrixPower(std::vector<std::vector<real_t>> A, int n);

	std::vector<std::vector<real_t>> abs(std::vector<std::vector<real_t>> A);

	real_t det(std::vector<std::vector<real_t>> A, int d);

	real_t trace(std::vector<std::vector<real_t>> A);

	std::vector<std::vector<real_t>> cofactor(std::vector<std::vector<real_t>> A, int n, int i, int j);
	std::vector<std::vector<real_t>> adjoint(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> inverse(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> pinverse(std::vector<std::vector<real_t>> A);

	std::vector<std::vector<real_t>> zeromat(int n, int m);
	std::vector<std::vector<real_t>> onemat(int n, int m);
	std::vector<std::vector<real_t>> full(int n, int m, int k);

	std::vector<std::vector<real_t>> sin(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> cos(std::vector<std::vector<real_t>> A);

	std::vector<std::vector<real_t>> rotate(std::vector<std::vector<real_t>> A, real_t theta, int axis = -1);

	std::vector<std::vector<real_t>> max(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);
	real_t max(std::vector<std::vector<real_t>> A);
	real_t min(std::vector<std::vector<real_t>> A);

	std::vector<std::vector<real_t>> round(std::vector<std::vector<real_t>> A);

	real_t norm_2(std::vector<std::vector<real_t>> A);

	std::vector<std::vector<real_t>> identity(real_t d);

	std::vector<std::vector<real_t>> cov(std::vector<std::vector<real_t>> A);

	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> eig(std::vector<std::vector<real_t>> A);

	struct EigenResultOld {
		std::vector<std::vector<real_t>> eigen_vectors;
		std::vector<std::vector<real_t>> eigen_values;
	};

	EigenResultOld eigen_old(std::vector<std::vector<real_t>> A);

	struct SVDResultOld {
		std::vector<std::vector<real_t>> U;
		std::vector<std::vector<real_t>> S;
		std::vector<std::vector<real_t>> Vt;
	};

	SVDResultOld SVD(std::vector<std::vector<real_t>> A);

	std::vector<real_t> vectorProjection(std::vector<real_t> a, std::vector<real_t> b);

	std::vector<std::vector<real_t>> gramSchmidtProcess(std::vector<std::vector<real_t>> A);

	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> QRD(std::vector<std::vector<real_t>> A);

	struct QRDResult {
		std::vector<std::vector<real_t>> Q;
		std::vector<std::vector<real_t>> R;
	};

	QRDResult qrd(std::vector<std::vector<real_t>> A);

	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> chol(std::vector<std::vector<real_t>> A);

	struct CholeskyResult {
		std::vector<std::vector<real_t>> L;
		std::vector<std::vector<real_t>> Lt;
	};

	CholeskyResult cholesky(std::vector<std::vector<real_t>> A);

	real_t sum_elements(std::vector<std::vector<real_t>> A);

	std::vector<real_t> flatten(std::vector<std::vector<real_t>> A);

	std::vector<real_t> solve(std::vector<std::vector<real_t>> A, std::vector<real_t> b);

	bool positiveDefiniteChecker(std::vector<std::vector<real_t>> A);

	bool negativeDefiniteChecker(std::vector<std::vector<real_t>> A);

	bool zeroEigenvalue(std::vector<std::vector<real_t>> A);

	void printMatrix(std::vector<std::vector<real_t>> A);

	// VECTOR FUNCTIONS

	std::vector<std::vector<real_t>> outerProduct(std::vector<real_t> a, std::vector<real_t> b); // This multiplies a, bT
	std::vector<real_t> hadamard_product(std::vector<real_t> a, std::vector<real_t> b);

	std::vector<real_t> elementWiseDivision(std::vector<real_t> a, std::vector<real_t> b);

	std::vector<real_t> scalarMultiply(real_t scalar, std::vector<real_t> a);

	std::vector<real_t> scalarAdd(real_t scalar, std::vector<real_t> a);

	std::vector<real_t> addition(std::vector<real_t> a, std::vector<real_t> b);

	std::vector<real_t> subtraction(std::vector<real_t> a, std::vector<real_t> b);

	std::vector<real_t> subtractMatrixRows(std::vector<real_t> a, std::vector<std::vector<real_t>> B);

	std::vector<real_t> log(std::vector<real_t> a);
	std::vector<real_t> log10(std::vector<real_t> a);
	std::vector<real_t> exp(std::vector<real_t> a);
	std::vector<real_t> erf(std::vector<real_t> a);
	std::vector<real_t> exponentiate(std::vector<real_t> a, real_t p);
	std::vector<real_t> sqrt(std::vector<real_t> a);
	std::vector<real_t> cbrt(std::vector<real_t> a);

	real_t dot(std::vector<real_t> a, std::vector<real_t> b);

	std::vector<real_t> cross(std::vector<real_t> a, std::vector<real_t> b);

	std::vector<real_t> abs(std::vector<real_t> a);

	std::vector<real_t> zerovec(int n);
	std::vector<real_t> onevec(int n);
	std::vector<real_t> full(int n, int k);

	std::vector<std::vector<real_t>> diag(std::vector<real_t> a);

	std::vector<real_t> sin(std::vector<real_t> a);
	std::vector<real_t> cos(std::vector<real_t> a);

	std::vector<real_t> max(std::vector<real_t> a, std::vector<real_t> b);

	real_t max(std::vector<real_t> a);

	real_t min(std::vector<real_t> a);

	std::vector<real_t> round(std::vector<real_t> a);

	real_t euclideanDistance(std::vector<real_t> a, std::vector<real_t> b);

	real_t norm_2(std::vector<real_t> a);

	real_t norm_sq(std::vector<real_t> a);

	real_t sum_elements(std::vector<real_t> a);

	real_t cosineSimilarity(std::vector<real_t> a, std::vector<real_t> b);

	void printVector(std::vector<real_t> a);

	// MATRIX-VECTOR FUNCTIONS
	std::vector<std::vector<real_t>> mat_vec_add(std::vector<std::vector<real_t>> A, std::vector<real_t> b);
	std::vector<real_t> mat_vec_mult(std::vector<std::vector<real_t>> A, std::vector<real_t> b);

	// TENSOR FUNCTIONS
	std::vector<std::vector<std::vector<real_t>>> addition(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);

	std::vector<std::vector<std::vector<real_t>>> elementWiseDivision(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);

	std::vector<std::vector<std::vector<real_t>>> sqrt(std::vector<std::vector<std::vector<real_t>>> A);

	std::vector<std::vector<std::vector<real_t>>> exponentiate(std::vector<std::vector<std::vector<real_t>>> A, real_t p);

	std::vector<std::vector<real_t>> tensor_vec_mult(std::vector<std::vector<std::vector<real_t>>> A, std::vector<real_t> b);

	std::vector<real_t> flatten(std::vector<std::vector<std::vector<real_t>>> A);

	void printTensor(std::vector<std::vector<std::vector<real_t>>> A);

	std::vector<std::vector<std::vector<real_t>>> scalarMultiply(real_t scalar, std::vector<std::vector<std::vector<real_t>>> A);
	std::vector<std::vector<std::vector<real_t>>> scalarAdd(real_t scalar, std::vector<std::vector<std::vector<real_t>>> A);

	std::vector<std::vector<std::vector<real_t>>> resize(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);

	std::vector<std::vector<std::vector<real_t>>> hadamard_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);

	std::vector<std::vector<std::vector<real_t>>> max(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);

	std::vector<std::vector<std::vector<real_t>>> abs(std::vector<std::vector<std::vector<real_t>>> A);

	real_t norm_2(std::vector<std::vector<std::vector<real_t>>> A);

	std::vector<std::vector<std::vector<real_t>>> vector_wise_tensor_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<real_t>> B);
};

#endif /* LinAlg_hpp */