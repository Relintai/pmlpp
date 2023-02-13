//
//  LinAlg.cpp
//
//  Created by Marc Melikyan on 1/8/21.
//

#include "lin_alg_old.h"

#include "core/math/math_funcs.h"

#include "../stat/stat.h"

#include <cmath>
#include <iostream>
#include <map>
#include <random>

std::vector<std::vector<real_t>> MLPPLinAlgOld::gramMatrix(std::vector<std::vector<real_t>> A) {
	return matmult(transpose(A), A); // AtA
}

bool MLPPLinAlgOld::linearIndependenceChecker(std::vector<std::vector<real_t>> A) {
	if (det(gramMatrix(A), A.size()) == 0) {
		return false;
	}
	return true;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::gaussianNoise(int n, int m) {
	std::random_device rd;
	std::default_random_engine generator(rd());

	std::vector<std::vector<real_t>> A;
	A.resize(n);
	for (int i = 0; i < n; i++) {
		A[i].resize(m);
		for (int j = 0; j < m; j++) {
			std::normal_distribution<real_t> distribution(0, 1); // Standard normal distribution. Mean of 0, std of 1.
			A[i][j] = distribution(generator);
		}
	}
	return A;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::addition(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[0].size(); j++) {
			C[i][j] = A[i][j] + B[i][j];
		}
	}
	return C;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::subtraction(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[0].size(); j++) {
			C[i][j] = A[i][j] - B[i][j];
		}
	}
	return C;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::matmult(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(B[0].size());
	}

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t k = 0; k < B.size(); k++) {
			for (uint32_t j = 0; j < B[0].size(); j++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	return C;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::hadamard_product(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[0].size(); j++) {
			C[i][j] = A[i][j] * B[i][j];
		}
	}
	return C;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::kronecker_product(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;

	// [1,1,1,1]   [1,2,3,4,5]
	// [1,1,1,1]   [1,2,3,4,5]
	//             [1,2,3,4,5]

	// [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
	// [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
	// [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
	// [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
	// [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]
	// [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5] [1,2,3,4,5]

	// Resulting matrix: A.size() * B.size()
	//                   A[0].size() * B[0].size()

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < B.size(); j++) {
			std::vector<std::vector<real_t>> row;
			for (uint32_t k = 0; k < A[0].size(); k++) {
				row.push_back(scalarMultiply(A[i][k], B[j]));
			}
			C.push_back(flatten(row));
		}
	}
	return C;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::elementWiseDivision(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			C[i][j] = A[i][j] / B[i][j];
		}
	}
	return C;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::transpose(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> AT;
	AT.resize(A[0].size());
	for (uint32_t i = 0; i < AT.size(); i++) {
		AT[i].resize(A.size());
	}

	for (uint32_t i = 0; i < A[0].size(); i++) {
		for (uint32_t j = 0; j < A.size(); j++) {
			AT[i][j] = A[j][i];
		}
	}
	return AT;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::scalarMultiply(real_t scalar, std::vector<std::vector<real_t>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			A[i][j] *= scalar;
		}
	}
	return A;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::scalarAdd(real_t scalar, std::vector<std::vector<real_t>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			A[i][j] += scalar;
		}
	}
	return A;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::log(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::log(A[i][j]);
		}
	}
	return B;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::log10(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::log10(A[i][j]);
		}
	}
	return B;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::exp(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::exp(A[i][j]);
		}
	}
	return B;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::erf(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::erf(A[i][j]);
		}
	}
	return B;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::exponentiate(std::vector<std::vector<real_t>> A, real_t p) {
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			A[i][j] = std::pow(A[i][j], p);
		}
	}
	return A;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::sqrt(std::vector<std::vector<real_t>> A) {
	return exponentiate(A, 0.5);
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::cbrt(std::vector<std::vector<real_t>> A) {
	return exponentiate(A, real_t(1) / real_t(3));
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::abs(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < B.size(); i++) {
		for (uint32_t j = 0; j < B[i].size(); j++) {
			B[i][j] = std::abs(A[i][j]);
		}
	}
	return B;
}

real_t MLPPLinAlgOld::det(std::vector<std::vector<real_t>> A, int d) {
	real_t deter = 0;
	std::vector<std::vector<real_t>> B;
	B.resize(d);
	for (int i = 0; i < d; i++) {
		B[i].resize(d);
	}

	/* This is the base case in which the input is a 2x2 square matrix.
	Recursion is performed unless and until we reach this base case,
	such that we recieve a scalar as the result. */
	if (d == 2) {
		return A[0][0] * A[1][1] - A[0][1] * A[1][0];
	}

	else {
		for (int i = 0; i < d; i++) {
			int sub_i = 0;
			for (int j = 1; j < d; j++) {
				int sub_j = 0;
				for (int k = 0; k < d; k++) {
					if (k == i) {
						continue;
					}
					B[sub_i][sub_j] = A[j][k];
					sub_j++;
				}
				sub_i++;
			}
			deter += std::pow(-1, i) * A[0][i] * det(B, d - 1);
		}
	}
	return deter;
}

real_t MLPPLinAlgOld::trace(std::vector<std::vector<real_t>> A) {
	real_t trace = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		trace += A[i][i];
	}
	return trace;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::cofactor(std::vector<std::vector<real_t>> A, int n, int i, int j) {
	std::vector<std::vector<real_t>> cof;
	cof.resize(A.size());
	for (uint32_t ii = 0; ii < cof.size(); ii++) {
		cof[ii].resize(A.size());
	}
	int sub_i = 0, sub_j = 0;

	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			if (row != i && col != j) {
				cof[sub_i][sub_j++] = A[row][col];

				if (sub_j == n - 1) {
					sub_j = 0;
					sub_i++;
				}
			}
		}
	}
	return cof;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::adjoint(std::vector<std::vector<real_t>> A) {
	//Resizing the initial adjoint matrix
	std::vector<std::vector<real_t>> adj;
	adj.resize(A.size());
	for (uint32_t i = 0; i < adj.size(); i++) {
		adj[i].resize(A.size());
	}

	// Checking for the case where the given N x N matrix is a scalar
	if (A.size() == 1) {
		adj[0][0] = 1;
		return adj;
	}

	if (A.size() == 2) {
		adj[0][0] = A[1][1];
		adj[1][1] = A[0][0];

		adj[0][1] = -A[0][1];
		adj[1][0] = -A[1][0];
		return adj;
	}

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A.size(); j++) {
			std::vector<std::vector<real_t>> cof = cofactor(A, int(A.size()), i, j);
			// 1 if even, -1 if odd
			int sign = (i + j) % 2 == 0 ? 1 : -1;
			adj[j][i] = sign * det(cof, int(A.size()) - 1);
		}
	}
	return adj;
}

// The inverse can be computed as (1 / determinant(A)) * adjoint(A)
std::vector<std::vector<real_t>> MLPPLinAlgOld::inverse(std::vector<std::vector<real_t>> A) {
	return scalarMultiply(1 / det(A, int(A.size())), adjoint(A));
}

// This is simply the Moore-Penrose least squares approximation of the inverse.
std::vector<std::vector<real_t>> MLPPLinAlgOld::pinverse(std::vector<std::vector<real_t>> A) {
	return matmult(inverse(matmult(transpose(A), A)), transpose(A));
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::zeromat(int n, int m) {
	std::vector<std::vector<real_t>> zeromat;
	zeromat.resize(n);
	for (uint32_t i = 0; i < zeromat.size(); i++) {
		zeromat[i].resize(m);
	}
	return zeromat;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::onemat(int n, int m) {
	return full(n, m, 1);
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::full(int n, int m, int k) {
	std::vector<std::vector<real_t>> full;
	full.resize(n);
	for (uint32_t i = 0; i < full.size(); i++) {
		full[i].resize(m);
	}
	for (uint32_t i = 0; i < full.size(); i++) {
		for (uint32_t j = 0; j < full[i].size(); j++) {
			full[i][j] = k;
		}
	}
	return full;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::sin(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::sin(A[i][j]);
		}
	}
	return B;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::cos(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::cos(A[i][j]);
		}
	}
	return B;
}

std::vector<real_t> MLPPLinAlgOld::max(std::vector<real_t> a, std::vector<real_t> b) {
	std::vector<real_t> c;
	c.resize(a.size());
	for (uint32_t i = 0; i < c.size(); i++) {
		if (a[i] >= b[i]) {
			c[i] = a[i];
		} else {
			c[i] = b[i];
		}
	}
	return c;
}

real_t MLPPLinAlgOld::max(std::vector<std::vector<real_t>> A) {
	return max(flatten(A));
}

real_t MLPPLinAlgOld::min(std::vector<std::vector<real_t>> A) {
	return min(flatten(A));
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::round(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			B[i][j] = std::round(A[i][j]);
		}
	}
	return B;
}

real_t MLPPLinAlgOld::norm_2(std::vector<std::vector<real_t>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			sum += A[i][j] * A[i][j];
		}
	}
	return std::sqrt(sum);
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::identity(real_t d) {
	std::vector<std::vector<real_t>> identityMat;
	identityMat.resize(d);
	for (uint32_t i = 0; i < identityMat.size(); i++) {
		identityMat[i].resize(d);
	}
	for (uint32_t i = 0; i < identityMat.size(); i++) {
		for (uint32_t j = 0; j < identityMat.size(); j++) {
			if (i == j) {
				identityMat[i][j] = 1;
			} else {
				identityMat[i][j] = 0;
			}
		}
	}
	return identityMat;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::cov(std::vector<std::vector<real_t>> A) {
	MLPPStat stat;
	std::vector<std::vector<real_t>> covMat;
	covMat.resize(A.size());
	for (uint32_t i = 0; i < covMat.size(); i++) {
		covMat[i].resize(A.size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A.size(); j++) {
			covMat[i][j] = stat.covariance(A[i], A[j]);
		}
	}
	return covMat;
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPLinAlgOld::eig(std::vector<std::vector<real_t>> A) {
	/*
	A (the entered parameter) in most use cases will be X'X, XX', etc. and must be symmetric.
	That simply means that 1) X' = X and 2) X is a square matrix. This function that computes the
	eigenvalues of a matrix is utilizing Jacobi's method.
	*/

	real_t diagonal = true; // Perform the iterative Jacobi algorithm unless and until we reach a diagonal matrix which yields us the eigenvals.

	std::map<int, int> val_to_vec;
	std::vector<std::vector<real_t>> a_new;
	std::vector<std::vector<real_t>> eigenvectors = identity(A.size());
	do {
		real_t a_ij = A[0][1];
		real_t sub_i = 0;
		real_t sub_j = 1;
		for (uint32_t i = 0; i < A.size(); i++) {
			for (uint32_t j = 0; j < A[i].size(); j++) {
				if (i != j && std::abs(A[i][j]) > a_ij) {
					a_ij = A[i][j];
					sub_i = i;
					sub_j = j;
				} else if (i != j && std::abs(A[i][j]) == a_ij) {
					if (i < sub_i) {
						a_ij = A[i][j];
						sub_i = i;
						sub_j = j;
					}
				}
			}
		}

		real_t a_ii = A[sub_i][sub_i];
		real_t a_jj = A[sub_j][sub_j];
		//real_t a_ji = A[sub_j][sub_i];
		real_t theta;

		if (a_ii == a_jj) {
			theta = M_PI / 4;
		} else {
			theta = 0.5 * atan(2 * a_ij / (a_ii - a_jj));
		}

		std::vector<std::vector<real_t>> P = identity(A.size());
		P[sub_i][sub_j] = -std::sin(theta);
		P[sub_i][sub_i] = std::cos(theta);
		P[sub_j][sub_j] = std::cos(theta);
		P[sub_j][sub_i] = std::sin(theta);

		a_new = matmult(matmult(inverse(P), A), P);

		for (uint32_t i = 0; i < a_new.size(); i++) {
			for (uint32_t j = 0; j < a_new[i].size(); j++) {
				if (i != j && std::round(a_new[i][j]) == 0) {
					a_new[i][j] = 0;
				}
			}
		}

		bool non_zero = false;
		for (uint32_t i = 0; i < a_new.size(); i++) {
			for (uint32_t j = 0; j < a_new[i].size(); j++) {
				if (i != j && std::round(a_new[i][j]) != 0) {
					non_zero = true;
				}
			}
		}

		if (non_zero) {
			diagonal = false;
		} else {
			diagonal = true;
		}

		if (a_new == A) {
			diagonal = true;
			for (uint32_t i = 0; i < a_new.size(); i++) {
				for (uint32_t j = 0; j < a_new[i].size(); j++) {
					if (i != j) {
						a_new[i][j] = 0;
					}
				}
			}
		}

		eigenvectors = matmult(eigenvectors, P);
		A = a_new;

	} while (!diagonal);

	std::vector<std::vector<real_t>> a_new_prior = a_new;

	// Bubble Sort. Should change this later.
	for (uint32_t i = 0; i < a_new.size() - 1; i++) {
		for (uint32_t j = 0; j < a_new.size() - 1 - i; j++) {
			if (a_new[j][j] < a_new[j + 1][j + 1]) {
				real_t temp = a_new[j + 1][j + 1];
				a_new[j + 1][j + 1] = a_new[j][j];
				a_new[j][j] = temp;
			}
		}
	}

	for (uint32_t i = 0; i < a_new.size(); i++) {
		for (uint32_t j = 0; j < a_new.size(); j++) {
			if (a_new[i][i] == a_new_prior[j][j]) {
				val_to_vec[i] = j;
			}
		}
	}

	std::vector<std::vector<real_t>> eigen_temp = eigenvectors;
	for (uint32_t i = 0; i < eigenvectors.size(); i++) {
		for (uint32_t j = 0; j < eigenvectors[i].size(); j++) {
			eigenvectors[i][j] = eigen_temp[i][val_to_vec[j]];
		}
	}
	return { eigenvectors, a_new };
}

MLPPLinAlgOld::EigenResultOld MLPPLinAlgOld::eigen_old(std::vector<std::vector<real_t>> A) {
	/*
	A (the entered parameter) in most use cases will be X'X, XX', etc. and must be symmetric.
	That simply means that 1) X' = X and 2) X is a square matrix. This function that computes the
	eigenvalues of a matrix is utilizing Jacobi's method.
	*/

	real_t diagonal = true; // Perform the iterative Jacobi algorithm unless and until we reach a diagonal matrix which yields us the eigenvals.

	std::map<int, int> val_to_vec;
	std::vector<std::vector<real_t>> a_new;
	std::vector<std::vector<real_t>> eigenvectors = identity(A.size());
	do {
		real_t a_ij = A[0][1];
		real_t sub_i = 0;
		real_t sub_j = 1;
		for (uint32_t i = 0; i < A.size(); i++) {
			for (uint32_t j = 0; j < A[i].size(); j++) {
				if (i != j && std::abs(A[i][j]) > a_ij) {
					a_ij = A[i][j];
					sub_i = i;
					sub_j = j;
				} else if (i != j && std::abs(A[i][j]) == a_ij) {
					if (i < sub_i) {
						a_ij = A[i][j];
						sub_i = i;
						sub_j = j;
					}
				}
			}
		}

		real_t a_ii = A[sub_i][sub_i];
		real_t a_jj = A[sub_j][sub_j];
		//real_t a_ji = A[sub_j][sub_i];
		real_t theta;

		if (a_ii == a_jj) {
			theta = M_PI / 4;
		} else {
			theta = 0.5 * atan(2 * a_ij / (a_ii - a_jj));
		}

		std::vector<std::vector<real_t>> P = identity(A.size());
		P[sub_i][sub_j] = -std::sin(theta);
		P[sub_i][sub_i] = std::cos(theta);
		P[sub_j][sub_j] = std::cos(theta);
		P[sub_j][sub_i] = std::sin(theta);

		a_new = matmult(matmult(inverse(P), A), P);

		for (uint32_t i = 0; i < a_new.size(); i++) {
			for (uint32_t j = 0; j < a_new[i].size(); j++) {
				if (i != j && std::round(a_new[i][j]) == 0) {
					a_new[i][j] = 0;
				}
			}
		}

		bool non_zero = false;
		for (uint32_t i = 0; i < a_new.size(); i++) {
			for (uint32_t j = 0; j < a_new[i].size(); j++) {
				if (i != j && std::round(a_new[i][j]) != 0) {
					non_zero = true;
				}
			}
		}

		if (non_zero) {
			diagonal = false;
		} else {
			diagonal = true;
		}

		if (a_new == A) {
			diagonal = true;
			for (uint32_t i = 0; i < a_new.size(); i++) {
				for (uint32_t j = 0; j < a_new[i].size(); j++) {
					if (i != j) {
						a_new[i][j] = 0;
					}
				}
			}
		}

		eigenvectors = matmult(eigenvectors, P);
		A = a_new;

	} while (!diagonal);

	std::vector<std::vector<real_t>> a_new_prior = a_new;

	// Bubble Sort. Should change this later.
	for (uint32_t i = 0; i < a_new.size() - 1; i++) {
		for (uint32_t j = 0; j < a_new.size() - 1 - i; j++) {
			if (a_new[j][j] < a_new[j + 1][j + 1]) {
				real_t temp = a_new[j + 1][j + 1];
				a_new[j + 1][j + 1] = a_new[j][j];
				a_new[j][j] = temp;
			}
		}
	}

	for (uint32_t i = 0; i < a_new.size(); i++) {
		for (uint32_t j = 0; j < a_new.size(); j++) {
			if (a_new[i][i] == a_new_prior[j][j]) {
				val_to_vec[i] = j;
			}
		}
	}

	std::vector<std::vector<real_t>> eigen_temp = eigenvectors;
	for (uint32_t i = 0; i < eigenvectors.size(); i++) {
		for (uint32_t j = 0; j < eigenvectors[i].size(); j++) {
			eigenvectors[i][j] = eigen_temp[i][val_to_vec[j]];
		}
	}

	EigenResultOld res;
	res.eigen_vectors = eigenvectors;
	res.eigen_values = a_new;

	return res;
}

MLPPLinAlgOld::SVDResultOld MLPPLinAlgOld::SVD(std::vector<std::vector<real_t>> A) {
	EigenResultOld left_eigen = eigen_old(matmult(A, transpose(A)));
	EigenResultOld right_eigen = eigen_old(matmult(transpose(A), A));

	std::vector<std::vector<real_t>> singularvals = sqrt(left_eigen.eigen_values);
	std::vector<std::vector<real_t>> sigma = zeromat(A.size(), A[0].size());
	for (uint32_t i = 0; i < singularvals.size(); i++) {
		for (uint32_t j = 0; j < singularvals[i].size(); j++) {
			sigma[i][j] = singularvals[i][j];
		}
	}

	SVDResultOld res;
	res.U = left_eigen.eigen_vectors;
	res.S = sigma;
	res.Vt = right_eigen.eigen_vectors;

	return res;
}

std::vector<real_t> MLPPLinAlgOld::vectorProjection(std::vector<real_t> a, std::vector<real_t> b) {
	real_t product = dot(a, b) / dot(a, a);
	return scalarMultiply(product, a); // Projection of vector a onto b. Denotated as proj_a(b).
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::gramSchmidtProcess(std::vector<std::vector<real_t>> A) {
	A = transpose(A); // C++ vectors lack a mechanism to directly index columns. So, we transpose *a copy* of A for this purpose for ease of use.
	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[0].size());
	}

	B[0] = A[0]; // We set a_1 = b_1 as an initial condition.
	B[0] = scalarMultiply(1 / norm_2(B[0]), B[0]);
	for (uint32_t i = 1; i < B.size(); i++) {
		B[i] = A[i];
		for (int j = i - 1; j >= 0; j--) {
			B[i] = subtraction(B[i], vectorProjection(B[j], A[i]));
		}
		B[i] = scalarMultiply(1 / norm_2(B[i]), B[i]); // Very simply multiply all elements of vec B[i] by 1/||B[i]||_2
	}
	return transpose(B); // We re-transpose the marix.
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPLinAlgOld::QRD(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> Q = gramSchmidtProcess(A);
	std::vector<std::vector<real_t>> R = matmult(transpose(Q), A);
	return { Q, R };
}

MLPPLinAlgOld::QRDResult MLPPLinAlgOld::qrd(std::vector<std::vector<real_t>> A) {
	QRDResult res;

	res.Q = gramSchmidtProcess(A);
	res.R = matmult(transpose(res.Q), A);

	return res;
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPLinAlgOld::chol(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> L = zeromat(A.size(), A[0].size());
	for (uint32_t j = 0; j < L.size(); j++) { // Matrices entered must be square. No problem here.
		for (uint32_t i = j; i < L.size(); i++) {
			if (i == j) {
				real_t sum = 0;
				for (uint32_t k = 0; k < j; k++) {
					sum += L[i][k] * L[i][k];
				}
				L[i][j] = std::sqrt(A[i][j] - sum);
			} else { // That is, i!=j
				real_t sum = 0;
				for (uint32_t k = 0; k < j; k++) {
					sum += L[i][k] * L[j][k];
				}
				L[i][j] = (A[i][j] - sum) / L[j][j];
			}
		}
	}
	return { L, transpose(L) }; // Indeed, L.T is our upper triangular matrix.
}

MLPPLinAlgOld::CholeskyResult MLPPLinAlgOld::cholesky(std::vector<std::vector<real_t>> A) {
	std::vector<std::vector<real_t>> L = zeromat(A.size(), A[0].size());
	for (uint32_t j = 0; j < L.size(); j++) { // Matrices entered must be square. No problem here.
		for (uint32_t i = j; i < L.size(); i++) {
			if (i == j) {
				real_t sum = 0;
				for (uint32_t k = 0; k < j; k++) {
					sum += L[i][k] * L[i][k];
				}
				L[i][j] = std::sqrt(A[i][j] - sum);
			} else { // That is, i!=j
				real_t sum = 0;
				for (uint32_t k = 0; k < j; k++) {
					sum += L[i][k] * L[j][k];
				}
				L[i][j] = (A[i][j] - sum) / L[j][j];
			}
		}
	}

	CholeskyResult res;
	res.L = L;
	res.Lt = transpose(L); // Indeed, L.T is our upper triangular matrix.

	return res;
}

real_t MLPPLinAlgOld::sum_elements(std::vector<std::vector<real_t>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			sum += A[i][j];
		}
	}
	return sum;
}

std::vector<real_t> MLPPLinAlgOld::flatten(std::vector<std::vector<real_t>> A) {
	std::vector<real_t> a;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			a.push_back(A[i][j]);
		}
	}
	return a;
}

std::vector<real_t> MLPPLinAlgOld::solve(std::vector<std::vector<real_t>> A, std::vector<real_t> b) {
	return mat_vec_mult(inverse(A), b);
}

bool MLPPLinAlgOld::positiveDefiniteChecker(std::vector<std::vector<real_t>> A) {
	auto eig_result = eig(A);
	auto eigenvectors = std::get<0>(eig_result);
	auto eigenvals = std::get<1>(eig_result);

	std::vector<real_t> eigenvals_vec;
	for (uint32_t i = 0; i < eigenvals.size(); i++) {
		eigenvals_vec.push_back(eigenvals[i][i]);
	}
	for (uint32_t i = 0; i < eigenvals_vec.size(); i++) {
		if (eigenvals_vec[i] <= 0) { // Simply check to ensure all eigenvalues are positive.
			return false;
		}
	}
	return true;
}

bool MLPPLinAlgOld::negativeDefiniteChecker(std::vector<std::vector<real_t>> A) {
	auto eig_result = eig(A);
	auto eigenvectors = std::get<0>(eig_result);
	auto eigenvals = std::get<1>(eig_result);

	std::vector<real_t> eigenvals_vec;
	for (uint32_t i = 0; i < eigenvals.size(); i++) {
		eigenvals_vec.push_back(eigenvals[i][i]);
	}
	for (uint32_t i = 0; i < eigenvals_vec.size(); i++) {
		if (eigenvals_vec[i] >= 0) { // Simply check to ensure all eigenvalues are negative.
			return false;
		}
	}
	return true;
}

bool MLPPLinAlgOld::zeroEigenvalue(std::vector<std::vector<real_t>> A) {
	auto eig_result = eig(A);
	auto eigenvectors = std::get<0>(eig_result);
	auto eigenvals = std::get<1>(eig_result);

	std::vector<real_t> eigenvals_vec;
	for (uint32_t i = 0; i < eigenvals.size(); i++) {
		eigenvals_vec.push_back(eigenvals[i][i]);
	}
	for (uint32_t i = 0; i < eigenvals_vec.size(); i++) {
		if (eigenvals_vec[i] == 0) {
			return true;
		}
	}
	return false;
}

void MLPPLinAlgOld::printMatrix(std::vector<std::vector<real_t>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			std::cout << A[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::outerProduct(std::vector<real_t> a, std::vector<real_t> b) {
	std::vector<std::vector<real_t>> C;
	C.resize(a.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i] = scalarMultiply(a[i], b);
	}
	return C;
}

std::vector<real_t> MLPPLinAlgOld::hadamard_product(std::vector<real_t> a, std::vector<real_t> b) {
	std::vector<real_t> c;
	c.resize(a.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		c[i] = a[i] * b[i];
	}

	return c;
}

std::vector<real_t> MLPPLinAlgOld::elementWiseDivision(std::vector<real_t> a, std::vector<real_t> b) {
	std::vector<real_t> c;
	c.resize(a.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		c[i] = a[i] / b[i];
	}
	return c;
}

std::vector<real_t> MLPPLinAlgOld::scalarMultiply(real_t scalar, std::vector<real_t> a) {
	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] *= scalar;
	}
	return a;
}

std::vector<real_t> MLPPLinAlgOld::scalarAdd(real_t scalar, std::vector<real_t> a) {
	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] += scalar;
	}
	return a;
}

std::vector<real_t> MLPPLinAlgOld::addition(std::vector<real_t> a, std::vector<real_t> b) {
	std::vector<real_t> c;
	c.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		c[i] = a[i] + b[i];
	}
	return c;
}

std::vector<real_t> MLPPLinAlgOld::subtraction(std::vector<real_t> a, std::vector<real_t> b) {
	std::vector<real_t> c;
	c.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		c[i] = a[i] - b[i];
	}
	return c;
}

std::vector<real_t> MLPPLinAlgOld::subtractMatrixRows(std::vector<real_t> a, std::vector<std::vector<real_t>> B) {
	for (uint32_t i = 0; i < B.size(); i++) {
		a = subtraction(a, B[i]);
	}
	return a;
}

std::vector<real_t> MLPPLinAlgOld::log(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::log(a[i]);
	}
	return b;
}

std::vector<real_t> MLPPLinAlgOld::log10(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::log10(a[i]);
	}
	return b;
}

std::vector<real_t> MLPPLinAlgOld::exp(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::exp(a[i]);
	}
	return b;
}

std::vector<real_t> MLPPLinAlgOld::erf(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::erf(a[i]);
	}
	return b;
}

std::vector<real_t> MLPPLinAlgOld::exponentiate(std::vector<real_t> a, real_t p) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < b.size(); i++) {
		b[i] = std::pow(a[i], p);
	}
	return b;
}

std::vector<real_t> MLPPLinAlgOld::sqrt(std::vector<real_t> a) {
	return exponentiate(a, 0.5);
}

std::vector<real_t> MLPPLinAlgOld::cbrt(std::vector<real_t> a) {
	return exponentiate(a, real_t(1) / real_t(3));
}

std::vector<real_t> MLPPLinAlgOld::cross(std::vector<real_t> a, std::vector<real_t> b) {
	// Cross products exist in R^7 also. Though, I will limit it to R^3 as Wolfram does this.
	std::vector<std::vector<real_t>> mat = { onevec(3), a, b };

	real_t det1 = det({ { a[1], a[2] }, { b[1], b[2] } }, 2);
	real_t det2 = -det({ { a[0], a[2] }, { b[0], b[2] } }, 2);
	real_t det3 = det({ { a[0], a[1] }, { b[0], b[1] } }, 2);

	return { det1, det2, det3 };
}

std::vector<real_t> MLPPLinAlgOld::abs(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < b.size(); i++) {
		b[i] = std::abs(a[i]);
	}
	return b;
}

std::vector<real_t> MLPPLinAlgOld::zerovec(int n) {
	std::vector<real_t> zerovec;
	zerovec.resize(n);
	return zerovec;
}

std::vector<real_t> MLPPLinAlgOld::onevec(int n) {
	return full(n, 1);
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::diag(std::vector<real_t> a) {
	std::vector<std::vector<real_t>> B = zeromat(a.size(), a.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i][i] = a[i];
	}
	return B;
}

std::vector<real_t> MLPPLinAlgOld::full(int n, int k) {
	std::vector<real_t> full;
	full.resize(n);
	for (uint32_t i = 0; i < full.size(); i++) {
		full[i] = k;
	}
	return full;
}

std::vector<real_t> MLPPLinAlgOld::sin(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::sin(a[i]);
	}
	return b;
}

std::vector<real_t> MLPPLinAlgOld::cos(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::cos(a[i]);
	}
	return b;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::rotate(std::vector<std::vector<real_t>> A, real_t theta, int axis) {
	std::vector<std::vector<real_t>> rotationMatrix = { { std::cos(theta), -std::sin(theta) }, { std::sin(theta), std::cos(theta) } };
	if (axis == 0) {
		rotationMatrix = { { 1, 0, 0 }, { 0, std::cos(theta), -std::sin(theta) }, { 0, std::sin(theta), std::cos(theta) } };
	} else if (axis == 1) {
		rotationMatrix = { { std::cos(theta), 0, std::sin(theta) }, { 0, 1, 0 }, { -std::sin(theta), 0, std::cos(theta) } };
	} else if (axis == 2) {
		rotationMatrix = { { std::cos(theta), -std::sin(theta), 0 }, { std::sin(theta), std::cos(theta), 0 }, { 1, 0, 0 } };
	}

	return matmult(A, rotationMatrix);
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::max(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < A.size(); i++) {
		C[i] = max(A[i], B[i]);
	}
	return C;
}

real_t MLPPLinAlgOld::max(std::vector<real_t> a) {
	int max = a[0];
	for (uint32_t i = 0; i < a.size(); i++) {
		if (a[i] > max) {
			max = a[i];
		}
	}
	return max;
}

real_t MLPPLinAlgOld::min(std::vector<real_t> a) {
	int min = a[0];
	for (uint32_t i = 0; i < a.size(); i++) {
		if (a[i] < min) {
			min = a[i];
		}
	}
	return min;
}

std::vector<real_t> MLPPLinAlgOld::round(std::vector<real_t> a) {
	std::vector<real_t> b;
	b.resize(a.size());
	for (uint32_t i = 0; i < a.size(); i++) {
		b[i] = std::round(a[i]);
	}
	return b;
}

// Multidimensional Euclidean Distance
real_t MLPPLinAlgOld::euclideanDistance(std::vector<real_t> a, std::vector<real_t> b) {
	real_t dist = 0;
	for (uint32_t i = 0; i < a.size(); i++) {
		dist += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return std::sqrt(dist);
}

real_t MLPPLinAlgOld::norm_2(std::vector<real_t> a) {
	return std::sqrt(norm_sq(a));
}

real_t MLPPLinAlgOld::norm_sq(std::vector<real_t> a) {
	real_t n_sq = 0;
	for (uint32_t i = 0; i < a.size(); i++) {
		n_sq += a[i] * a[i];
	}
	return n_sq;
}

real_t MLPPLinAlgOld::sum_elements(std::vector<real_t> a) {
	real_t sum = 0;
	for (uint32_t i = 0; i < a.size(); i++) {
		sum += a[i];
	}
	return sum;
}

real_t MLPPLinAlgOld::cosineSimilarity(std::vector<real_t> a, std::vector<real_t> b) {
	return dot(a, b) / (norm_2(a) * norm_2(b));
}

void MLPPLinAlgOld::printVector(std::vector<real_t> a) {
	for (uint32_t i = 0; i < a.size(); i++) {
		std::cout << a[i] << " ";
	}
	std::cout << std::endl;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::mat_vec_add(std::vector<std::vector<real_t>> A, std::vector<real_t> b) {
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			A[i][j] += b[j];
		}
	}
	return A;
}

std::vector<real_t> MLPPLinAlgOld::mat_vec_mult(std::vector<std::vector<real_t>> A, std::vector<real_t> b) {
	std::vector<real_t> c;
	c.resize(A.size());

	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t k = 0; k < b.size(); k++) {
			c[i] += A[i][k] * b[k];
		}
	}
	return c;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlgOld::addition(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = addition(A[i], B[i]);
	}
	return A;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlgOld::elementWiseDivision(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = elementWiseDivision(A[i], B[i]);
	}
	return A;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlgOld::sqrt(std::vector<std::vector<std::vector<real_t>>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = sqrt(A[i]);
	}
	return A;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlgOld::exponentiate(std::vector<std::vector<std::vector<real_t>>> A, real_t p) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = exponentiate(A[i], p);
	}
	return A;
}

std::vector<std::vector<real_t>> MLPPLinAlgOld::tensor_vec_mult(std::vector<std::vector<std::vector<real_t>>> A, std::vector<real_t> b) {
	std::vector<std::vector<real_t>> C;
	C.resize(A.size());
	for (uint32_t i = 0; i < C.size(); i++) {
		C[i].resize(A[0].size());
	}
	for (uint32_t i = 0; i < C.size(); i++) {
		for (uint32_t j = 0; j < C[i].size(); j++) {
			C[i][j] = dot(A[i][j], b);
		}
	}
	return C;
}

std::vector<real_t> MLPPLinAlgOld::flatten(std::vector<std::vector<std::vector<real_t>>> A) {
	std::vector<real_t> c;
	for (uint32_t i = 0; i < A.size(); i++) {
		std::vector<real_t> flattenedVec = flatten(A[i]);
		c.insert(c.end(), flattenedVec.begin(), flattenedVec.end());
	}
	return c;
}

void MLPPLinAlgOld::printTensor(std::vector<std::vector<std::vector<real_t>>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		printMatrix(A[i]);
		if (i != A.size() - 1) {
			std::cout << std::endl;
		}
	}
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlgOld::scalarMultiply(real_t scalar, std::vector<std::vector<std::vector<real_t>>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = scalarMultiply(scalar, A[i]);
	}
	return A;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlgOld::scalarAdd(real_t scalar, std::vector<std::vector<std::vector<real_t>>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = scalarAdd(scalar, A[i]);
	}
	return A;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlgOld::resize(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B) {
	A.resize(B.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		A[i].resize(B[i].size());
		for (uint32_t j = 0; j < B[i].size(); j++) {
			A[i][j].resize(B[i][j].size());
		}
	}
	return A;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlgOld::max(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = max(A[i], B[i]);
	}
	return A;
}

std::vector<std::vector<std::vector<real_t>>> MLPPLinAlgOld::abs(std::vector<std::vector<std::vector<real_t>>> A) {
	for (uint32_t i = 0; i < A.size(); i++) {
		A[i] = abs(A[i]);
	}
	return A;
}

real_t MLPPLinAlgOld::norm_2(std::vector<std::vector<std::vector<real_t>>> A) {
	real_t sum = 0;
	for (uint32_t i = 0; i < A.size(); i++) {
		for (uint32_t j = 0; j < A[i].size(); j++) {
			for (uint32_t k = 0; k < A[i][j].size(); k++) {
				sum += A[i][j][k] * A[i][j][k];
			}
		}
	}
	return std::sqrt(sum);
}

// Bad implementation. Change this later.
std::vector<std::vector<std::vector<real_t>>> MLPPLinAlgOld::vector_wise_tensor_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<real_t>> B) {
	std::vector<std::vector<std::vector<real_t>>> C;
	C = resize(C, A);
	for (uint32_t i = 0; i < A[0].size(); i++) {
		for (uint32_t j = 0; j < A[0][i].size(); j++) {
			std::vector<real_t> currentVector;
			currentVector.resize(A.size());

			for (uint32_t k = 0; k < C.size(); k++) {
				currentVector[k] = A[k][i][j];
			}

			currentVector = mat_vec_mult(B, currentVector);

			for (uint32_t k = 0; k < C.size(); k++) {
				C[k][i][j] = currentVector[k];
			}
		}
	}
	return C;
}
