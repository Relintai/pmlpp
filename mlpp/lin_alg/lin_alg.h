
#ifndef MLPP_LIN_ALG_H
#define MLPP_LIN_ALG_H

//
//  LinAlg.hpp
//
//  Created by Marc Melikyan on 1/8/21.
//

//TODO Methods here should probably use error macros in a way where they get disabled in non-tools(?) (maybe release?) builds

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <tuple>
#include <vector>

class MLPPLinAlg : public Reference {
	GDCLASS(MLPPLinAlg, Reference);

public:
	// MATRIX FUNCTIONS

	std::vector<std::vector<real_t>> gramMatrix(std::vector<std::vector<real_t>> A);

	bool linearIndependenceChecker(std::vector<std::vector<real_t>> A);

	std::vector<std::vector<real_t>> gaussianNoise(int n, int m);
	Ref<MLPPMatrix> gaussian_noise(int n, int m);

	std::vector<std::vector<real_t>> addition(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);
	std::vector<std::vector<real_t>> subtraction(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);
	std::vector<std::vector<real_t>> matmult(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);

	Ref<MLPPMatrix> additionm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> subtractionm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> matmultm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	std::vector<std::vector<real_t>> hadamard_product(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);
	std::vector<std::vector<real_t>> kronecker_product(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);
	std::vector<std::vector<real_t>> elementWiseDivision(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);

	Ref<MLPPMatrix> hadamard_productm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> kronecker_productm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> element_wise_divisionm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	std::vector<std::vector<real_t>> transpose(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> scalarMultiply(real_t scalar, std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> scalarAdd(real_t scalar, std::vector<std::vector<real_t>> A);

	Ref<MLPPMatrix> transposem(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> scalar_multiplym(real_t scalar, const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> scalar_addm(real_t scalar, const Ref<MLPPMatrix> &A);

	std::vector<std::vector<real_t>> log(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> log10(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> exp(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> erf(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> exponentiate(std::vector<std::vector<real_t>> A, real_t p);
	std::vector<std::vector<real_t>> sqrt(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> cbrt(std::vector<std::vector<real_t>> A);

	Ref<MLPPMatrix> logm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> log10m(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> expm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> erfm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> exponentiatem(const Ref<MLPPMatrix> &A, real_t p);
	Ref<MLPPMatrix> sqrtm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> cbrtm(const Ref<MLPPMatrix> &A);

	std::vector<std::vector<real_t>> matrixPower(std::vector<std::vector<real_t>> A, int n);

	std::vector<std::vector<real_t>> abs(std::vector<std::vector<real_t>> A);

	Ref<MLPPMatrix> absm(const Ref<MLPPMatrix> &A);

	real_t det(std::vector<std::vector<real_t>> A, int d);
	real_t detm(const Ref<MLPPMatrix> &A, int d);

	real_t trace(std::vector<std::vector<real_t>> A);

	std::vector<std::vector<real_t>> cofactor(std::vector<std::vector<real_t>> A, int n, int i, int j);
	std::vector<std::vector<real_t>> adjoint(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> inverse(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> pinverse(std::vector<std::vector<real_t>> A);

	Ref<MLPPMatrix> cofactorm(const Ref<MLPPMatrix> &A, int n, int i, int j);
	Ref<MLPPMatrix> adjointm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> inversem(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> pinversem(const Ref<MLPPMatrix> &A);

	std::vector<std::vector<real_t>> zeromat(int n, int m);
	std::vector<std::vector<real_t>> onemat(int n, int m);
	std::vector<std::vector<real_t>> full(int n, int m, int k);

	Ref<MLPPMatrix> zeromatm(int n, int m);
	Ref<MLPPMatrix> onematm(int n, int m);
	Ref<MLPPMatrix> fullm(int n, int m, int k);

	std::vector<std::vector<real_t>> sin(std::vector<std::vector<real_t>> A);
	std::vector<std::vector<real_t>> cos(std::vector<std::vector<real_t>> A);

	Ref<MLPPMatrix> sinm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> cosm(const Ref<MLPPMatrix> &A);

	std::vector<std::vector<real_t>> rotate(std::vector<std::vector<real_t>> A, real_t theta, int axis = -1);

	std::vector<std::vector<real_t>> max(std::vector<std::vector<real_t>> A, std::vector<std::vector<real_t>> B);
	Ref<MLPPMatrix> max_nm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	real_t max(std::vector<std::vector<real_t>> A);
	real_t min(std::vector<std::vector<real_t>> A);

	std::vector<std::vector<real_t>> round(std::vector<std::vector<real_t>> A);

	real_t norm_2(std::vector<std::vector<real_t>> A);

	std::vector<std::vector<real_t>> identity(real_t d);
	Ref<MLPPMatrix> identitym(int d);

	std::vector<std::vector<real_t>> cov(std::vector<std::vector<real_t>> A);
	Ref<MLPPMatrix> covm(const Ref<MLPPMatrix> &A);

	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> eig(std::vector<std::vector<real_t>> A);

	struct EigenResultOld {
		std::vector<std::vector<real_t>> eigen_vectors;
		std::vector<std::vector<real_t>> eigen_values;
	};

	EigenResultOld eigen_old(std::vector<std::vector<real_t>> A);

	struct EigenResult {
		Ref<MLPPMatrix> eigen_vectors;
		Ref<MLPPMatrix> eigen_values;
	};

	EigenResult eigen(Ref<MLPPMatrix> A);

	struct SVDResultOld {
		std::vector<std::vector<real_t>> U;
		std::vector<std::vector<real_t>> S;
		std::vector<std::vector<real_t>> Vt;
	};

	SVDResultOld SVD(std::vector<std::vector<real_t>> A);

	struct SVDResult {
		Ref<MLPPMatrix> U;
		Ref<MLPPMatrix> S;
		Ref<MLPPMatrix> Vt;
	};

	SVDResult svd(const Ref<MLPPMatrix> &A);

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
	Ref<MLPPVector> flattenvv(const Vector<Ref<MLPPVector>> &A);
	Ref<MLPPVector> flattenv(const Ref<MLPPMatrix> &A);

	std::vector<real_t> solve(std::vector<std::vector<real_t>> A, std::vector<real_t> b);

	bool positiveDefiniteChecker(std::vector<std::vector<real_t>> A);

	bool negativeDefiniteChecker(std::vector<std::vector<real_t>> A);

	bool zeroEigenvalue(std::vector<std::vector<real_t>> A);

	void printMatrix(std::vector<std::vector<real_t>> A);

	// VECTOR FUNCTIONS

	std::vector<std::vector<real_t>> outerProduct(std::vector<real_t> a, std::vector<real_t> b); // This multiplies a, bT
	Ref<MLPPMatrix> outer_product(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b); // This multiplies a, bT

	std::vector<real_t> hadamard_product(std::vector<real_t> a, std::vector<real_t> b);
	Ref<MLPPVector> hadamard_productnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	void hadamard_productv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out);

	std::vector<real_t> elementWiseDivision(std::vector<real_t> a, std::vector<real_t> b);
	Ref<MLPPVector> element_wise_division(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	std::vector<real_t> scalarMultiply(real_t scalar, std::vector<real_t> a);
	Ref<MLPPVector> scalar_multiplynv(real_t scalar, const Ref<MLPPVector> &a);
	void scalar_multiplyv(real_t scalar, const Ref<MLPPVector> &a, Ref<MLPPVector> out);

	std::vector<real_t> scalarAdd(real_t scalar, std::vector<real_t> a);
	Ref<MLPPVector> scalar_addnv(real_t scalar, const Ref<MLPPVector> &a);
	void scalar_addv(real_t scalar, const Ref<MLPPVector> &a, Ref<MLPPVector> out);

	std::vector<real_t> addition(std::vector<real_t> a, std::vector<real_t> b);
	Ref<MLPPVector> additionnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	void additionv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out);

	std::vector<real_t> subtraction(std::vector<real_t> a, std::vector<real_t> b);
	Ref<MLPPVector> subtractionnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	void subtractionv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out);

	std::vector<real_t> subtractMatrixRows(std::vector<real_t> a, std::vector<std::vector<real_t>> B);
	Ref<MLPPVector> subtract_matrix_rows(const Ref<MLPPVector> &a, const Ref<MLPPMatrix> &B);

	std::vector<real_t> log(std::vector<real_t> a);
	std::vector<real_t> log10(std::vector<real_t> a);
	std::vector<real_t> exp(std::vector<real_t> a);
	std::vector<real_t> erf(std::vector<real_t> a);
	std::vector<real_t> exponentiate(std::vector<real_t> a, real_t p);
	std::vector<real_t> sqrt(std::vector<real_t> a);
	std::vector<real_t> cbrt(std::vector<real_t> a);

	Ref<MLPPVector> lognv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> log10nv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> expnv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> erfnv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> exponentiatenv(const Ref<MLPPVector> &a, real_t p);
	Ref<MLPPVector> sqrtnv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> cbrtnv(const Ref<MLPPVector> &a);

	real_t dot(std::vector<real_t> a, std::vector<real_t> b);
	real_t dotv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	std::vector<real_t> cross(std::vector<real_t> a, std::vector<real_t> b);

	std::vector<real_t> abs(std::vector<real_t> a);

	std::vector<real_t> zerovec(int n);
	std::vector<real_t> onevec(int n);
	std::vector<real_t> full(int n, int k);

	Ref<MLPPVector> absv(const Ref<MLPPVector> &a);

	Ref<MLPPVector> zerovecv(int n);
	Ref<MLPPVector> onevecv(int n);
	Ref<MLPPVector> fullv(int n, int k);

	std::vector<std::vector<real_t>> diag(std::vector<real_t> a);
	Ref<MLPPVector> diagm(const Ref<MLPPVector> &a);

	std::vector<real_t> sin(std::vector<real_t> a);
	std::vector<real_t> cos(std::vector<real_t> a);

	Ref<MLPPVector> sinv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> cosv(const Ref<MLPPVector> &a);

	std::vector<real_t> max(std::vector<real_t> a, std::vector<real_t> b);

	Ref<MLPPVector> maxnvv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	real_t max(std::vector<real_t> a);
	real_t min(std::vector<real_t> a);

	real_t maxvr(const Ref<MLPPVector> &a);
	real_t minvr(const Ref<MLPPVector> &a);

	std::vector<real_t> round(std::vector<real_t> a);

	real_t euclideanDistance(std::vector<real_t> a, std::vector<real_t> b);
	real_t euclidean_distance(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	real_t euclidean_distance_squared(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	real_t norm_2(std::vector<real_t> a);

	real_t norm_sq(std::vector<real_t> a);
	real_t norm_sqv(const Ref<MLPPVector> &a);

	real_t sum_elements(std::vector<real_t> a);
	real_t sum_elementsv(const Ref<MLPPVector> &a);

	real_t cosineSimilarity(std::vector<real_t> a, std::vector<real_t> b);

	void printVector(std::vector<real_t> a);

	// MATRIX-VECTOR FUNCTIONS
	std::vector<std::vector<real_t>> mat_vec_add(std::vector<std::vector<real_t>> A, std::vector<real_t> b);
	std::vector<real_t> mat_vec_mult(std::vector<std::vector<real_t>> A, std::vector<real_t> b);

	Ref<MLPPMatrix> mat_vec_addv(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b);
	Ref<MLPPVector> mat_vec_multv(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b);

	// TENSOR FUNCTIONS
	std::vector<std::vector<std::vector<real_t>>> addition(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);

	Vector<Ref<MLPPMatrix>> addition_vt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B);

	std::vector<std::vector<std::vector<real_t>>> elementWiseDivision(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);
	Vector<Ref<MLPPMatrix>> element_wise_division_vt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B);

	std::vector<std::vector<std::vector<real_t>>> sqrt(std::vector<std::vector<std::vector<real_t>>> A);
	Vector<Ref<MLPPMatrix>> sqrt_vt(const Vector<Ref<MLPPMatrix>> &A);

	std::vector<std::vector<std::vector<real_t>>> exponentiate(std::vector<std::vector<std::vector<real_t>>> A, real_t p);
	Vector<Ref<MLPPMatrix>> exponentiate_vt(const Vector<Ref<MLPPMatrix>> &A, real_t p);

	std::vector<std::vector<real_t>> tensor_vec_mult(std::vector<std::vector<std::vector<real_t>>> A, std::vector<real_t> b);

	std::vector<real_t> flatten(std::vector<std::vector<std::vector<real_t>>> A);

	void printTensor(std::vector<std::vector<std::vector<real_t>>> A);

	std::vector<std::vector<std::vector<real_t>>> scalarMultiply(real_t scalar, std::vector<std::vector<std::vector<real_t>>> A);
	std::vector<std::vector<std::vector<real_t>>> scalarAdd(real_t scalar, std::vector<std::vector<std::vector<real_t>>> A);

	Vector<Ref<MLPPMatrix>> scalar_multiply_vm(real_t scalar, Vector<Ref<MLPPMatrix>> A);
	Vector<Ref<MLPPMatrix>> scalar_add_vm(real_t scalar, Vector<Ref<MLPPMatrix>> A);

	std::vector<std::vector<std::vector<real_t>>> resize(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);

	Vector<Ref<MLPPMatrix>> resize_vt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B);

	std::vector<std::vector<std::vector<real_t>>> hadamard_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);

	std::vector<std::vector<std::vector<real_t>>> max(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);
	Vector<Ref<MLPPMatrix>> max_vt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B);

	std::vector<std::vector<std::vector<real_t>>> abs(std::vector<std::vector<std::vector<real_t>>> A);
	Vector<Ref<MLPPMatrix>> abs_vt(const Vector<Ref<MLPPMatrix>> &A);

	real_t norm_2(std::vector<std::vector<std::vector<real_t>>> A);

	std::vector<std::vector<std::vector<real_t>>> vector_wise_tensor_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<real_t>> B);

protected:
	static void _bind_methods();
};

#endif /* LinAlg_hpp */