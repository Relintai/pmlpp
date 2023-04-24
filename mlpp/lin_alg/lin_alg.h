
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

	//std::vector<std::vector<real_t>> gramMatrix(std::vector<std::vector<real_t>> A);
	//bool linearIndependenceChecker(std::vector<std::vector<real_t>> A);

	Ref<MLPPMatrix> gaussian_noise(int n, int m);

	Ref<MLPPMatrix> additionnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> subtractionnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> matmultnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	Ref<MLPPMatrix> hadamard_productnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> kronecker_productnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);
	Ref<MLPPMatrix> element_wise_divisionnvnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	Ref<MLPPMatrix> transposenm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> scalar_multiplynm(real_t scalar, const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> scalar_addnm(real_t scalar, const Ref<MLPPMatrix> &A);

	Ref<MLPPMatrix> lognm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> log10nm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> expnm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> erfnm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> exponentiatenm(const Ref<MLPPMatrix> &A, real_t p);
	Ref<MLPPMatrix> sqrtnm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> cbrtnm(const Ref<MLPPMatrix> &A);

	//std::vector<std::vector<real_t>> matrixPower(std::vector<std::vector<real_t>> A, int n);

	Ref<MLPPMatrix> absnm(const Ref<MLPPMatrix> &A);

	real_t detm(const Ref<MLPPMatrix> &A, int d);

	//real_t trace(std::vector<std::vector<real_t>> A);

	Ref<MLPPMatrix> cofactornm(const Ref<MLPPMatrix> &A, int n, int i, int j);
	Ref<MLPPMatrix> adjointnm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> inversenm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> pinversenm(const Ref<MLPPMatrix> &A);

	Ref<MLPPMatrix> zeromatnm(int n, int m);
	Ref<MLPPMatrix> onematnm(int n, int m);
	Ref<MLPPMatrix> fullnm(int n, int m, int k);

	Ref<MLPPMatrix> sinnm(const Ref<MLPPMatrix> &A);
	Ref<MLPPMatrix> cosnm(const Ref<MLPPMatrix> &A);

	//std::vector<std::vector<real_t>> rotate(std::vector<std::vector<real_t>> A, real_t theta, int axis = -1);

	Ref<MLPPMatrix> maxnm(const Ref<MLPPMatrix> &A, const Ref<MLPPMatrix> &B);

	//real_t max(std::vector<std::vector<real_t>> A);
	//real_t min(std::vector<std::vector<real_t>> A);

	//std::vector<std::vector<real_t>> round(std::vector<std::vector<real_t>> A);

	//real_t norm_2(std::vector<std::vector<real_t>> A);

	Ref<MLPPMatrix> identitym(int d);

	Ref<MLPPMatrix> covnm(const Ref<MLPPMatrix> &A);

	struct EigenResult {
		Ref<MLPPMatrix> eigen_vectors;
		Ref<MLPPMatrix> eigen_values;
	};

	EigenResult eigen(Ref<MLPPMatrix> A);

	struct SVDResult {
		Ref<MLPPMatrix> U;
		Ref<MLPPMatrix> S;
		Ref<MLPPMatrix> Vt;
	};

	SVDResult svd(const Ref<MLPPMatrix> &A);

	//std::vector<real_t> vectorProjection(std::vector<real_t> a, std::vector<real_t> b);

	//std::vector<std::vector<real_t>> gramSchmidtProcess(std::vector<std::vector<real_t>> A);

	/*
	struct QRDResult {
		std::vector<std::vector<real_t>> Q;
		std::vector<std::vector<real_t>> R;
	};
	*/

	//QRDResult qrd(std::vector<std::vector<real_t>> A);

	/*
	struct CholeskyResult {
		std::vector<std::vector<real_t>> L;
		std::vector<std::vector<real_t>> Lt;
	};

	CholeskyResult cholesky(std::vector<std::vector<real_t>> A);
	*/

	//real_t sum_elements(std::vector<std::vector<real_t>> A);

	Ref<MLPPVector> flattenvvnv(const Ref<MLPPMatrix> &A);

	/*
	std::vector<real_t> solve(std::vector<std::vector<real_t>> A, std::vector<real_t> b);

	bool positiveDefiniteChecker(std::vector<std::vector<real_t>> A);

	bool negativeDefiniteChecker(std::vector<std::vector<real_t>> A);

	bool zeroEigenvalue(std::vector<std::vector<real_t>> A);
	*/

	// VECTOR FUNCTIONS

	Ref<MLPPVector> flattenmnv(const Vector<Ref<MLPPVector>> &A);

	Ref<MLPPVector> hadamard_productnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	void hadamard_productv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out);

	Ref<MLPPVector> element_wise_divisionnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	Ref<MLPPVector> scalar_multiplynv(real_t scalar, const Ref<MLPPVector> &a);
	void scalar_multiplyv(real_t scalar, const Ref<MLPPVector> &a, Ref<MLPPVector> out);

	Ref<MLPPVector> scalar_addnv(real_t scalar, const Ref<MLPPVector> &a);
	void scalar_addv(real_t scalar, const Ref<MLPPVector> &a, Ref<MLPPVector> out);

	Ref<MLPPVector> additionnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	void additionv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out);

	Ref<MLPPVector> subtractionnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	void subtractionv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b, Ref<MLPPVector> out);

	Ref<MLPPVector> lognv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> log10nv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> expnv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> erfnv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> exponentiatenv(const Ref<MLPPVector> &a, real_t p);
	Ref<MLPPVector> sqrtnv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> cbrtnv(const Ref<MLPPVector> &a);

	real_t dotnv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	//std::vector<real_t> cross(std::vector<real_t> a, std::vector<real_t> b);

	Ref<MLPPVector> absv(const Ref<MLPPVector> &a);

	Ref<MLPPVector> zerovecnv(int n);
	Ref<MLPPVector> onevecnv(int n);
	Ref<MLPPVector> fullnv(int n, int k);

	Ref<MLPPVector> sinnv(const Ref<MLPPVector> &a);
	Ref<MLPPVector> cosnv(const Ref<MLPPVector> &a);

	Ref<MLPPVector> maxnvv(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	real_t maxvr(const Ref<MLPPVector> &a);
	real_t minvr(const Ref<MLPPVector> &a);

	//std::vector<real_t> round(std::vector<real_t> a);

	real_t euclidean_distance(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);
	real_t euclidean_distance_squared(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b);

	/*
	real_t norm_2(std::vector<real_t> a);
	*/

	real_t norm_sqv(const Ref<MLPPVector> &a);

	real_t sum_elementsv(const Ref<MLPPVector> &a);

	//real_t cosineSimilarity(std::vector<real_t> a, std::vector<real_t> b);

	// MATRIX-VECTOR FUNCTIONS
	Ref<MLPPVector> mat_vec_multnv(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b);
	Ref<MLPPVector> subtract_matrix_rowsnv(const Ref<MLPPVector> &a, const Ref<MLPPMatrix> &B);

	Ref<MLPPMatrix> outer_product(const Ref<MLPPVector> &a, const Ref<MLPPVector> &b); // This multiplies a, bT
	Ref<MLPPMatrix> mat_vec_addnm(const Ref<MLPPMatrix> &A, const Ref<MLPPVector> &b);
	Ref<MLPPMatrix> diagnm(const Ref<MLPPVector> &a);

	// TENSOR FUNCTIONS
	Vector<Ref<MLPPMatrix>> additionnvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B);

	Vector<Ref<MLPPMatrix>> element_wise_divisionnvnvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B);

	Vector<Ref<MLPPMatrix>> sqrtnvt(const Vector<Ref<MLPPMatrix>> &A);

	Vector<Ref<MLPPMatrix>> exponentiatenvt(const Vector<Ref<MLPPMatrix>> &A, real_t p);

	//std::vector<std::vector<real_t>> tensor_vec_mult(std::vector<std::vector<std::vector<real_t>>> A, std::vector<real_t> b);

	//std::vector<real_t> flatten(std::vector<std::vector<std::vector<real_t>>> A);

	Vector<Ref<MLPPMatrix>> scalar_multiplynvt(real_t scalar, Vector<Ref<MLPPMatrix>> A);
	Vector<Ref<MLPPMatrix>> scalar_addnvt(real_t scalar, Vector<Ref<MLPPMatrix>> A);

	Vector<Ref<MLPPMatrix>> resizenvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B);

	//std::vector<std::vector<std::vector<real_t>>> hadamard_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<std::vector<real_t>>> B);

	Vector<Ref<MLPPMatrix>> maxnvt(const Vector<Ref<MLPPMatrix>> &A, const Vector<Ref<MLPPMatrix>> &B);
	Vector<Ref<MLPPMatrix>> absnvt(const Vector<Ref<MLPPMatrix>> &A);

	//real_t norm_2(std::vector<std::vector<std::vector<real_t>>> A);

	//std::vector<std::vector<std::vector<real_t>>> vector_wise_tensor_product(std::vector<std::vector<std::vector<real_t>>> A, std::vector<std::vector<real_t>> B);

protected:
	static void _bind_methods();
};

#endif /* LinAlg_hpp */