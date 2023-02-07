//
//  PCA.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "pca.h"
#include "../data/data.h"
#include "../lin_alg/lin_alg.h"

#include <iostream>
#include <random>



MLPPPCA::MLPPPCA(std::vector<std::vector<real_t>> inputSet, int k) :
		inputSet(inputSet), k(k) {
}

std::vector<std::vector<real_t>> MLPPPCA::principalComponents() {
	MLPPLinAlg alg;
	MLPPData data;

	MLPPLinAlg::SDVResultOld svr_res = alg.SVD(alg.cov(inputSet));
	X_normalized = data.meanCentering(inputSet);
	U_reduce.resize(svr_res.U.size());
	for (int i = 0; i < k; i++) {
		for (int j = 0; j < svr_res.U.size(); j++) {
			U_reduce[j].push_back(svr_res.U[j][i]);
		}
	}
	Z = alg.matmult(alg.transpose(U_reduce), X_normalized);
	return Z;
}
// Simply tells us the percentage of variance maintained.
real_t MLPPPCA::score() {
	MLPPLinAlg alg;
	std::vector<std::vector<real_t>> X_approx = alg.matmult(U_reduce, Z);
	real_t num, den = 0;
	for (int i = 0; i < X_normalized.size(); i++) {
		num += alg.norm_sq(alg.subtraction(X_normalized[i], X_approx[i]));
	}
	num /= X_normalized.size();
	for (int i = 0; i < X_normalized.size(); i++) {
		den += alg.norm_sq(X_normalized[i]);
	}

	den /= X_normalized.size();
	if (den == 0) {
		den += 1e-10; // For numerical sanity as to not recieve a domain error
	}
	return 1 - num / den;
}

