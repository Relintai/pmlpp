//
//  Transforms.cpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "transforms.h"
#include "../lin_alg/lin_alg.h"
#include <cmath>
#include <iostream>
#include <string>

/*
// DCT ii.
// https://www.mathworks.com/help/images/discrete-cosine-transform.html
std::vector<std::vector<real_t>> MLPPTransforms::discreteCosineTransform(std::vector<std::vector<real_t>> A) {
	MLPPLinAlg alg;
	A = alg.scalarAdd(-128, A); // Center around 0.

	std::vector<std::vector<real_t>> B;
	B.resize(A.size());
	for (uint32_t i = 0; i < B.size(); i++) {
		B[i].resize(A[i].size());
	}

	int M = A.size();

	for (uint32_t i = 0; i < B.size(); i++) {
		for (uint32_t j = 0; j < B[i].size(); j++) {
			real_t sum = 0;
			real_t alphaI;
			if (i == 0) {
				alphaI = 1 / std::sqrt(M);
			} else {
				alphaI = std::sqrt(real_t(2) / real_t(M));
			}
			real_t alphaJ;
			if (j == 0) {
				alphaJ = 1 / std::sqrt(M);
			} else {
				alphaJ = std::sqrt(real_t(2) / real_t(M));
			}

			for (uint32_t k = 0; k < B.size(); k++) {
				for (uint32_t f = 0; f < B[k].size(); f++) {
					sum += A[k][f] * std::cos((Math_PI * i * (2 * k + 1)) / (2 * M)) * std::cos((Math_PI * j * (2 * f + 1)) / (2 * M));
				}
			}
			B[i][j] = sum;
			B[i][j] *= alphaI * alphaJ;
		}
	}
	return B;
}
*/

void MLPPTransforms::_bind_methods() {
}
