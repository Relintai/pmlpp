//
//  UniLinReg.cpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#include "uni_lin_reg_old.h"

#include "../lin_alg/lin_alg_old.h"
#include "../stat/stat_old.h"

#include <iostream>

// General Multivariate Linear Regression Model
// ŷ = b0 + b1x1 + b2x2 + ... + bkxk

// Univariate Linear Regression Model
// ŷ = b0 + b1x1

MLPPUniLinRegOld::MLPPUniLinRegOld(std::vector<real_t> x, std::vector<real_t> y) :
		inputSet(x), outputSet(y) {
	MLPPStatOld estimator;
	b1 = estimator.b1Estimation(inputSet, outputSet);
	b0 = estimator.b0Estimation(inputSet, outputSet);
}

std::vector<real_t> MLPPUniLinRegOld::modelSetTest(std::vector<real_t> x) {
	MLPPLinAlgOld alg;
	return alg.scalarAdd(b0, alg.scalarMultiply(b1, x));
}

real_t MLPPUniLinRegOld::modelTest(real_t input) {
	return b0 + b1 * input;
}
