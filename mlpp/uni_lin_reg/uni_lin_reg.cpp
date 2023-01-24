//
//  UniLinReg.cpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#include "uni_lin_reg.h"
#include "../lin_alg/lin_alg.h"
#include "../stat/stat.h"
#include <iostream>

// General Multivariate Linear Regression Model
// ŷ = b0 + b1x1 + b2x2 + ... + bkxk

// Univariate Linear Regression Model
// ŷ = b0 + b1x1


UniLinReg::UniLinReg(std::vector<double> x, std::vector<double> y) :
		inputSet(x), outputSet(y) {
	Stat estimator;
	b1 = estimator.b1Estimation(inputSet, outputSet);
	b0 = estimator.b0Estimation(inputSet, outputSet);
}

std::vector<double> UniLinReg::modelSetTest(std::vector<double> x) {
	LinAlg alg;
	return alg.scalarAdd(b0, alg.scalarMultiply(b1, x));
}

double UniLinReg::modelTest(double input) {
	return b0 + b1 * input;
}

