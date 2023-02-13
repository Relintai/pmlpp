//
//  Reg.cpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "cost_old.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include <cmath>
#include <iostream>

real_t MLPPCostOld::MSE(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		sum += (y_hat[i] - y[i]) * (y_hat[i] - y[i]);
	}
	return sum / 2 * y_hat.size();
}

real_t MLPPCostOld::MSE(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		for (uint32_t j = 0; j < y_hat[i].size(); j++) {
			sum += (y_hat[i][j] - y[i][j]) * (y_hat[i][j] - y[i][j]);
		}
	}
	return sum / 2 * y_hat.size();
}

std::vector<real_t> MLPPCostOld::MSEDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPLinAlg alg;
	return alg.subtraction(y_hat, y);
}

std::vector<std::vector<real_t>> MLPPCostOld::MSEDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPLinAlg alg;
	return alg.subtraction(y_hat, y);
}

real_t MLPPCostOld::RMSE(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		sum += (y_hat[i] - y[i]) * (y_hat[i] - y[i]);
	}
	return sqrt(sum / y_hat.size());
}

real_t MLPPCostOld::RMSE(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		for (uint32_t j = 0; j < y_hat[i].size(); j++) {
			sum += (y_hat[i][j] - y[i][j]) * (y_hat[i][j] - y[i][j]);
		}
	}
	return sqrt(sum / y_hat.size());
}

std::vector<real_t> MLPPCostOld::RMSEDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPLinAlg alg;
	return alg.scalarMultiply(1 / (2 * sqrt(MSE(y_hat, y))), MSEDeriv(y_hat, y));
}

std::vector<std::vector<real_t>> MLPPCostOld::RMSEDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPLinAlg alg;
	return alg.scalarMultiply(1 / (2 / sqrt(MSE(y_hat, y))), MSEDeriv(y_hat, y));
}

real_t MLPPCostOld::MAE(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		sum += abs((y_hat[i] - y[i]));
	}
	return sum / y_hat.size();
}

real_t MLPPCostOld::MAE(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		for (uint32_t j = 0; j < y_hat[i].size(); j++) {
			sum += abs((y_hat[i][j] - y[i][j]));
		}
	}
	return sum / y_hat.size();
}

std::vector<real_t> MLPPCostOld::MAEDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	std::vector<real_t> deriv;
	deriv.resize(y_hat.size());
	for (uint32_t i = 0; i < deriv.size(); i++) {
		if (y_hat[i] < 0) {
			deriv[i] = -1;
		} else if (y_hat[i] == 0) {
			deriv[i] = 0;
		} else {
			deriv[i] = 1;
		}
	}
	return deriv;
}

std::vector<std::vector<real_t>> MLPPCostOld::MAEDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	std::vector<std::vector<real_t>> deriv;
	deriv.resize(y_hat.size());
	for (uint32_t i = 0; i < deriv.size(); i++) {
		deriv.resize(y_hat[i].size());
	}
	for (uint32_t i = 0; i < deriv.size(); i++) {
		for (uint32_t j = 0; j < deriv[i].size(); j++) {
			if (y_hat[i][j] < 0) {
				deriv[i][j] = -1;
			} else if (y_hat[i][j] == 0) {
				deriv[i][j] = 0;
			} else {
				deriv[i][j] = 1;
			}
		}
	}
	return deriv;
}

real_t MLPPCostOld::MBE(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		sum += (y_hat[i] - y[i]);
	}
	return sum / y_hat.size();
}

real_t MLPPCostOld::MBE(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		for (uint32_t j = 0; j < y_hat[i].size(); j++) {
			sum += (y_hat[i][j] - y[i][j]);
		}
	}
	return sum / y_hat.size();
}

std::vector<real_t> MLPPCostOld::MBEDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPLinAlg alg;
	return alg.onevec(y_hat.size());
}

std::vector<std::vector<real_t>> MLPPCostOld::MBEDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPLinAlg alg;
	return alg.onemat(y_hat.size(), y_hat[0].size());
}

real_t MLPPCostOld::LogLoss(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	real_t eps = 1e-8;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		sum += -(y[i] * std::log(y_hat[i] + eps) + (1 - y[i]) * std::log(1 - y_hat[i] + eps));
	}

	return sum / y_hat.size();
}

real_t MLPPCostOld::LogLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	real_t eps = 1e-8;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		for (uint32_t j = 0; j < y_hat[i].size(); j++) {
			sum += -(y[i][j] * std::log(y_hat[i][j] + eps) + (1 - y[i][j]) * std::log(1 - y_hat[i][j] + eps));
		}
	}

	return sum / y_hat.size();
}

std::vector<real_t> MLPPCostOld::LogLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPLinAlg alg;
	return alg.addition(alg.scalarMultiply(-1, alg.elementWiseDivision(y, y_hat)), alg.elementWiseDivision(alg.scalarMultiply(-1, alg.scalarAdd(-1, y)), alg.scalarMultiply(-1, alg.scalarAdd(-1, y_hat))));
}

std::vector<std::vector<real_t>> MLPPCostOld::LogLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPLinAlg alg;
	return alg.addition(alg.scalarMultiply(-1, alg.elementWiseDivision(y, y_hat)), alg.elementWiseDivision(alg.scalarMultiply(-1, alg.scalarAdd(-1, y)), alg.scalarMultiply(-1, alg.scalarAdd(-1, y_hat))));
}

real_t MLPPCostOld::CrossEntropy(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		sum += y[i] * std::log(y_hat[i]);
	}

	return -1 * sum;
}

real_t MLPPCostOld::CrossEntropy(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		for (uint32_t j = 0; j < y_hat[i].size(); j++) {
			sum += y[i][j] * std::log(y_hat[i][j]);
		}
	}

	return -1 * sum;
}

std::vector<real_t> MLPPCostOld::CrossEntropyDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPLinAlg alg;
	return alg.scalarMultiply(-1, alg.elementWiseDivision(y, y_hat));
}

std::vector<std::vector<real_t>> MLPPCostOld::CrossEntropyDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPLinAlg alg;
	return alg.scalarMultiply(-1, alg.elementWiseDivision(y, y_hat));
}

real_t MLPPCostOld::HuberLoss(std::vector<real_t> y_hat, std::vector<real_t> y, real_t delta) {
	MLPPLinAlg alg;
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		if (abs(y[i] - y_hat[i]) <= delta) {
			sum += (y[i] - y_hat[i]) * (y[i] - y_hat[i]);
		} else {
			sum += 2 * delta * abs(y[i] - y_hat[i]) - delta * delta;
		}
	}
	return sum;
}

real_t MLPPCostOld::HuberLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y, real_t delta) {
	MLPPLinAlg alg;
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		for (uint32_t j = 0; j < y_hat[i].size(); j++) {
			if (abs(y[i][j] - y_hat[i][j]) <= delta) {
				sum += (y[i][j] - y_hat[i][j]) * (y[i][j] - y_hat[i][j]);
			} else {
				sum += 2 * delta * abs(y[i][j] - y_hat[i][j]) - delta * delta;
			}
		}
	}
	return sum;
}

std::vector<real_t> MLPPCostOld::HuberLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y, real_t delta) {
	MLPPLinAlg alg;
	std::vector<real_t> deriv;
	deriv.resize(y_hat.size());

	for (uint32_t i = 0; i < y_hat.size(); i++) {
		if (abs(y[i] - y_hat[i]) <= delta) {
			deriv.push_back(-(y[i] - y_hat[i]));
		} else {
			if (y_hat[i] > 0 || y_hat[i] < 0) {
				deriv.push_back(2 * delta * (y_hat[i] / abs(y_hat[i])));
			} else {
				deriv.push_back(0);
			}
		}
	}
	return deriv;
}

std::vector<std::vector<real_t>> MLPPCostOld::HuberLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y, real_t delta) {
	MLPPLinAlg alg;

	std::vector<std::vector<real_t>> deriv;
	deriv.resize(y_hat.size());
	for (uint32_t i = 0; i < deriv.size(); i++) {
		deriv[i].resize(y_hat[i].size());
	}

	for (uint32_t i = 0; i < y_hat.size(); i++) {
		for (uint32_t j = 0; j < y_hat[i].size(); j++) {
			if (abs(y[i][j] - y_hat[i][j]) <= delta) {
				deriv[i].push_back(-(y[i][j] - y_hat[i][j]));
			} else {
				if (y_hat[i][j] > 0 || y_hat[i][j] < 0) {
					deriv[i].push_back(2 * delta * (y_hat[i][j] / abs(y_hat[i][j])));
				} else {
					deriv[i].push_back(0);
				}
			}
		}
	}
	return deriv;
}

real_t MLPPCostOld::HingeLoss(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		sum += fmax(0, 1 - y[i] * y_hat[i]);
	}

	return sum / y_hat.size();
}

real_t MLPPCostOld::HingeLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		for (uint32_t j = 0; j < y_hat[i].size(); j++) {
			sum += fmax(0, 1 - y[i][j] * y_hat[i][j]);
		}
	}

	return sum / y_hat.size();
}

std::vector<real_t> MLPPCostOld::HingeLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	std::vector<real_t> deriv;
	deriv.resize(y_hat.size());
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		if (1 - y[i] * y_hat[i] > 0) {
			deriv[i] = -y[i];
		} else {
			deriv[i] = 0;
		}
	}
	return deriv;
}

std::vector<std::vector<real_t>> MLPPCostOld::HingeLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	std::vector<std::vector<real_t>> deriv;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		for (uint32_t j = 0; j < y_hat[i].size(); j++) {
			if (1 - y[i][j] * y_hat[i][j] > 0) {
				deriv[i][j] = -y[i][j];
			} else {
				deriv[i][j] = 0;
			}
		}
	}
	return deriv;
}

real_t MLPPCostOld::WassersteinLoss(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		sum += y_hat[i] * y[i];
	}
	return -sum / y_hat.size();
}

real_t MLPPCostOld::WassersteinLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		for (uint32_t j = 0; j < y_hat[i].size(); j++) {
			sum += y_hat[i][j] * y[i][j];
		}
	}
	return -sum / y_hat.size();
}

std::vector<real_t> MLPPCostOld::WassersteinLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPLinAlg alg;
	return alg.scalarMultiply(-1, y); // Simple.
}

std::vector<std::vector<real_t>> MLPPCostOld::WassersteinLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPLinAlg alg;
	return alg.scalarMultiply(-1, y); // Simple.
}

real_t MLPPCostOld::HingeLoss(std::vector<real_t> y_hat, std::vector<real_t> y, std::vector<real_t> weights, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	return C * HingeLoss(y_hat, y) + regularization.regTerm(weights, 1, 0, "Ridge");
}
real_t MLPPCostOld::HingeLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y, std::vector<std::vector<real_t>> weights, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	return C * HingeLoss(y_hat, y) + regularization.regTerm(weights, 1, 0, "Ridge");
}

std::vector<real_t> MLPPCostOld::HingeLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	return alg.scalarMultiply(C, HingeLossDeriv(y_hat, y));
}
std::vector<std::vector<real_t>> MLPPCostOld::HingeLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	return alg.scalarMultiply(C, HingeLossDeriv(y_hat, y));
}

real_t MLPPCostOld::dualFormSVM(std::vector<real_t> alpha, std::vector<std::vector<real_t>> X, std::vector<real_t> y) {
	MLPPLinAlg alg;
	std::vector<std::vector<real_t>> Y = alg.diag(y); // Y is a diagnoal matrix. Y[i][j] = y[i] if i = i, else Y[i][j] = 0. Yt = Y.
	std::vector<std::vector<real_t>> K = alg.matmult(X, alg.transpose(X)); // TO DO: DON'T forget to add non-linear kernelizations.
	std::vector<std::vector<real_t>> Q = alg.matmult(alg.matmult(alg.transpose(Y), K), Y);
	real_t alphaQ = alg.matmult(alg.matmult({ alpha }, Q), alg.transpose({ alpha }))[0][0];
	std::vector<real_t> one = alg.onevec(alpha.size());

	return -alg.dot(one, alpha) + 0.5 * alphaQ;
}

std::vector<real_t> MLPPCostOld::dualFormSVMDeriv(std::vector<real_t> alpha, std::vector<std::vector<real_t>> X, std::vector<real_t> y) {
	MLPPLinAlg alg;
	std::vector<std::vector<real_t>> Y = alg.zeromat(y.size(), y.size());
	for (uint32_t i = 0; i < y.size(); i++) {
		Y[i][i] = y[i]; // Y is a diagnoal matrix. Y[i][j] = y[i] if i = i, else Y[i][j] = 0. Yt = Y.
	}
	std::vector<std::vector<real_t>> K = alg.matmult(X, alg.transpose(X)); // TO DO: DON'T forget to add non-linear kernelizations.
	std::vector<std::vector<real_t>> Q = alg.matmult(alg.matmult(alg.transpose(Y), K), Y);
	std::vector<real_t> alphaQDeriv = alg.mat_vec_mult(Q, alpha);
	std::vector<real_t> one = alg.onevec(alpha.size());

	return alg.subtraction(alphaQDeriv, one);
}
