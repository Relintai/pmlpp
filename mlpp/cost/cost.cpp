//
//  Reg.cpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "cost.h"
#include "../lin_alg/lin_alg.h"
#include "../regularization/reg.h"
#include <cmath>
#include <iostream>


double MLPPCost::MSE(std::vector<double> y_hat, std::vector<double> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += (y_hat[i] - y[i]) * (y_hat[i] - y[i]);
	}
	return sum / 2 * y_hat.size();
}

double MLPPCost::MSE(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += (y_hat[i][j] - y[i][j]) * (y_hat[i][j] - y[i][j]);
		}
	}
	return sum / 2 * y_hat.size();
}

std::vector<double> MLPPCost::MSEDeriv(std::vector<double> y_hat, std::vector<double> y) {
	LinAlg alg;
	return alg.subtraction(y_hat, y);
}

std::vector<std::vector<double>> MLPPCost::MSEDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	LinAlg alg;
	return alg.subtraction(y_hat, y);
}

double MLPPCost::RMSE(std::vector<double> y_hat, std::vector<double> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += (y_hat[i] - y[i]) * (y_hat[i] - y[i]);
	}
	return sqrt(sum / y_hat.size());
}

double MLPPCost::RMSE(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += (y_hat[i][j] - y[i][j]) * (y_hat[i][j] - y[i][j]);
		}
	}
	return sqrt(sum / y_hat.size());
}

std::vector<double> MLPPCost::RMSEDeriv(std::vector<double> y_hat, std::vector<double> y) {
	LinAlg alg;
	return alg.scalarMultiply(1 / (2 * sqrt(MSE(y_hat, y))), MSEDeriv(y_hat, y));
}

std::vector<std::vector<double>> MLPPCost::RMSEDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	LinAlg alg;
	return alg.scalarMultiply(1 / (2 / sqrt(MSE(y_hat, y))), MSEDeriv(y_hat, y));
}

double MLPPCost::MAE(std::vector<double> y_hat, std::vector<double> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += abs((y_hat[i] - y[i]));
	}
	return sum / y_hat.size();
}

double MLPPCost::MAE(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += abs((y_hat[i][j] - y[i][j]));
		}
	}
	return sum / y_hat.size();
}

std::vector<double> MLPPCost::MAEDeriv(std::vector<double> y_hat, std::vector<double> y) {
	std::vector<double> deriv;
	deriv.resize(y_hat.size());
	for (int i = 0; i < deriv.size(); i++) {
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

std::vector<std::vector<double>> MLPPCost::MAEDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	std::vector<std::vector<double>> deriv;
	deriv.resize(y_hat.size());
	for (int i = 0; i < deriv.size(); i++) {
		deriv.resize(y_hat[i].size());
	}
	for (int i = 0; i < deriv.size(); i++) {
		for (int j = 0; j < deriv[i].size(); j++) {
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

double MLPPCost::MBE(std::vector<double> y_hat, std::vector<double> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += (y_hat[i] - y[i]);
	}
	return sum / y_hat.size();
}

double MLPPCost::MBE(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += (y_hat[i][j] - y[i][j]);
		}
	}
	return sum / y_hat.size();
}

std::vector<double> MLPPCost::MBEDeriv(std::vector<double> y_hat, std::vector<double> y) {
	LinAlg alg;
	return alg.onevec(y_hat.size());
}

std::vector<std::vector<double>> MLPPCost::MBEDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	LinAlg alg;
	return alg.onemat(y_hat.size(), y_hat[0].size());
}

double MLPPCost::LogLoss(std::vector<double> y_hat, std::vector<double> y) {
	double sum = 0;
	double eps = 1e-8;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += -(y[i] * std::log(y_hat[i] + eps) + (1 - y[i]) * std::log(1 - y_hat[i] + eps));
	}

	return sum / y_hat.size();
}

double MLPPCost::LogLoss(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	double sum = 0;
	double eps = 1e-8;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += -(y[i][j] * std::log(y_hat[i][j] + eps) + (1 - y[i][j]) * std::log(1 - y_hat[i][j] + eps));
		}
	}

	return sum / y_hat.size();
}

std::vector<double> MLPPCost::LogLossDeriv(std::vector<double> y_hat, std::vector<double> y) {
	LinAlg alg;
	return alg.addition(alg.scalarMultiply(-1, alg.elementWiseDivision(y, y_hat)), alg.elementWiseDivision(alg.scalarMultiply(-1, alg.scalarAdd(-1, y)), alg.scalarMultiply(-1, alg.scalarAdd(-1, y_hat))));
}

std::vector<std::vector<double>> MLPPCost::LogLossDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	LinAlg alg;
	return alg.addition(alg.scalarMultiply(-1, alg.elementWiseDivision(y, y_hat)), alg.elementWiseDivision(alg.scalarMultiply(-1, alg.scalarAdd(-1, y)), alg.scalarMultiply(-1, alg.scalarAdd(-1, y_hat))));
}

double MLPPCost::CrossEntropy(std::vector<double> y_hat, std::vector<double> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += y[i] * std::log(y_hat[i]);
	}

	return -1 * sum;
}

double MLPPCost::CrossEntropy(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += y[i][j] * std::log(y_hat[i][j]);
		}
	}

	return -1 * sum;
}

std::vector<double> MLPPCost::CrossEntropyDeriv(std::vector<double> y_hat, std::vector<double> y) {
	LinAlg alg;
	return alg.scalarMultiply(-1, alg.elementWiseDivision(y, y_hat));
}

std::vector<std::vector<double>> MLPPCost::CrossEntropyDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	LinAlg alg;
	return alg.scalarMultiply(-1, alg.elementWiseDivision(y, y_hat));
}

double MLPPCost::HuberLoss(std::vector<double> y_hat, std::vector<double> y, double delta) {
	LinAlg alg;
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		if (abs(y[i] - y_hat[i]) <= delta) {
			sum += (y[i] - y_hat[i]) * (y[i] - y_hat[i]);
		} else {
			sum += 2 * delta * abs(y[i] - y_hat[i]) - delta * delta;
		}
	}
	return sum;
}

double MLPPCost::HuberLoss(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y, double delta) {
	LinAlg alg;
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			if (abs(y[i][j] - y_hat[i][j]) <= delta) {
				sum += (y[i][j] - y_hat[i][j]) * (y[i][j] - y_hat[i][j]);
			} else {
				sum += 2 * delta * abs(y[i][j] - y_hat[i][j]) - delta * delta;
			}
		}
	}
	return sum;
}

std::vector<double> MLPPCost::HuberLossDeriv(std::vector<double> y_hat, std::vector<double> y, double delta) {
	LinAlg alg;
	double sum = 0;
	std::vector<double> deriv;
	deriv.resize(y_hat.size());

	for (int i = 0; i < y_hat.size(); i++) {
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

std::vector<std::vector<double>> MLPPCost::HuberLossDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y, double delta) {
	LinAlg alg;
	double sum = 0;
	std::vector<std::vector<double>> deriv;
	deriv.resize(y_hat.size());
	for (int i = 0; i < deriv.size(); i++) {
		deriv[i].resize(y_hat[i].size());
	}

	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
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

double MLPPCost::HingeLoss(std::vector<double> y_hat, std::vector<double> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += fmax(0, 1 - y[i] * y_hat[i]);
	}

	return sum / y_hat.size();
}

double MLPPCost::HingeLoss(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += fmax(0, 1 - y[i][j] * y_hat[i][j]);
		}
	}

	return sum / y_hat.size();
}

std::vector<double> MLPPCost::HingeLossDeriv(std::vector<double> y_hat, std::vector<double> y) {
	std::vector<double> deriv;
	deriv.resize(y_hat.size());
	for (int i = 0; i < y_hat.size(); i++) {
		if (1 - y[i] * y_hat[i] > 0) {
			deriv[i] = -y[i];
		} else {
			deriv[i] = 0;
		}
	}
	return deriv;
}

std::vector<std::vector<double>> MLPPCost::HingeLossDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	std::vector<std::vector<double>> deriv;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			if (1 - y[i][j] * y_hat[i][j] > 0) {
				deriv[i][j] = -y[i][j];
			} else {
				deriv[i][j] = 0;
			}
		}
	}
	return deriv;
}

double MLPPCost::WassersteinLoss(std::vector<double> y_hat, std::vector<double> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += y_hat[i] * y[i];
	}
	return -sum / y_hat.size();
}

double MLPPCost::WassersteinLoss(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	double sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += y_hat[i][j] * y[i][j];
		}
	}
	return -sum / y_hat.size();
}

std::vector<double> MLPPCost::WassersteinLossDeriv(std::vector<double> y_hat, std::vector<double> y) {
	LinAlg alg;
	return alg.scalarMultiply(-1, y); // Simple.
}

std::vector<std::vector<double>> MLPPCost::WassersteinLossDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y) {
	LinAlg alg;
	return alg.scalarMultiply(-1, y); // Simple.
}

double MLPPCost::HingeLoss(std::vector<double> y_hat, std::vector<double> y, std::vector<double> weights, double C) {
	LinAlg alg;
	Reg regularization;
	return C * HingeLoss(y_hat, y) + regularization.regTerm(weights, 1, 0, "Ridge");
}
double MLPPCost::HingeLoss(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y, std::vector<std::vector<double>> weights, double C) {
	LinAlg alg;
	Reg regularization;
	return C * HingeLoss(y_hat, y) + regularization.regTerm(weights, 1, 0, "Ridge");
}

std::vector<double> MLPPCost::HingeLossDeriv(std::vector<double> y_hat, std::vector<double> y, double C) {
	LinAlg alg;
	Reg regularization;
	return alg.scalarMultiply(C, HingeLossDeriv(y_hat, y));
}
std::vector<std::vector<double>> MLPPCost::HingeLossDeriv(std::vector<std::vector<double>> y_hat, std::vector<std::vector<double>> y, double C) {
	LinAlg alg;
	Reg regularization;
	return alg.scalarMultiply(C, HingeLossDeriv(y_hat, y));
}

double MLPPCost::dualFormSVM(std::vector<double> alpha, std::vector<std::vector<double>> X, std::vector<double> y) {
	LinAlg alg;
	std::vector<std::vector<double>> Y = alg.diag(y); // Y is a diagnoal matrix. Y[i][j] = y[i] if i = i, else Y[i][j] = 0. Yt = Y.
	std::vector<std::vector<double>> K = alg.matmult(X, alg.transpose(X)); // TO DO: DON'T forget to add non-linear kernelizations.
	std::vector<std::vector<double>> Q = alg.matmult(alg.matmult(alg.transpose(Y), K), Y);
	double alphaQ = alg.matmult(alg.matmult({ alpha }, Q), alg.transpose({ alpha }))[0][0];
	std::vector<double> one = alg.onevec(alpha.size());

	return -alg.dot(one, alpha) + 0.5 * alphaQ;
}

std::vector<double> MLPPCost::dualFormSVMDeriv(std::vector<double> alpha, std::vector<std::vector<double>> X, std::vector<double> y) {
	LinAlg alg;
	std::vector<std::vector<double>> Y = alg.zeromat(y.size(), y.size());
	for (int i = 0; i < y.size(); i++) {
		Y[i][i] = y[i]; // Y is a diagnoal matrix. Y[i][j] = y[i] if i = i, else Y[i][j] = 0. Yt = Y.
	}
	std::vector<std::vector<double>> K = alg.matmult(X, alg.transpose(X)); // TO DO: DON'T forget to add non-linear kernelizations.
	std::vector<std::vector<double>> Q = alg.matmult(alg.matmult(alg.transpose(Y), K), Y);
	std::vector<double> alphaQDeriv = alg.mat_vec_mult(Q, alpha);
	std::vector<double> one = alg.onevec(alpha.size());

	return alg.subtraction(alphaQDeriv, one);
}
