//
//  Activation.cpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "activation.h"
#include "../lin_alg/lin_alg.h"
#include <algorithm>
#include <cmath>
#include <iostream>

real_t MLPPActivation::linear(real_t z, bool deriv) {
	if (deriv) {
		return 1;
	}
	return z;
}

std::vector<real_t> MLPPActivation::linear(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		MLPPLinAlg alg;
		return alg.onevec(z.size());
	}
	return z;
}

std::vector<std::vector<real_t>> MLPPActivation::linear(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		MLPPLinAlg alg;
		return alg.onemat(z.size(), z[0].size());
	}
	return z;
}

real_t MLPPActivation::sigmoid(real_t z, bool deriv) {
	if (deriv) {
		return sigmoid(z) * (1 - sigmoid(z));
	}
	return 1 / (1 + exp(-z));
}

std::vector<real_t> MLPPActivation::sigmoid(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.subtraction(sigmoid(z), alg.hadamard_product(sigmoid(z), sigmoid(z)));
	}
	return alg.elementWiseDivision(alg.onevec(z.size()), alg.addition(alg.onevec(z.size()), alg.exp(alg.scalarMultiply(-1, z))));
}

std::vector<std::vector<real_t>> MLPPActivation::sigmoid(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.subtraction(sigmoid(z), alg.hadamard_product(sigmoid(z), sigmoid(z)));
	}
	return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.addition(alg.onemat(z.size(), z[0].size()), alg.exp(alg.scalarMultiply(-1, z))));
}

std::vector<real_t> MLPPActivation::softmax(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	std::vector<real_t> a;
	a.resize(z.size());
	std::vector<real_t> expZ = alg.exp(z);
	real_t sum = 0;

	for (int i = 0; i < z.size(); i++) {
		sum += expZ[i];
	}
	for (int i = 0; i < z.size(); i++) {
		a[i] = expZ[i] / sum;
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivation::softmax(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (int i = 0; i < z.size(); i++) {
		a[i] = softmax(z[i]);
	}
	return a;
}

std::vector<real_t> MLPPActivation::adjSoftmax(std::vector<real_t> z) {
	MLPPLinAlg alg;
	std::vector<real_t> a;
	real_t C = -*std::max_element(z.begin(), z.end());
	z = alg.scalarAdd(C, z);

	return softmax(z);
}

std::vector<std::vector<real_t>> MLPPActivation::adjSoftmax(std::vector<std::vector<real_t>> z) {
	MLPPLinAlg alg;
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (int i = 0; i < z.size(); i++) {
		a[i] = adjSoftmax(z[i]);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivation::softmaxDeriv(std::vector<real_t> z) {
	MLPPLinAlg alg;
	std::vector<std::vector<real_t>> deriv;
	std::vector<real_t> a = softmax(z);
	deriv.resize(a.size());
	for (int i = 0; i < deriv.size(); i++) {
		deriv[i].resize(a.size());
	}
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < z.size(); j++) {
			if (i == j) {
				deriv[i][j] = a[i] * (1 - a[i]);
			} else {
				deriv[i][j] = -a[i] * a[j];
			}
		}
	}
	return deriv;
}

std::vector<std::vector<std::vector<real_t>>> MLPPActivation::softmaxDeriv(std::vector<std::vector<real_t>> z) {
	MLPPLinAlg alg;
	std::vector<std::vector<std::vector<real_t>>> deriv;
	std::vector<std::vector<real_t>> a = softmax(z);

	deriv.resize(a.size());
	for (int i = 0; i < deriv.size(); i++) {
		deriv[i].resize(a.size());
	}
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < z.size(); j++) {
			if (i == j) {
				deriv[i][j] = alg.subtraction(a[i], alg.hadamard_product(a[i], a[i]));
			} else {
				deriv[i][j] = alg.scalarMultiply(-1, alg.hadamard_product(a[i], a[j]));
			}
		}
	}
	return deriv;
}

real_t MLPPActivation::softplus(real_t z, bool deriv) {
	if (deriv) {
		return sigmoid(z);
	}
	return std::log(1 + exp(z));
}

std::vector<real_t> MLPPActivation::softplus(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		return sigmoid(z);
	}
	MLPPLinAlg alg;
	return alg.log(alg.addition(alg.onevec(z.size()), alg.exp(z)));
}

std::vector<std::vector<real_t>> MLPPActivation::softplus(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		return sigmoid(z);
	}
	MLPPLinAlg alg;
	return alg.log(alg.addition(alg.onemat(z.size(), z[0].size()), alg.exp(z)));
}

real_t MLPPActivation::softsign(real_t z, bool deriv) {
	if (deriv) {
		return 1 / ((1 + abs(z)) * (1 + abs(z)));
	}
	return z / (1 + abs(z));
}

std::vector<real_t> MLPPActivation::softsign(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onevec(z.size()), alg.exponentiate(alg.addition(alg.onevec(z.size()), alg.abs(z)), 2));
	}
	return alg.elementWiseDivision(z, alg.addition(alg.onevec(z.size()), alg.abs(z)));
}

std::vector<std::vector<real_t>> MLPPActivation::softsign(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.exponentiate(alg.addition(alg.onemat(z.size(), z[0].size()), alg.abs(z)), 2));
	}
	return alg.elementWiseDivision(z, alg.addition(alg.onemat(z.size(), z[0].size()), alg.abs(z)));
}

real_t MLPPActivation::gaussianCDF(real_t z, bool deriv) {
	if (deriv) {
		return (1 / sqrt(2 * M_PI)) * exp(-z * z / 2);
	}
	return 0.5 * (1 + erf(z / sqrt(2)));
}

std::vector<real_t> MLPPActivation::gaussianCDF(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.scalarMultiply(1 / sqrt(2 * M_PI), alg.exp(alg.scalarMultiply(-1 / 2, alg.hadamard_product(z, z))));
	}
	return alg.scalarMultiply(0.5, alg.addition(alg.onevec(z.size()), alg.erf(alg.scalarMultiply(1 / sqrt(2), z))));
}

std::vector<std::vector<real_t>> MLPPActivation::gaussianCDF(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.scalarMultiply(1 / sqrt(2 * M_PI), alg.exp(alg.scalarMultiply(-1 / 2, alg.hadamard_product(z, z))));
	}
	return alg.scalarMultiply(0.5, alg.addition(alg.onemat(z.size(), z[0].size()), alg.erf(alg.scalarMultiply(1 / sqrt(2), z))));
}

real_t MLPPActivation::cloglog(real_t z, bool deriv) {
	if (deriv) {
		return exp(z - exp(z));
	}
	return 1 - exp(-exp(z));
}

std::vector<real_t> MLPPActivation::cloglog(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.exp(alg.scalarMultiply(-1, alg.exp(z)));
	}
	return alg.scalarMultiply(-1, alg.scalarAdd(-1, alg.exp(alg.scalarMultiply(-1, alg.exp(z)))));
}

std::vector<std::vector<real_t>> MLPPActivation::cloglog(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.exp(alg.scalarMultiply(-1, alg.exp(z)));
	}
	return alg.scalarMultiply(-1, alg.scalarAdd(-1, alg.exp(alg.scalarMultiply(-1, alg.exp(z)))));
}

real_t MLPPActivation::logit(real_t z, bool deriv) {
	if (deriv) {
		return 1 / z - 1 / (z - 1);
	}
	return std::log(z / (1 - z));
}

std::vector<real_t> MLPPActivation::logit(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.subtraction(alg.elementWiseDivision(alg.onevec(z.size()), z), alg.elementWiseDivision(alg.onevec(z.size()), alg.subtraction(z, alg.onevec(z.size()))));
	}
	return alg.log(alg.elementWiseDivision(z, alg.subtraction(alg.onevec(z.size()), z)));
}

std::vector<std::vector<real_t>> MLPPActivation::logit(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.subtraction(alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), z), alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.subtraction(z, alg.onemat(z.size(), z[0].size()))));
	}
	return alg.log(alg.elementWiseDivision(z, alg.subtraction(alg.onemat(z.size(), z[0].size()), z)));
}

real_t MLPPActivation::unitStep(real_t z, bool deriv) {
	if (deriv) {
		return 0;
	}
	return z < 0 ? 0 : 1;
}

std::vector<real_t> MLPPActivation::unitStep(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		std::vector<real_t> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = unitStep(z[i], 1);
		}
		return deriv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = unitStep(z[i]);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivation::unitStep(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = unitStep(z[i], 1);
		}
		return deriv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = unitStep(z[i]);
	}
	return a;
}

real_t MLPPActivation::swish(real_t z, bool deriv) {
	if (deriv) {
		return swish(z) + sigmoid(z) * (1 - swish(z));
	}
	return z * sigmoid(z);
}

std::vector<real_t> MLPPActivation::swish(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		alg.addition(swish(z), alg.subtraction(sigmoid(z), alg.hadamard_product(sigmoid(z), swish(z))));
	}
	return alg.hadamard_product(z, sigmoid(z));
}

std::vector<std::vector<real_t>> MLPPActivation::swish(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		alg.addition(swish(z), alg.subtraction(sigmoid(z), alg.hadamard_product(sigmoid(z), swish(z))));
	}
	return alg.hadamard_product(z, sigmoid(z));
}

real_t MLPPActivation::mish(real_t z, bool deriv) {
	if (deriv) {
		return sech(softplus(z)) * sech(softplus(z)) * z * sigmoid(z) + mish(z) / z;
	}
	return z * tanh(softplus(z));
}

std::vector<real_t> MLPPActivation::mish(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.addition(alg.hadamard_product(alg.hadamard_product(alg.hadamard_product(sech(softplus(z)), sech(softplus(z))), z), sigmoid(z)), alg.elementWiseDivision(mish(z), z));
	}
	return alg.hadamard_product(z, tanh(softplus(z)));
}

std::vector<std::vector<real_t>> MLPPActivation::mish(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.addition(alg.hadamard_product(alg.hadamard_product(alg.hadamard_product(sech(softplus(z)), sech(softplus(z))), z), sigmoid(z)), alg.elementWiseDivision(mish(z), z));
	}
	return alg.hadamard_product(z, tanh(softplus(z)));
}

real_t MLPPActivation::sinc(real_t z, bool deriv) {
	if (deriv) {
		return (z * std::cos(z) - std::sin(z)) / (z * z);
	}
	return std::sin(z) / z;
}

std::vector<real_t> MLPPActivation::sinc(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.subtraction(alg.hadamard_product(z, alg.cos(z)), alg.sin(z)), alg.hadamard_product(z, z));
	}
	return alg.elementWiseDivision(alg.sin(z), z);
}

std::vector<std::vector<real_t>> MLPPActivation::sinc(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.subtraction(alg.hadamard_product(z, alg.cos(z)), alg.sin(z)), alg.hadamard_product(z, z));
	}
	return alg.elementWiseDivision(alg.sin(z), z);
}

real_t MLPPActivation::RELU(real_t z, bool deriv) {
	if (deriv) {
		if (z <= 0) {
			return 0;
		} else {
			return 1;
		}
	}
	return fmax(0, z);
}

std::vector<real_t> MLPPActivation::RELU(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		std::vector<real_t> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = RELU(z[i], 1);
		}
		return deriv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = RELU(z[i]);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivation::RELU(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = RELU(z[i], 1);
		}
		return deriv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = RELU(z[i]);
	}
	return a;
}

real_t MLPPActivation::leakyReLU(real_t z, real_t c, bool deriv) {
	if (deriv) {
		if (z <= 0) {
			return c;
		} else {
			return 1;
		}
	}
	return fmax(c * z, z);
}

std::vector<real_t> MLPPActivation::leakyReLU(std::vector<real_t> z, real_t c, bool deriv) {
	if (deriv) {
		std::vector<real_t> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = leakyReLU(z[i], c, 1);
		}
		return deriv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = leakyReLU(z[i], c);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivation::leakyReLU(std::vector<std::vector<real_t>> z, real_t c, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = leakyReLU(z[i], c, 1);
		}
		return deriv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = leakyReLU(z[i], c);
	}
	return a;
}

real_t MLPPActivation::ELU(real_t z, real_t c, bool deriv) {
	if (deriv) {
		if (z <= 0) {
			return c * exp(z);
		} else {
			return 1;
		}
	}
	if (z >= 0) {
		return z;
	} else {
		return c * (exp(z) - 1);
	}
}

std::vector<real_t> MLPPActivation::ELU(std::vector<real_t> z, real_t c, bool deriv) {
	if (deriv) {
		std::vector<real_t> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = ELU(z[i], c, 1);
		}
		return deriv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = ELU(z[i], c);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivation::ELU(std::vector<std::vector<real_t>> z, real_t c, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = ELU(z[i], c, 1);
		}
		return deriv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = ELU(z[i], c);
	}
	return a;
}

real_t MLPPActivation::SELU(real_t z, real_t lambda, real_t c, bool deriv) {
	if (deriv) {
		return ELU(z, c, 1);
	}
	return lambda * ELU(z, c);
}

std::vector<real_t> MLPPActivation::SELU(std::vector<real_t> z, real_t lambda, real_t c, bool deriv) {
	if (deriv) {
		std::vector<real_t> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = SELU(z[i], lambda, c, 1);
		}
		return deriv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = SELU(z[i], lambda, c);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivation::SELU(std::vector<std::vector<real_t>> z, real_t lambda, real_t c, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = SELU(z[i], lambda, c, 1);
		}
		return deriv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = SELU(z[i], lambda, c);
	}
	return a;
}

real_t MLPPActivation::GELU(real_t z, bool deriv) {
	if (deriv) {
		return 0.5 * tanh(0.0356774 * std::pow(z, 3) + 0.797885 * z) + (0.0535161 * std::pow(z, 3) + 0.398942 * z) * std::pow(sech(0.0356774 * std::pow(z, 3) + 0.797885 * z), 2) + 0.5;
	}
	return 0.5 * z * (1 + tanh(sqrt(2 / M_PI) * (z + 0.044715 * std::pow(z, 3))));
}

std::vector<real_t> MLPPActivation::GELU(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		std::vector<real_t> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = GELU(z[i], 1);
		}
		return deriv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = GELU(z[i]);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivation::GELU(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = GELU(z[i], 1);
		}
		return deriv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = GELU(z[i]);
	}
	return a;
}

real_t MLPPActivation::sign(real_t z, bool deriv) {
	if (deriv) {
		return 0;
	}
	if (z < 0) {
		return -1;
	} else if (z == 0) {
		return 0;
	} else {
		return 1;
	}
}

std::vector<real_t> MLPPActivation::sign(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		std::vector<real_t> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = sign(z[i], 1);
		}
		return deriv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = sign(z[i]);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivation::sign(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = sign(z[i], 1);
		}
		return deriv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (int i = 0; i < a.size(); i++) {
		a[i] = sign(z[i]);
	}
	return a;
}

real_t MLPPActivation::sinh(real_t z, bool deriv) {
	if (deriv) {
		return cosh(z);
	}
	return 0.5 * (exp(z) - exp(-z));
}

std::vector<real_t> MLPPActivation::sinh(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		return cosh(z);
	}
	MLPPLinAlg alg;
	return alg.scalarMultiply(0.5, alg.subtraction(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
}

std::vector<std::vector<real_t>> MLPPActivation::sinh(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		return cosh(z);
	}
	MLPPLinAlg alg;
	return alg.scalarMultiply(0.5, alg.subtraction(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
}

real_t MLPPActivation::cosh(real_t z, bool deriv) {
	if (deriv) {
		return sinh(z);
	}
	return 0.5 * (exp(z) + exp(-z));
}

std::vector<real_t> MLPPActivation::cosh(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		return sinh(z);
	}
	MLPPLinAlg alg;
	return alg.scalarMultiply(0.5, alg.addition(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
}

std::vector<std::vector<real_t>> MLPPActivation::cosh(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		return sinh(z);
	}
	MLPPLinAlg alg;
	return alg.scalarMultiply(0.5, alg.addition(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
}

real_t MLPPActivation::tanh(real_t z, bool deriv) {
	if (deriv) {
		return 1 - tanh(z) * tanh(z);
	}
	return (exp(z) - exp(-z)) / (exp(z) + exp(-z));
}

std::vector<real_t> MLPPActivation::tanh(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.scalarMultiply(-1, alg.scalarAdd(-1, alg.hadamard_product(tanh(z), tanh(z))));
	}
	return alg.elementWiseDivision(alg.subtraction(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))), alg.addition(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
}

std::vector<std::vector<real_t>> MLPPActivation::tanh(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.scalarMultiply(-1, alg.scalarAdd(-1, alg.hadamard_product(tanh(z), tanh(z))));
	}

	return alg.elementWiseDivision(alg.subtraction(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))), alg.addition(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
}

real_t MLPPActivation::csch(real_t z, bool deriv) {
	if (deriv) {
		return -csch(z) * coth(z);
	}
	return 1 / sinh(z);
}

std::vector<real_t> MLPPActivation::csch(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.hadamard_product(alg.scalarMultiply(-1, csch(z)), coth(z));
	}
	return alg.elementWiseDivision(alg.onevec(z.size()), sinh(z));
}

std::vector<std::vector<real_t>> MLPPActivation::csch(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.hadamard_product(alg.scalarMultiply(-1, csch(z)), coth(z));
	}
	return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), sinh(z));
}

real_t MLPPActivation::sech(real_t z, bool deriv) {
	if (deriv) {
		return -sech(z) * tanh(z);
	}
	return 1 / cosh(z);
}

std::vector<real_t> MLPPActivation::sech(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.hadamard_product(alg.scalarMultiply(-1, sech(z)), tanh(z));
	}
	return alg.elementWiseDivision(alg.onevec(z.size()), cosh(z));

	// return activation(z, deriv, static_cast<void (*)(real_t, bool)>(&sech));
}

std::vector<std::vector<real_t>> MLPPActivation::sech(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.hadamard_product(alg.scalarMultiply(-1, sech(z)), tanh(z));
	}
	return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), cosh(z));

	// return activation(z, deriv, static_cast<void (*)(real_t, bool)>(&sech));
}

real_t MLPPActivation::coth(real_t z, bool deriv) {
	if (deriv) {
		return -csch(z) * csch(z);
	}
	return 1 / tanh(z);
}

std::vector<real_t> MLPPActivation::coth(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.hadamard_product(alg.scalarMultiply(-1, csch(z)), csch(z));
	}
	return alg.elementWiseDivision(alg.onevec(z.size()), tanh(z));
}

std::vector<std::vector<real_t>> MLPPActivation::coth(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.hadamard_product(alg.scalarMultiply(-1, csch(z)), csch(z));
	}
	return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), tanh(z));
}

real_t MLPPActivation::arsinh(real_t z, bool deriv) {
	if (deriv) {
		return 1 / sqrt(z * z + 1);
	}
	return std::log(z + sqrt(z * z + 1));
}

std::vector<real_t> MLPPActivation::arsinh(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onevec(z.size()), alg.sqrt(alg.addition(alg.hadamard_product(z, z), alg.onevec(z.size()))));
	}
	return alg.log(alg.addition(z, alg.sqrt(alg.addition(alg.hadamard_product(z, z), alg.onevec(z.size())))));
}

std::vector<std::vector<real_t>> MLPPActivation::arsinh(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.sqrt(alg.addition(alg.hadamard_product(z, z), alg.onemat(z.size(), z[0].size()))));
	}
	return alg.log(alg.addition(z, alg.sqrt(alg.addition(alg.hadamard_product(z, z), alg.onemat(z.size(), z[0].size())))));
}

real_t MLPPActivation::arcosh(real_t z, bool deriv) {
	if (deriv) {
		return 1 / sqrt(z * z - 1);
	}
	return std::log(z + sqrt(z * z - 1));
}

std::vector<real_t> MLPPActivation::arcosh(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onevec(z.size()), alg.sqrt(alg.subtraction(alg.hadamard_product(z, z), alg.onevec(z.size()))));
	}
	return alg.log(alg.addition(z, alg.sqrt(alg.subtraction(alg.hadamard_product(z, z), alg.onevec(z.size())))));
}

std::vector<std::vector<real_t>> MLPPActivation::arcosh(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.sqrt(alg.subtraction(alg.hadamard_product(z, z), alg.onemat(z.size(), z[0].size()))));
	}
	return alg.log(alg.addition(z, alg.sqrt(alg.subtraction(alg.hadamard_product(z, z), alg.onemat(z.size(), z[0].size())))));
}

real_t MLPPActivation::artanh(real_t z, bool deriv) {
	if (deriv) {
		return 1 / (1 - z * z);
	}
	return 0.5 * std::log((1 + z) / (1 - z));
}

std::vector<real_t> MLPPActivation::artanh(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onevec(z.size()), alg.subtraction(alg.onevec(z.size()), alg.hadamard_product(z, z)));
	}
	return alg.scalarMultiply(0.5, alg.log(alg.elementWiseDivision(alg.addition(alg.onevec(z.size()), z), alg.subtraction(alg.onevec(z.size()), z))));
}

std::vector<std::vector<real_t>> MLPPActivation::artanh(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.subtraction(alg.onemat(z.size(), z[0].size()), alg.hadamard_product(z, z)));
	}
	return alg.scalarMultiply(0.5, alg.log(alg.elementWiseDivision(alg.addition(alg.onemat(z.size(), z[0].size()), z), alg.subtraction(alg.onemat(z.size(), z[0].size()), z))));
}

real_t MLPPActivation::arcsch(real_t z, bool deriv) {
	if (deriv) {
		return -1 / ((z * z) * sqrt(1 + (1 / (z * z))));
	}
	return std::log(sqrt(1 + (1 / (z * z))) + (1 / z));
}

std::vector<real_t> MLPPActivation::arcsch(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.full(z.size(), -1), alg.hadamard_product(alg.hadamard_product(z, z), alg.sqrt(alg.addition(alg.onevec(z.size()), alg.elementWiseDivision(alg.onevec(z.size()), alg.hadamard_product(z, z))))));
	}
	return alg.log(alg.addition(alg.sqrt(alg.addition(alg.onevec(z.size()), alg.elementWiseDivision(alg.onevec(z.size()), alg.hadamard_product(z, z)))), alg.elementWiseDivision(alg.onevec(z.size()), z)));
}

std::vector<std::vector<real_t>> MLPPActivation::arcsch(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.full(z.size(), z[0].size(), -1), alg.hadamard_product(alg.hadamard_product(z, z), alg.sqrt(alg.addition(alg.onemat(z.size(), z[0].size()), alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.hadamard_product(z, z))))));
	}
	return alg.log(alg.addition(alg.sqrt(alg.addition(alg.onemat(z.size(), z[0].size()), alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.hadamard_product(z, z)))), alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), z)));
}

real_t MLPPActivation::arsech(real_t z, bool deriv) {
	if (deriv) {
		return -1 / (z * sqrt(1 - z * z));
	}
	return std::log((1 / z) + ((1 / z) + 1) * ((1 / z) - 1));
}

std::vector<real_t> MLPPActivation::arsech(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.full(z.size(), -1), alg.hadamard_product(z, alg.sqrt(alg.subtraction(alg.onevec(z.size()), alg.hadamard_product(z, z)))));
	}
	return alg.log(alg.addition(alg.elementWiseDivision(alg.onevec(z.size()), z), alg.hadamard_product(alg.addition(alg.elementWiseDivision(alg.onevec(z.size()), z), alg.onevec(z.size())), alg.subtraction(alg.elementWiseDivision(alg.onevec(z.size()), z), alg.onevec(z.size())))));
}

std::vector<std::vector<real_t>> MLPPActivation::arsech(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.full(z.size(), z[0].size(), -1), alg.hadamard_product(z, alg.sqrt(alg.subtraction(alg.onemat(z.size(), z[0].size()), alg.hadamard_product(z, z)))));
	}
	return alg.log(alg.addition(alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), z), alg.hadamard_product(alg.addition(alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), z), alg.onemat(z.size(), z[0].size())), alg.subtraction(alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), z), alg.onemat(z.size(), z[0].size())))));
}

real_t MLPPActivation::arcoth(real_t z, bool deriv) {
	if (deriv) {
		return 1 / (1 - z * z);
	}
	return 0.5 * std::log((1 + z) / (z - 1));
}

std::vector<real_t> MLPPActivation::arcoth(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onevec(z.size()), alg.subtraction(alg.onevec(z.size()), alg.hadamard_product(z, z)));
	}
	return alg.scalarMultiply(0.5, alg.log(alg.elementWiseDivision(alg.addition(alg.onevec(z.size()), z), alg.subtraction(z, alg.onevec(z.size())))));
}

std::vector<std::vector<real_t>> MLPPActivation::arcoth(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.subtraction(alg.onemat(z.size(), z[0].size()), alg.hadamard_product(z, z)));
	}
	return alg.scalarMultiply(0.5, alg.log(alg.elementWiseDivision(alg.addition(alg.onemat(z.size(), z[0].size()), z), alg.subtraction(z, alg.onemat(z.size(), z[0].size())))));
}

// TO DO: Implement this template activation
std::vector<real_t> MLPPActivation::activation(std::vector<real_t> z, bool deriv, real_t (*function)(real_t, bool)) {
	if (deriv) {
		std::vector<real_t> deriv;
		deriv.resize(z.size());
		for (int i = 0; i < z.size(); i++) {
			deriv[i] = function(z[i], 1);
		}
		return deriv;
	}
	std::vector<real_t> a;
	a.resize(z.size());
	for (int i = 0; i < z.size(); i++) {
		a[i] = function(z[i], deriv);
	}
	return a;
}
