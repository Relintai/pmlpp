//
//  Reg.cpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "reg_old.h"

#include "core/math/math_defs.h"

#include "../activation/activation_old.h"
#include "../lin_alg/lin_alg_old.h"

#include <iostream>
#include <random>

real_t MLPPRegOld::regTerm(std::vector<real_t> weights, real_t lambda, real_t alpha, std::string p_reg) {
	if (p_reg == "Ridge") {
		real_t reg = 0;
		for (uint32_t i = 0; i < weights.size(); i++) {
			reg += weights[i] * weights[i];
		}
		return reg * lambda / 2;
	} else if (p_reg == "Lasso") {
		real_t reg = 0;
		for (uint32_t i = 0; i < weights.size(); i++) {
			reg += abs(weights[i]);
		}
		return reg * lambda;
	} else if (p_reg == "ElasticNet") {
		real_t reg = 0;
		for (uint32_t i = 0; i < weights.size(); i++) {
			reg += alpha * abs(weights[i]); // Lasso Reg
			reg += ((1 - alpha) / 2) * weights[i] * weights[i]; // Ridge Reg
		}
		return reg * lambda;
	}
	return 0;
}

real_t MLPPRegOld::regTerm(std::vector<std::vector<real_t>> weights, real_t lambda, real_t alpha, std::string p_reg) {
	if (p_reg == "Ridge") {
		real_t reg = 0;
		for (uint32_t i = 0; i < weights.size(); i++) {
			for (uint32_t j = 0; j < weights[i].size(); j++) {
				reg += weights[i][j] * weights[i][j];
			}
		}
		return reg * lambda / 2;
	} else if (p_reg == "Lasso") {
		real_t reg = 0;
		for (uint32_t i = 0; i < weights.size(); i++) {
			for (uint32_t j = 0; j < weights[i].size(); j++) {
				reg += abs(weights[i][j]);
			}
		}
		return reg * lambda;
	} else if (p_reg == "ElasticNet") {
		real_t reg = 0;
		for (uint32_t i = 0; i < weights.size(); i++) {
			for (uint32_t j = 0; j < weights[i].size(); j++) {
				reg += alpha * abs(weights[i][j]); // Lasso Reg
				reg += ((1 - alpha) / 2) * weights[i][j] * weights[i][j]; // Ridge Reg
			}
		}
		return reg * lambda;
	}
	return 0;
}

std::vector<real_t> MLPPRegOld::regWeights(std::vector<real_t> weights, real_t lambda, real_t alpha, std::string reg) {
	MLPPLinAlgOld alg;
	if (reg == "WeightClipping") {
		return regDerivTerm(weights, lambda, alpha, reg);
	}
	return alg.subtraction(weights, regDerivTerm(weights, lambda, alpha, reg));
	// for(int i = 0; i < weights.size(); i++){
	//     weights[i] -= regDerivTerm(weights, lambda, alpha, reg, i);
	// }
	// return weights;
}

std::vector<std::vector<real_t>> MLPPRegOld::regWeights(std::vector<std::vector<real_t>> weights, real_t lambda, real_t alpha, std::string reg) {
	MLPPLinAlgOld alg;
	if (reg == "WeightClipping") {
		return regDerivTerm(weights, lambda, alpha, reg);
	}
	return alg.subtraction(weights, regDerivTerm(weights, lambda, alpha, reg));
	// for(int i = 0; i < weights.size(); i++){
	//     for(int j = 0; j < weights[i].size(); j++){
	//         weights[i][j] -= regDerivTerm(weights, lambda, alpha, reg, i, j);
	//     }
	// }
	// return weights;
}

std::vector<real_t> MLPPRegOld::regDerivTerm(std::vector<real_t> weights, real_t lambda, real_t alpha, std::string reg) {
	std::vector<real_t> regDeriv;
	regDeriv.resize(weights.size());

	for (uint32_t i = 0; i < regDeriv.size(); i++) {
		regDeriv[i] = regDerivTerm(weights, lambda, alpha, reg, i);
	}
	return regDeriv;
}

std::vector<std::vector<real_t>> MLPPRegOld::regDerivTerm(std::vector<std::vector<real_t>> weights, real_t lambda, real_t alpha, std::string reg) {
	std::vector<std::vector<real_t>> regDeriv;
	regDeriv.resize(weights.size());
	for (uint32_t i = 0; i < regDeriv.size(); i++) {
		regDeriv[i].resize(weights[0].size());
	}

	for (uint32_t i = 0; i < regDeriv.size(); i++) {
		for (uint32_t j = 0; j < regDeriv[i].size(); j++) {
			regDeriv[i][j] = regDerivTerm(weights, lambda, alpha, reg, i, j);
		}
	}
	return regDeriv;
}

real_t MLPPRegOld::regDerivTerm(std::vector<real_t> weights, real_t lambda, real_t alpha, std::string reg, int j) {
	MLPPActivationOld act;
	if (reg == "Ridge") {
		return lambda * weights[j];
	} else if (reg == "Lasso") {
		return lambda * act.sign(weights[j]);
	} else if (reg == "ElasticNet") {
		return alpha * lambda * act.sign(weights[j]) + (1 - alpha) * lambda * weights[j];
	} else if (reg == "WeightClipping") { // Preparation for Wasserstein GANs.
		// We assume lambda is the lower clipping threshold, while alpha is the higher clipping threshold.
		// alpha > lambda.
		if (weights[j] > alpha) {
			return alpha;
		} else if (weights[j] < lambda) {
			return lambda;
		} else {
			return weights[j];
		}
	} else {
		return 0;
	}
}

real_t MLPPRegOld::regDerivTerm(std::vector<std::vector<real_t>> weights, real_t lambda, real_t alpha, std::string reg, int i, int j) {
	MLPPActivationOld act;
	if (reg == "Ridge") {
		return lambda * weights[i][j];
	} else if (reg == "Lasso") {
		return lambda * act.sign(weights[i][j]);
	} else if (reg == "ElasticNet") {
		return alpha * lambda * act.sign(weights[i][j]) + (1 - alpha) * lambda * weights[i][j];
	} else if (reg == "WeightClipping") { // Preparation for Wasserstein GANs.
		// We assume lambda is the lower clipping threshold, while alpha is the higher clipping threshold.
		// alpha > lambda.
		if (weights[i][j] > alpha) {
			return alpha;
		} else if (weights[i][j] < lambda) {
			return lambda;
		} else {
			return weights[i][j];
		}
	} else {
		return 0;
	}
}
