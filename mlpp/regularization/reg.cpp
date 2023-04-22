//
//  Reg.cpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "reg.h"

#include "core/math/math_defs.h"

#include "../activation/activation.h"
#include "../lin_alg/lin_alg.h"

#include <iostream>
#include <random>

real_t MLPPReg::reg_termv(const Ref<MLPPVector> &weights, real_t lambda, real_t alpha, MLPPReg::RegularizationType p_reg) {
	int size = weights->size();
	const real_t *weights_ptr = weights->ptr();

	if (p_reg == REGULARIZATION_TYPE_RIDGE) {
		real_t reg = 0;
		for (int i = 0; i < size; ++i) {
			real_t wi = weights_ptr[i];
			reg += wi * wi;
		}
		return reg * lambda / 2;
	} else if (p_reg == REGULARIZATION_TYPE_LASSO) {
		real_t reg = 0;
		for (int i = 0; i < size; ++i) {
			reg += ABS(weights_ptr[i]);
		}
		return reg * lambda;
	} else if (p_reg == REGULARIZATION_TYPE_ELASTIC_NET) {
		real_t reg = 0;
		for (int i = 0; i < size; ++i) {
			real_t wi = weights_ptr[i];
			reg += alpha * ABS(wi); // Lasso Reg
			reg += ((1 - alpha) / 2) * wi * wi; // Ridge Reg
		}
		return reg * lambda;
	}

	return 0;
}
real_t MLPPReg::reg_termm(const Ref<MLPPMatrix> &weights, real_t lambda, real_t alpha, MLPPReg::RegularizationType p_reg) {
	int size = weights->data_size();
	const real_t *weights_ptr = weights->ptr();

	if (p_reg == REGULARIZATION_TYPE_RIDGE) {
		real_t reg = 0;
		for (int i = 0; i < size; ++i) {
			real_t wi = weights_ptr[i];
			reg += wi * wi;
		}
		return reg * lambda / 2;
	} else if (p_reg == REGULARIZATION_TYPE_LASSO) {
		real_t reg = 0;
		for (int i = 0; i < size; ++i) {
			reg += ABS(weights_ptr[i]);
		}
		return reg * lambda;
	} else if (p_reg == REGULARIZATION_TYPE_ELASTIC_NET) {
		real_t reg = 0;
		for (int i = 0; i < size; ++i) {
			real_t wi = weights_ptr[i];
			reg += alpha * ABS(wi); // Lasso Reg
			reg += ((1 - alpha) / 2) * wi * wi; // Ridge Reg
		}
		return reg * lambda;
	}

	return 0;
}

Ref<MLPPVector> MLPPReg::reg_weightsv(const Ref<MLPPVector> &weights, real_t lambda, real_t alpha, MLPPReg::RegularizationType p_reg) {
	MLPPLinAlg alg;

	if (p_reg == REGULARIZATION_TYPE_WEIGHT_CLIPPING) {
		return reg_deriv_termv(weights, lambda, alpha, p_reg);
	}

	return alg.subtractionnv(weights, reg_deriv_termv(weights, lambda, alpha, p_reg));

	// for(int i = 0; i < weights.size(); i++){
	//     weights[i] -= regDerivTerm(weights, lambda, alpha, reg, i);
	// }
	// return weights;
}
Ref<MLPPMatrix> MLPPReg::reg_weightsm(const Ref<MLPPMatrix> &weights, real_t lambda, real_t alpha, MLPPReg::RegularizationType reg) {
	MLPPLinAlg alg;

	if (reg == REGULARIZATION_TYPE_WEIGHT_CLIPPING) {
		return reg_deriv_termm(weights, lambda, alpha, reg);
	}

	return alg.subtractionnm(weights, reg_deriv_termm(weights, lambda, alpha, reg));

	// for(int i = 0; i < weights.size(); i++){
	//     for(int j = 0; j < weights[i].size(); j++){
	//         weights[i][j] -= regDerivTerm(weights, lambda, alpha, reg, i, j);
	//     }
	// }
	// return weights;
}

Ref<MLPPVector> MLPPReg::reg_deriv_termv(const Ref<MLPPVector> &weights, real_t lambda, real_t alpha, MLPPReg::RegularizationType reg) {
	Ref<MLPPVector> reg_driv;
	reg_driv.instance();

	int size = weights->size();

	reg_driv->resize(size);

	real_t *reg_driv_ptr = reg_driv->ptrw();

	for (int i = 0; i < size; ++i) {
		reg_driv_ptr[i] = reg_deriv_termvr(weights, lambda, alpha, reg, i);
	}

	return reg_driv;
}
Ref<MLPPMatrix> MLPPReg::reg_deriv_termm(const Ref<MLPPMatrix> &weights, real_t lambda, real_t alpha, MLPPReg::RegularizationType reg) {
	Ref<MLPPMatrix> reg_driv;
	reg_driv.instance();

	Size2i size = weights->size();

	reg_driv->resize(size);

	real_t *reg_driv_ptr = reg_driv->ptrw();

	for (int i = 0; i < size.y; ++i) {
		for (int j = 0; j < size.x; ++j) {
			reg_driv_ptr[reg_driv->calculate_index(i, j)] = reg_deriv_termmr(weights, lambda, alpha, reg, i, j);
		}
	}

	return reg_driv;
}

MLPPReg::MLPPReg() {
}
MLPPReg::~MLPPReg() {
}

void MLPPReg::_bind_methods() {
	ClassDB::bind_method(D_METHOD("reg_termv", "weights", "lambda", "alpha", "reg"), &MLPPReg::reg_termv);
	ClassDB::bind_method(D_METHOD("reg_termm", "weights", "lambda", "alpha", "reg"), &MLPPReg::reg_termm);

	ClassDB::bind_method(D_METHOD("reg_weightsv", "weights", "lambda", "alpha", "reg"), &MLPPReg::reg_weightsv);
	ClassDB::bind_method(D_METHOD("reg_weightsm", "weights", "lambda", "alpha", "reg"), &MLPPReg::reg_weightsm);

	ClassDB::bind_method(D_METHOD("reg_deriv_termv", "weights", "lambda", "alpha", "reg"), &MLPPReg::reg_deriv_termv);
	ClassDB::bind_method(D_METHOD("reg_deriv_termm", "weights", "lambda", "alpha", "reg"), &MLPPReg::reg_deriv_termm);

	BIND_ENUM_CONSTANT(REGULARIZATION_TYPE_NONE);
	BIND_ENUM_CONSTANT(REGULARIZATION_TYPE_RIDGE);
	BIND_ENUM_CONSTANT(REGULARIZATION_TYPE_LASSO);
	BIND_ENUM_CONSTANT(REGULARIZATION_TYPE_ELASTIC_NET);
	BIND_ENUM_CONSTANT(REGULARIZATION_TYPE_WEIGHT_CLIPPING);
}

real_t MLPPReg::reg_deriv_termvr(const Ref<MLPPVector> &weights, real_t lambda, real_t alpha, MLPPReg::RegularizationType reg, int j) {
	MLPPActivation act;

	real_t wj = weights->get_element(j);

	if (reg == REGULARIZATION_TYPE_RIDGE) {
		return lambda * wj;
	} else if (reg == REGULARIZATION_TYPE_LASSO) {
		return lambda * act.sign_normr(wj);
	} else if (reg == REGULARIZATION_TYPE_ELASTIC_NET) {
		return alpha * lambda * act.sign_normr(wj) + (1 - alpha) * lambda * wj;
	} else if (reg == REGULARIZATION_TYPE_WEIGHT_CLIPPING) { // Preparation for Wasserstein GANs.
		// We assume lambda is the lower clipping threshold, while alpha is the higher clipping threshold.
		// alpha > lambda.
		if (wj > alpha) {
			return alpha;
		} else if (wj < lambda) {
			return lambda;
		} else {
			return wj;
		}
	} else {
		return 0;
	}
}
real_t MLPPReg::reg_deriv_termmr(const Ref<MLPPMatrix> &weights, real_t lambda, real_t alpha, MLPPReg::RegularizationType reg, int i, int j) {
	MLPPActivation act;

	real_t wj = weights->get_element(i, j);

	if (reg == REGULARIZATION_TYPE_RIDGE) {
		return lambda * wj;
	} else if (reg == REGULARIZATION_TYPE_LASSO) {
		return lambda * act.sign_normr(wj);
	} else if (reg == REGULARIZATION_TYPE_ELASTIC_NET) {
		return alpha * lambda * act.sign_normr(wj) + (1 - alpha) * lambda * wj;
	} else if (reg == REGULARIZATION_TYPE_WEIGHT_CLIPPING) { // Preparation for Wasserstein GANs.
		// We assume lambda is the lower clipping threshold, while alpha is the higher clipping threshold.
		// alpha > lambda.
		if (wj > alpha) {
			return alpha;
		} else if (wj < lambda) {
			return lambda;
		} else {
			return wj;
		}
	} else {
		return 0;
	}
}
