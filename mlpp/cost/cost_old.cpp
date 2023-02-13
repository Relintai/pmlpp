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

real_t MLPPCostOld::msev(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	int y_hat_size = y_hat->size();

	ERR_FAIL_COND_V(y_hat_size != y->size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_size; ++i) {
		sum += (y_hat_ptr[i] - y_ptr[i]) * (y_hat_ptr[i] - y_ptr[i]);
	}

	return sum / 2 * y_hat_size;
}
real_t MLPPCostOld::msem(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	int y_hat_data_size = y_hat->data_size();

	ERR_FAIL_COND_V(y_hat_data_size != y->data_size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_data_size; ++i) {
		sum += (y_hat_ptr[i] - y_ptr[i]) * (y_hat_ptr[i] - y_ptr[i]);
	}

	return sum / 2.0 * static_cast<real_t>(y_hat_data_size);
}

Ref<MLPPVector> MLPPCostOld::mse_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;
	return alg.subtractionnv(y_hat, y);
}

Ref<MLPPMatrix> MLPPCostOld::mse_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;
	return alg.subtractionm(y_hat, y);
}

real_t MLPPCostOld::rmsev(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	int y_hat_size = y_hat->size();

	ERR_FAIL_COND_V(y_hat_size != y->size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_size; ++i) {
		sum += (y_hat_ptr[i] - y_ptr[i]) * (y_hat_ptr[i] - y_ptr[i]);
	}

	return Math::sqrt(sum / static_cast<real_t>(y_hat_size));
}
real_t MLPPCostOld::rmsem(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	int y_hat_data_size = y_hat->data_size();

	ERR_FAIL_COND_V(y_hat_data_size != y->data_size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_data_size; ++i) {
		sum += (y_hat_ptr[i] - y_ptr[i]) * (y_hat_ptr[i] - y_ptr[i]);
	}

	return Math::sqrt(sum / static_cast<real_t>(y_hat->size().y));
}

Ref<MLPPVector> MLPPCostOld::rmse_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(1 / (2.0 * Math::sqrt(msev(y_hat, y))), mse_derivv(y_hat, y));
}
Ref<MLPPMatrix> MLPPCostOld::rmse_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;

	return alg.scalar_multiplym(1 / (2.0 / Math::sqrt(msem(y_hat, y))), mse_derivm(y_hat, y));
}

real_t MLPPCostOld::maev(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	int y_hat_size = y_hat->size();

	ERR_FAIL_COND_V(y_hat_size != y->size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_size; i++) {
		sum += ABS((y_hat_ptr[i] - y_ptr[i]));
	}
	return sum / static_cast<real_t>(y_hat_size);
}
real_t MLPPCostOld::maem(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	int y_hat_data_size = y_hat->data_size();

	ERR_FAIL_COND_V(y_hat_data_size != y->data_size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_data_size; ++i) {
		sum += ABS((y_hat_ptr[i] - y_ptr[i]));
	}

	return sum / static_cast<real_t>(y_hat_data_size);
}

Ref<MLPPVector> MLPPCostOld::mae_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	int y_hat_size = y_hat->size();

	const real_t *y_hat_ptr = y_hat->ptr();

	Ref<MLPPVector> deriv;
	deriv.instance();
	deriv->resize(y_hat_size);
	real_t *deriv_ptr = deriv->ptrw();

	for (int i = 0; i < y_hat_size; ++i) {
		int y_hat_ptr_i = y_hat_ptr[i];

		if (y_hat_ptr_i < 0) {
			deriv_ptr[i] = -1;
		} else if (y_hat_ptr_i == 0) {
			deriv_ptr[i] = 0;
		} else {
			deriv_ptr[i] = 1;
		}
	}

	return deriv;
}
Ref<MLPPMatrix> MLPPCostOld::mae_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	int y_hat_data_size = y_hat->data_size();

	const real_t *y_hat_ptr = y_hat->ptr();

	Ref<MLPPMatrix> deriv;
	deriv.instance();
	deriv->resize(y_hat->size());
	real_t *deriv_ptr = deriv->ptrw();

	for (int i = 0; i < y_hat_data_size; ++i) {
		int y_hat_ptr_i = y_hat_ptr[i];

		if (y_hat_ptr_i < 0) {
			deriv_ptr[i] = -1;
		} else if (y_hat_ptr_i == 0) {
			deriv_ptr[i] = 0;
		} else {
			deriv_ptr[i] = 1;
		}
	}

	return deriv;
}

real_t MLPPCostOld::mbev(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	int y_hat_size = y_hat->size();

	ERR_FAIL_COND_V(y_hat_size != y->size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_size; ++i) {
		sum += (y_hat_ptr[i] - y_ptr[i]);
	}

	return sum / static_cast<real_t>(y_hat_size);
}
real_t MLPPCostOld::mbem(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	int y_hat_data_size = y_hat->data_size();

	ERR_FAIL_COND_V(y_hat_data_size != y->data_size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_data_size; ++i) {
		sum += (y_hat_ptr[i] - y_ptr[i]);
	}

	return sum / static_cast<real_t>(y_hat_data_size);
}

Ref<MLPPVector> MLPPCostOld::mbe_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;
	return alg.onevecv(y_hat->size());
}
Ref<MLPPMatrix> MLPPCostOld::mbe_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;
	return alg.onematm(y_hat->size().x, y_hat->size().y);
}

// Classification Costs
real_t MLPPCostOld::log_lossv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	int y_hat_size = y_hat->size();

	ERR_FAIL_COND_V(y_hat_size != y->size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	real_t eps = 1e-8;
	for (int i = 0; i < y_hat_size; ++i) {
		sum += -(y_ptr[i] * Math::log(y_hat_ptr[i] + eps) + (1 - y_ptr[i]) * Math::log(1 - y_hat_ptr[i] + eps));
	}

	return sum / static_cast<real_t>(y_hat_size);
}

real_t MLPPCostOld::log_lossm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	int y_hat_data_size = y_hat->data_size();

	ERR_FAIL_COND_V(y_hat_data_size != y->data_size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	real_t eps = 1e-8;
	for (int i = 0; i < y_hat_data_size; ++i) {
		sum += -(y_ptr[i] * Math::log(y_hat_ptr[i] + eps) + (1 - y_ptr[i]) * Math::log(1 - y_hat_ptr[i] + eps));
	}

	return sum / static_cast<real_t>(y_hat_data_size);
}

Ref<MLPPVector> MLPPCostOld::log_loss_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;
	return alg.additionnv(
			alg.scalar_multiplynv(-1, alg.element_wise_division(y, y_hat)),
			alg.element_wise_division(alg.scalar_multiplynv(-1, alg.scalar_addnv(-1, y)), alg.scalar_multiplynv(-1, alg.scalar_addnv(-1, y_hat))));
}

Ref<MLPPMatrix> MLPPCostOld::log_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;
	return alg.additionm(
			alg.scalar_multiplym(-1, alg.element_wise_divisionm(y, y_hat)),
			alg.element_wise_divisionm(alg.scalar_multiplym(-1, alg.scalar_addm(-1, y)), alg.scalar_multiplym(-1, alg.scalar_addm(-1, y_hat))));
}

real_t MLPPCostOld::cross_entropyv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	int y_hat_size = y_hat->size();

	ERR_FAIL_COND_V(y_hat_size != y->size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_size; ++i) {
		sum += y_ptr[i] * Math::log(y_hat_ptr[i]);
	}

	return -1 * sum;
}
real_t MLPPCostOld::cross_entropym(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	int y_hat_data_size = y_hat->data_size();

	ERR_FAIL_COND_V(y_hat_data_size != y->data_size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_data_size; ++i) {
		sum += y_ptr[i] * Math::log(y_hat_ptr[i]);
	}

	return -1 * sum;
}

Ref<MLPPVector> MLPPCostOld::cross_entropy_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;
	return alg.scalar_multiplynv(-1, alg.element_wise_division(y, y_hat));
}
Ref<MLPPMatrix> MLPPCostOld::cross_entropy_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;
	return alg.scalar_multiplym(-1, alg.element_wise_divisionm(y, y_hat));
}

real_t MLPPCostOld::huber_lossv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y, real_t delta) {
	int y_hat_size = y_hat->size();

	ERR_FAIL_COND_V(y_hat_size != y->size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	MLPPLinAlg alg;
	real_t sum = 0;
	for (int i = 0; i < y_hat_size; ++i) {
		if (ABS(y_ptr[i] - y_hat_ptr[i]) <= delta) {
			sum += (y_ptr[i] - y_hat_ptr[i]) * (y_ptr[i] - y_hat_ptr[i]);
		} else {
			sum += 2 * delta * ABS(y_ptr[i] - y_hat_ptr[i]) - delta * delta;
		}
	}

	return sum;
}
real_t MLPPCostOld::huber_lossm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y, real_t delta) {
	int y_hat_data_size = y_hat->data_size();

	ERR_FAIL_COND_V(y_hat_data_size != y->data_size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	MLPPLinAlg alg;
	real_t sum = 0;
	for (int i = 0; i < y_hat_data_size; ++i) {
		if (ABS(y_ptr[i] - y_hat_ptr[i]) <= delta) {
			sum += (y_ptr[i] - y_hat_ptr[i]) * (y_ptr[i] - y_hat_ptr[i]);
		} else {
			sum += 2 * delta * ABS(y_ptr[i] - y_hat_ptr[i]) - delta * delta;
		}
	}

	return sum;
}

Ref<MLPPVector> MLPPCostOld::huber_loss_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y, real_t delta) {
	int y_hat_size = y_hat->size();

	ERR_FAIL_COND_V(y_hat_size != y->size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	MLPPLinAlg alg;

	Ref<MLPPVector> deriv;
	deriv.instance();
	deriv->resize(y_hat->size());

	real_t *deriv_ptr = deriv->ptrw();

	for (int i = 0; i < y_hat_size; ++i) {
		if (ABS(y_ptr[i] - y_hat_ptr[i]) <= delta) {
			deriv_ptr[i] = (-(y_ptr[i] - y_hat_ptr[i]));
		} else {
			if (y_hat_ptr[i] > 0 || y_hat_ptr[i] < 0) {
				deriv_ptr[i] = (2 * delta * (y_hat_ptr[i] / ABS(y_hat_ptr[i])));
			} else {
				deriv_ptr[i] = (0);
			}
		}
	}

	return deriv;
}
Ref<MLPPMatrix> MLPPCostOld::huber_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y, real_t delta) {
	int y_hat_data_size = y_hat->data_size();

	ERR_FAIL_COND_V(y_hat_data_size != y->data_size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	MLPPLinAlg alg;

	Ref<MLPPMatrix> deriv;
	deriv.instance();
	deriv->resize(y_hat->size());

	real_t *deriv_ptr = deriv->ptrw();

	for (int i = 0; i < y_hat_data_size; ++i) {
		if (ABS(y_ptr[i] - y_hat_ptr[i]) <= delta) {
			deriv_ptr[i] = (-(y_ptr[i] - y_hat_ptr[i]));
		} else {
			if (y_hat_ptr[i] > 0 || y_hat_ptr[i] < 0) {
				deriv_ptr[i] = (2 * delta * (y_hat_ptr[i] / ABS(y_hat_ptr[i])));
			} else {
				deriv_ptr[i] = (0);
			}
		}
	}

	return deriv;
}

real_t MLPPCostOld::hinge_lossv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	int y_hat_size = y_hat->size();

	ERR_FAIL_COND_V(y_hat_size != y->size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_size; ++i) {
		sum += MAX(0, 1 - y_ptr[i] * y_hat_ptr[i]);
	}

	return sum / static_cast<real_t>(y_hat_size);
}
real_t MLPPCostOld::hinge_lossm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	int y_hat_data_size = y_hat->data_size();

	ERR_FAIL_COND_V(y_hat_data_size != y->data_size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_data_size; ++i) {
		sum += MAX(0, 1 - y_ptr[i] * y_hat_ptr[i]);
	}

	return sum / static_cast<real_t>(y_hat_data_size);
}

Ref<MLPPVector> MLPPCostOld::hinge_loss_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	int y_hat_size = y_hat->size();

	ERR_FAIL_COND_V(y_hat_size != y->size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	Ref<MLPPVector> deriv;
	deriv.instance();
	deriv->resize(y_hat->size());

	real_t *deriv_ptr = deriv->ptrw();

	for (int i = 0; i < y_hat_size; ++i) {
		if (1 - y_ptr[i] * y_hat_ptr[i] > 0) {
			deriv_ptr[i] = -y_ptr[i];
		} else {
			deriv_ptr[i] = 0;
		}
	}

	return deriv;
}
Ref<MLPPMatrix> MLPPCostOld::hinge_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	int y_hat_data_size = y_hat->data_size();

	ERR_FAIL_COND_V(y_hat_data_size != y->data_size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	Ref<MLPPMatrix> deriv;
	deriv.instance();
	deriv->resize(y_hat->size());

	real_t *deriv_ptr = deriv->ptrw();

	for (int i = 0; i < y_hat_data_size; ++i) {
		if (1 - y_ptr[i] * y_hat_ptr[i] > 0) {
			deriv_ptr[i] = -y_ptr[i];
		} else {
			deriv_ptr[i] = 0;
		}
	}

	return deriv;
}

real_t MLPPCostOld::hinge_losswv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y, const Ref<MLPPVector> &weights, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;

	return C * hinge_lossv(y_hat, y) + regularization.reg_termv(weights, 1, 0, MLPPReg::REGULARIZATION_TYPE_RIDGE);
}
real_t MLPPCostOld::hinge_losswm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y, const Ref<MLPPMatrix> &weights, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;

	return C * hinge_lossm(y_hat, y) + regularization.reg_termv(weights, 1, 0, MLPPReg::REGULARIZATION_TYPE_RIDGE);
}

Ref<MLPPVector> MLPPCostOld::hinge_loss_derivwv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;

	return alg.scalar_multiplynv(C, hinge_loss_derivv(y_hat, y));
}
Ref<MLPPMatrix> MLPPCostOld::hinge_loss_derivwm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;

	return alg.scalar_multiplym(C, hinge_loss_derivm(y_hat, y));
}

real_t MLPPCostOld::wasserstein_lossv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	int y_hat_size = y_hat->size();

	ERR_FAIL_COND_V(y_hat_size != y->size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_size; ++i) {
		sum += y_hat_ptr[i] * y_ptr[i];
	}

	return -sum / static_cast<real_t>(y_hat_size);
}
real_t MLPPCostOld::wasserstein_lossm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	int y_hat_data_size = y_hat->data_size();

	ERR_FAIL_COND_V(y_hat_data_size != y->data_size(), 0);

	const real_t *y_hat_ptr = y_hat->ptr();
	const real_t *y_ptr = y->ptr();

	real_t sum = 0;
	for (int i = 0; i < y_hat_data_size; ++i) {
		sum += y_hat_ptr[i] * y_ptr[i];
	}

	return -sum / static_cast<real_t>(y_hat_data_size);
}

Ref<MLPPVector> MLPPCostOld::wasserstein_loss_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(-1, y); // Simple.
}
Ref<MLPPMatrix> MLPPCostOld::wasserstein_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;

	return alg.scalar_multiplym(-1, y); // Simple.
}

real_t MLPPCostOld::dual_form_svm(const Ref<MLPPVector> &alpha, const Ref<MLPPMatrix> &X, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;

	Ref<MLPPMatrix> Y = alg.diagm(y); // Y is a diagnoal matrix. Y[i][j] = y[i] if i = i, else Y[i][j] = 0. Yt = Y.
	Ref<MLPPMatrix> K = alg.matmultm(X, alg.transposem(X)); // TO DO: DON'T forget to add non-linear kernelizations.
	Ref<MLPPMatrix> Q = alg.matmultm(alg.matmultm(alg.transposem(Y), K), Y);

	Ref<MLPPMatrix> alpha_m;
	alpha_m.instance();
	alpha_m->resize(Size2i(alpha->size(), 1));
	alpha_m->set_row_mlpp_vector(0, alpha);

	Ref<MLPPMatrix> alpha_m_res = alg.matmultm(alg.matmultm(alpha_m, Q), alg.transposem(alpha_m));

	real_t alphaQ = alpha_m_res->get_element(0, 0);
	Ref<MLPPVector> one = alg.onevecv(alpha->size());

	return -alg.dotv(one, alpha) + 0.5 * alphaQ;
}

Ref<MLPPVector> MLPPCostOld::dual_form_svm_deriv(const Ref<MLPPVector> &alpha, const Ref<MLPPMatrix> &X, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;

	Ref<MLPPMatrix> Y = alg.diagm(y); // Y is a diagnoal matrix. Y[i][j] = y[i] if i = i, else Y[i][j] = 0. Yt = Y.
	Ref<MLPPMatrix> K = alg.matmultm(X, alg.transposem(X)); // TO DO: DON'T forget to add non-linear kernelizations.
	Ref<MLPPMatrix> Q = alg.matmultm(alg.matmultm(alg.transposem(Y), K), Y);
	Ref<MLPPVector> alphaQDeriv = alg.mat_vec_multv(Q, alpha);
	Ref<MLPPVector> one = alg.onevecv(alpha->size());

	return alg.subtractionm(alphaQDeriv, one);
}

MLPPCostOld::VectorCostFunctionPointer MLPPCostOld::get_cost_function_ptr_normal_vector(const MLPPCostOld::CostTypes cost) {
	switch (cost) {
		case COST_TYPE_MSE:
			return &MLPPCostOld::msev;
		case COST_TYPE_RMSE:
			return &MLPPCostOld::rmsev;
		case COST_TYPE_MAE:
			return &MLPPCostOld::maev;
		case COST_TYPE_MBE:
			return &MLPPCostOld::mbev;
		case COST_TYPE_LOGISTIC_LOSS:
			return &MLPPCostOld::log_lossv;
		case COST_TYPE_CROSS_ENTROPY:
			return &MLPPCostOld::cross_entropyv;
		case COST_TYPE_HINGE_LOSS:
			return &MLPPCostOld::hinge_lossv;
		case COST_TYPE_WASSERSTEIN_LOSS:
			return &MLPPCostOld::wasserstein_lossv;
		default:
			return NULL;
	}
}
MLPPCostOld::MatrixCostFunctionPointer MLPPCostOld::get_cost_function_ptr_normal_matrix(const MLPPCostOld::CostTypes cost) {
	switch (cost) {
		case COST_TYPE_MSE:
			return &MLPPCostOld::msem;
		case COST_TYPE_RMSE:
			return &MLPPCostOld::rmsem;
		case COST_TYPE_MAE:
			return &MLPPCostOld::maem;
		case COST_TYPE_MBE:
			return &MLPPCostOld::mbem;
		case COST_TYPE_LOGISTIC_LOSS:
			return &MLPPCostOld::log_lossm;
		case COST_TYPE_CROSS_ENTROPY:
			return &MLPPCostOld::cross_entropym;
		case COST_TYPE_HINGE_LOSS:
			return &MLPPCostOld::hinge_lossm;
		case COST_TYPE_WASSERSTEIN_LOSS:
			return &MLPPCostOld::wasserstein_lossm;
		default:
			return NULL;
	}
}

MLPPCostOld::VectorDerivCostFunctionPointer MLPPCostOld::get_cost_function_ptr_deriv_vector(const MLPPCostOld::CostTypes cost) {
	switch (cost) {
		case COST_TYPE_MSE:
			return &MLPPCostOld::mse_derivv;
		case COST_TYPE_RMSE:
			return &MLPPCostOld::rmse_derivv;
		case COST_TYPE_MAE:
			return &MLPPCostOld::mae_derivv;
		case COST_TYPE_MBE:
			return &MLPPCostOld::mbe_derivv;
		case COST_TYPE_LOGISTIC_LOSS:
			return &MLPPCostOld::log_loss_derivv;
		case COST_TYPE_CROSS_ENTROPY:
			return &MLPPCostOld::cross_entropy_derivv;
		case COST_TYPE_HINGE_LOSS:
			return &MLPPCostOld::hinge_loss_derivv;
		case COST_TYPE_WASSERSTEIN_LOSS:
			return &MLPPCostOld::wasserstein_loss_derivv;
		default:
			return NULL;
	}
}
MLPPCostOld::MatrixDerivCostFunctionPointer MLPPCostOld::get_cost_function_ptr_deriv_matrix(const MLPPCostOld::CostTypes cost) {
	switch (cost) {
		case COST_TYPE_MSE:
			return &MLPPCostOld::mse_derivm;
		case COST_TYPE_RMSE:
			return &MLPPCostOld::rmse_derivm;
		case COST_TYPE_MAE:
			return &MLPPCostOld::mae_derivm;
		case COST_TYPE_MBE:
			return &MLPPCostOld::mbe_derivm;
		case COST_TYPE_LOGISTIC_LOSS:
			return &MLPPCostOld::log_loss_derivm;
		case COST_TYPE_CROSS_ENTROPY:
			return &MLPPCostOld::cross_entropy_derivm;
		case COST_TYPE_HINGE_LOSS:
			return &MLPPCostOld::hinge_loss_derivm;
		case COST_TYPE_WASSERSTEIN_LOSS:
			return &MLPPCostOld::wasserstein_loss_derivm;
		default:
			return NULL;
	}
}

real_t MLPPCostOld::run_cost_norm_vector(const CostTypes cost, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	switch (cost) {
		case COST_TYPE_MSE:
			return msev(y_hat, y);
		case COST_TYPE_RMSE:
			return rmsev(y_hat, y);
		case COST_TYPE_MAE:
			return maev(y_hat, y);
		case COST_TYPE_MBE:
			return mbev(y_hat, y);
		case COST_TYPE_LOGISTIC_LOSS:
			return log_lossv(y_hat, y);
		case COST_TYPE_CROSS_ENTROPY:
			return cross_entropyv(y_hat, y);
		case COST_TYPE_HINGE_LOSS:
			return hinge_lossv(y_hat, y);
		case COST_TYPE_WASSERSTEIN_LOSS:
			return wasserstein_lossv(y_hat, y);
		default:
			return 0;
	}
}
real_t MLPPCostOld::run_cost_norm_matrix(const CostTypes cost, const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	switch (cost) {
		case COST_TYPE_MSE:
			return msem(y_hat, y);
		case COST_TYPE_RMSE:
			return rmsem(y_hat, y);
		case COST_TYPE_MAE:
			return maem(y_hat, y);
		case COST_TYPE_MBE:
			return mbem(y_hat, y);
		case COST_TYPE_LOGISTIC_LOSS:
			return log_lossm(y_hat, y);
		case COST_TYPE_CROSS_ENTROPY:
			return cross_entropym(y_hat, y);
		case COST_TYPE_HINGE_LOSS:
			return hinge_lossm(y_hat, y);
		case COST_TYPE_WASSERSTEIN_LOSS:
			return wasserstein_lossm(y_hat, y);
		default:
			return 0;
	}
}

Ref<MLPPVector> MLPPCostOld::run_cost_deriv_vector(const CostTypes cost, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	switch (cost) {
		case COST_TYPE_MSE:
			return mse_derivv(y_hat, y);
		case COST_TYPE_RMSE:
			return rmse_derivv(y_hat, y);
		case COST_TYPE_MAE:
			return mae_derivv(y_hat, y);
		case COST_TYPE_MBE:
			return mbe_derivv(y_hat, y);
		case COST_TYPE_LOGISTIC_LOSS:
			return log_loss_derivv(y_hat, y);
		case COST_TYPE_CROSS_ENTROPY:
			return cross_entropy_derivv(y_hat, y);
		case COST_TYPE_HINGE_LOSS:
			return hinge_loss_derivv(y_hat, y);
		case COST_TYPE_WASSERSTEIN_LOSS:
			return wasserstein_loss_derivv(y_hat, y);
		default:
			return Ref<MLPPVector>();
	}
}
Ref<MLPPMatrix> MLPPCostOld::run_cost_deriv_matrix(const CostTypes cost, const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	switch (cost) {
		case COST_TYPE_MSE:
			return mse_derivm(y_hat, y);
		case COST_TYPE_RMSE:
			return rmse_derivm(y_hat, y);
		case COST_TYPE_MAE:
			return mae_derivm(y_hat, y);
		case COST_TYPE_MBE:
			return mbe_derivm(y_hat, y);
		case COST_TYPE_LOGISTIC_LOSS:
			return log_loss_derivm(y_hat, y);
		case COST_TYPE_CROSS_ENTROPY:
			return cross_entropy_derivm(y_hat, y);
		case COST_TYPE_HINGE_LOSS:
			return hinge_loss_derivm(y_hat, y);
		case COST_TYPE_WASSERSTEIN_LOSS:
			return wasserstein_loss_derivm(y_hat, y);
		default:
			return Ref<MLPPMatrix>();
	}
}

// ======  OLD  ======

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

void MLPPCostOld::_bind_methods() {
	ClassDB::bind_method(D_METHOD("msev", "y_hat", "y"), &MLPPCostOld::msev);
	ClassDB::bind_method(D_METHOD("msem", "y_hat", "y"), &MLPPCostOld::msem);

	ClassDB::bind_method(D_METHOD("mse_derivv", "y_hat", "y"), &MLPPCostOld::mse_derivv);
	ClassDB::bind_method(D_METHOD("mse_derivm", "y_hat", "y"), &MLPPCostOld::mse_derivm);

	ClassDB::bind_method(D_METHOD("rmsev", "y_hat", "y"), &MLPPCostOld::rmsev);
	ClassDB::bind_method(D_METHOD("rmsem", "y_hat", "y"), &MLPPCostOld::rmsem);

	ClassDB::bind_method(D_METHOD("rmse_derivv", "y_hat", "y"), &MLPPCostOld::rmse_derivv);
	ClassDB::bind_method(D_METHOD("rmse_derivm", "y_hat", "y"), &MLPPCostOld::rmse_derivm);

	ClassDB::bind_method(D_METHOD("maev", "y_hat", "y"), &MLPPCostOld::maev);
	ClassDB::bind_method(D_METHOD("maem", "y_hat", "y"), &MLPPCostOld::maem);

	ClassDB::bind_method(D_METHOD("mae_derivv", "y_hat", "y"), &MLPPCostOld::mae_derivv);
	ClassDB::bind_method(D_METHOD("mae_derivm", "y_hat", "y"), &MLPPCostOld::mae_derivm);

	ClassDB::bind_method(D_METHOD("mbev", "y_hat", "y"), &MLPPCostOld::mbev);
	ClassDB::bind_method(D_METHOD("mbem", "y_hat", "y"), &MLPPCostOld::mbem);

	ClassDB::bind_method(D_METHOD("mbe_derivv", "y_hat", "y"), &MLPPCostOld::mbe_derivv);
	ClassDB::bind_method(D_METHOD("mbe_derivm", "y_hat", "y"), &MLPPCostOld::mbe_derivm);

	ClassDB::bind_method(D_METHOD("log_lossv", "y_hat", "y"), &MLPPCostOld::log_lossv);
	ClassDB::bind_method(D_METHOD("log_lossm", "y_hat", "y"), &MLPPCostOld::log_lossm);

	ClassDB::bind_method(D_METHOD("log_loss_derivv", "y_hat", "y"), &MLPPCostOld::log_loss_derivv);
	ClassDB::bind_method(D_METHOD("log_loss_derivm", "y_hat", "y"), &MLPPCostOld::log_loss_derivm);

	ClassDB::bind_method(D_METHOD("cross_entropyv", "y_hat", "y"), &MLPPCostOld::cross_entropyv);
	ClassDB::bind_method(D_METHOD("cross_entropym", "y_hat", "y"), &MLPPCostOld::cross_entropym);

	ClassDB::bind_method(D_METHOD("cross_entropy_derivv", "y_hat", "y"), &MLPPCostOld::cross_entropy_derivv);
	ClassDB::bind_method(D_METHOD("cross_entropy_derivm", "y_hat", "y"), &MLPPCostOld::cross_entropy_derivm);

	ClassDB::bind_method(D_METHOD("huber_lossv", "y_hat", "y"), &MLPPCostOld::huber_lossv);
	ClassDB::bind_method(D_METHOD("huber_lossm", "y_hat", "y"), &MLPPCostOld::huber_lossm);

	ClassDB::bind_method(D_METHOD("huber_loss_derivv", "y_hat", "y"), &MLPPCostOld::huber_loss_derivv);
	ClassDB::bind_method(D_METHOD("huber_loss_derivm", "y_hat", "y"), &MLPPCostOld::huber_loss_derivm);

	ClassDB::bind_method(D_METHOD("hinge_lossv", "y_hat", "y"), &MLPPCostOld::hinge_lossv);
	ClassDB::bind_method(D_METHOD("hinge_lossm", "y_hat", "y"), &MLPPCostOld::hinge_lossm);

	ClassDB::bind_method(D_METHOD("hinge_loss_derivv", "y_hat", "y"), &MLPPCostOld::hinge_loss_derivv);
	ClassDB::bind_method(D_METHOD("hinge_loss_derivm", "y_hat", "y"), &MLPPCostOld::hinge_loss_derivm);

	ClassDB::bind_method(D_METHOD("hinge_losswv", "y_hat", "y"), &MLPPCostOld::hinge_losswv);
	ClassDB::bind_method(D_METHOD("hinge_losswm", "y_hat", "y"), &MLPPCostOld::hinge_losswm);

	ClassDB::bind_method(D_METHOD("hinge_loss_derivwv", "y_hat", "y", "C"), &MLPPCostOld::hinge_loss_derivwv);
	ClassDB::bind_method(D_METHOD("hinge_loss_derivwm", "y_hat", "y", "C"), &MLPPCostOld::hinge_loss_derivwm);

	ClassDB::bind_method(D_METHOD("wasserstein_lossv", "y_hat", "y"), &MLPPCostOld::wasserstein_lossv);
	ClassDB::bind_method(D_METHOD("wasserstein_lossm", "y_hat", "y"), &MLPPCostOld::wasserstein_lossm);

	ClassDB::bind_method(D_METHOD("wasserstein_loss_derivv", "y_hat", "y"), &MLPPCostOld::wasserstein_loss_derivv);
	ClassDB::bind_method(D_METHOD("wasserstein_loss_derivm", "y_hat", "y"), &MLPPCostOld::wasserstein_loss_derivm);

	ClassDB::bind_method(D_METHOD("dual_form_svm", "alpha", "X", "y"), &MLPPCostOld::dual_form_svm);
	ClassDB::bind_method(D_METHOD("dual_form_svm_deriv", "alpha", "X", "y"), &MLPPCostOld::dual_form_svm_deriv);

	ClassDB::bind_method(D_METHOD("run_cost_norm_vector", "cost", "y_hat", "y"), &MLPPCostOld::run_cost_norm_vector);
	ClassDB::bind_method(D_METHOD("run_cost_norm_matrix", "cost", "y_hat", "y"), &MLPPCostOld::run_cost_norm_matrix);

	ClassDB::bind_method(D_METHOD("run_cost_deriv_vector", "cost", "y_hat", "y"), &MLPPCostOld::run_cost_deriv_vector);
	ClassDB::bind_method(D_METHOD("run_cost_deriv_matrix", "cost", "y_hat", "y"), &MLPPCostOld::run_cost_deriv_matrix);

	BIND_ENUM_CONSTANT(COST_TYPE_MSE);
	BIND_ENUM_CONSTANT(COST_TYPE_RMSE);
	BIND_ENUM_CONSTANT(COST_TYPE_MAE);
	BIND_ENUM_CONSTANT(COST_TYPE_MBE);
	BIND_ENUM_CONSTANT(COST_TYPE_LOGISTIC_LOSS);
	BIND_ENUM_CONSTANT(COST_TYPE_CROSS_ENTROPY);
	BIND_ENUM_CONSTANT(COST_TYPE_HINGE_LOSS);
	BIND_ENUM_CONSTANT(COST_TYPE_WASSERSTEIN_LOSS);
}
