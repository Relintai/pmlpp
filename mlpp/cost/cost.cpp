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

real_t MLPPCost::msev(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
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
real_t MLPPCost::msem(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
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

Ref<MLPPVector> MLPPCost::mse_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;
	return alg.subtractionnv(y_hat, y);
}

Ref<MLPPMatrix> MLPPCost::mse_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;
	return alg.subtractionnm(y_hat, y);
}

real_t MLPPCost::rmsev(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
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
real_t MLPPCost::rmsem(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
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

Ref<MLPPVector> MLPPCost::rmse_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(1 / (2.0 * Math::sqrt(msev(y_hat, y))), mse_derivv(y_hat, y));
}
Ref<MLPPMatrix> MLPPCost::rmse_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynm(1 / (2.0 / Math::sqrt(msem(y_hat, y))), mse_derivm(y_hat, y));
}

real_t MLPPCost::maev(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
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
real_t MLPPCost::maem(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
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

Ref<MLPPVector> MLPPCost::mae_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
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
Ref<MLPPMatrix> MLPPCost::mae_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
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

real_t MLPPCost::mbev(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
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
real_t MLPPCost::mbem(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
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

Ref<MLPPVector> MLPPCost::mbe_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;
	return alg.onevecnv(y_hat->size());
}
Ref<MLPPMatrix> MLPPCost::mbe_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;
	return alg.onematnm(y_hat->size().x, y_hat->size().y);
}

// Classification Costs
real_t MLPPCost::log_lossv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
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

real_t MLPPCost::log_lossm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
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

Ref<MLPPVector> MLPPCost::log_loss_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;
	return alg.additionnv(
			alg.scalar_multiplynv(-1, alg.element_wise_divisionnv(y, y_hat)),
			alg.element_wise_divisionnv(alg.scalar_multiplynv(-1, alg.scalar_addnv(-1, y)), alg.scalar_multiplynv(-1, alg.scalar_addnv(-1, y_hat))));
}

Ref<MLPPMatrix> MLPPCost::log_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;
	return alg.additionnm(
			alg.scalar_multiplynm(-1, alg.element_wise_divisionnvnm(y, y_hat)),
			alg.element_wise_divisionnvnm(alg.scalar_multiplynm(-1, alg.scalar_addnm(-1, y)), alg.scalar_multiplynm(-1, alg.scalar_addnm(-1, y_hat))));
}

real_t MLPPCost::cross_entropyv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
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
real_t MLPPCost::cross_entropym(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
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

Ref<MLPPVector> MLPPCost::cross_entropy_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;
	return alg.scalar_multiplynv(-1, alg.element_wise_divisionnv(y, y_hat));
}
Ref<MLPPMatrix> MLPPCost::cross_entropy_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;
	return alg.scalar_multiplynm(-1, alg.element_wise_divisionnvnm(y, y_hat));
}

real_t MLPPCost::huber_lossv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y, real_t delta) {
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
real_t MLPPCost::huber_lossm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y, real_t delta) {
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

Ref<MLPPVector> MLPPCost::huber_loss_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y, real_t delta) {
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
Ref<MLPPMatrix> MLPPCost::huber_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y, real_t delta) {
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

real_t MLPPCost::hinge_lossv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
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
real_t MLPPCost::hinge_lossm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
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

Ref<MLPPVector> MLPPCost::hinge_loss_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
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
Ref<MLPPMatrix> MLPPCost::hinge_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
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

real_t MLPPCost::hinge_losswv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y, const Ref<MLPPVector> &weights, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;

	return C * hinge_lossv(y_hat, y) + regularization.reg_termv(weights, 1, 0, MLPPReg::REGULARIZATION_TYPE_RIDGE);
}
real_t MLPPCost::hinge_losswm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y, const Ref<MLPPMatrix> &weights, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;

	return C * hinge_lossm(y_hat, y) + regularization.reg_termv(weights, 1, 0, MLPPReg::REGULARIZATION_TYPE_RIDGE);
}

Ref<MLPPVector> MLPPCost::hinge_loss_derivwv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;

	return alg.scalar_multiplynv(C, hinge_loss_derivv(y_hat, y));
}
Ref<MLPPMatrix> MLPPCost::hinge_loss_derivwm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;

	return alg.scalar_multiplynm(C, hinge_loss_derivm(y_hat, y));
}

real_t MLPPCost::wasserstein_lossv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
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
real_t MLPPCost::wasserstein_lossm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
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

Ref<MLPPVector> MLPPCost::wasserstein_loss_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(-1, y); // Simple.
}
Ref<MLPPMatrix> MLPPCost::wasserstein_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynm(-1, y); // Simple.
}

real_t MLPPCost::dual_form_svm(const Ref<MLPPVector> &alpha, const Ref<MLPPMatrix> &X, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;

	Ref<MLPPMatrix> Y = alg.diagnm(y); // Y is a diagnoal matrix. Y[i][j] = y[i] if i = i, else Y[i][j] = 0. Yt = Y.
	Ref<MLPPMatrix> K = alg.matmultnm(X, alg.transposenm(X)); // TO DO: DON'T forget to add non-linear kernelizations.
	Ref<MLPPMatrix> Q = alg.matmultnm(alg.matmultnm(alg.transposenm(Y), K), Y);

	Ref<MLPPMatrix> alpha_m;
	alpha_m.instance();
	alpha_m->resize(Size2i(alpha->size(), 1));
	alpha_m->set_row_mlpp_vector(0, alpha);

	Ref<MLPPMatrix> alpha_m_res = alg.matmultnm(alg.matmultnm(alpha_m, Q), alg.transposenm(alpha_m));

	real_t alphaQ = alpha_m_res->element_get(0, 0);
	Ref<MLPPVector> one = alg.onevecnv(alpha->size());

	return -alg.dotnv(one, alpha) + 0.5 * alphaQ;
}

Ref<MLPPVector> MLPPCost::dual_form_svm_deriv(const Ref<MLPPVector> &alpha, const Ref<MLPPMatrix> &X, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;

	Ref<MLPPMatrix> Y = alg.diagnm(y); // Y is a diagnoal matrix. Y[i][j] = y[i] if i = i, else Y[i][j] = 0. Yt = Y.
	Ref<MLPPMatrix> K = alg.matmultnm(X, alg.transposenm(X)); // TO DO: DON'T forget to add non-linear kernelizations.
	Ref<MLPPMatrix> Q = alg.matmultnm(alg.matmultnm(alg.transposenm(Y), K), Y);
	Ref<MLPPVector> alphaQDeriv = alg.mat_vec_multnv(Q, alpha);
	Ref<MLPPVector> one = alg.onevecnv(alpha->size());

	return alg.subtractionnm(alphaQDeriv, one);
}

MLPPCost::VectorCostFunctionPointer MLPPCost::get_cost_function_ptr_normal_vector(const MLPPCost::CostTypes cost) {
	switch (cost) {
		case COST_TYPE_MSE:
			return &MLPPCost::msev;
		case COST_TYPE_RMSE:
			return &MLPPCost::rmsev;
		case COST_TYPE_MAE:
			return &MLPPCost::maev;
		case COST_TYPE_MBE:
			return &MLPPCost::mbev;
		case COST_TYPE_LOGISTIC_LOSS:
			return &MLPPCost::log_lossv;
		case COST_TYPE_CROSS_ENTROPY:
			return &MLPPCost::cross_entropyv;
		case COST_TYPE_HINGE_LOSS:
			return &MLPPCost::hinge_lossv;
		case COST_TYPE_WASSERSTEIN_LOSS:
			return &MLPPCost::wasserstein_lossv;
		default:
			return NULL;
	}
}
MLPPCost::MatrixCostFunctionPointer MLPPCost::get_cost_function_ptr_normal_matrix(const MLPPCost::CostTypes cost) {
	switch (cost) {
		case COST_TYPE_MSE:
			return &MLPPCost::msem;
		case COST_TYPE_RMSE:
			return &MLPPCost::rmsem;
		case COST_TYPE_MAE:
			return &MLPPCost::maem;
		case COST_TYPE_MBE:
			return &MLPPCost::mbem;
		case COST_TYPE_LOGISTIC_LOSS:
			return &MLPPCost::log_lossm;
		case COST_TYPE_CROSS_ENTROPY:
			return &MLPPCost::cross_entropym;
		case COST_TYPE_HINGE_LOSS:
			return &MLPPCost::hinge_lossm;
		case COST_TYPE_WASSERSTEIN_LOSS:
			return &MLPPCost::wasserstein_lossm;
		default:
			return NULL;
	}
}

MLPPCost::VectorDerivCostFunctionPointer MLPPCost::get_cost_function_ptr_deriv_vector(const MLPPCost::CostTypes cost) {
	switch (cost) {
		case COST_TYPE_MSE:
			return &MLPPCost::mse_derivv;
		case COST_TYPE_RMSE:
			return &MLPPCost::rmse_derivv;
		case COST_TYPE_MAE:
			return &MLPPCost::mae_derivv;
		case COST_TYPE_MBE:
			return &MLPPCost::mbe_derivv;
		case COST_TYPE_LOGISTIC_LOSS:
			return &MLPPCost::log_loss_derivv;
		case COST_TYPE_CROSS_ENTROPY:
			return &MLPPCost::cross_entropy_derivv;
		case COST_TYPE_HINGE_LOSS:
			return &MLPPCost::hinge_loss_derivv;
		case COST_TYPE_WASSERSTEIN_LOSS:
			return &MLPPCost::wasserstein_loss_derivv;
		default:
			return NULL;
	}
}
MLPPCost::MatrixDerivCostFunctionPointer MLPPCost::get_cost_function_ptr_deriv_matrix(const MLPPCost::CostTypes cost) {
	switch (cost) {
		case COST_TYPE_MSE:
			return &MLPPCost::mse_derivm;
		case COST_TYPE_RMSE:
			return &MLPPCost::rmse_derivm;
		case COST_TYPE_MAE:
			return &MLPPCost::mae_derivm;
		case COST_TYPE_MBE:
			return &MLPPCost::mbe_derivm;
		case COST_TYPE_LOGISTIC_LOSS:
			return &MLPPCost::log_loss_derivm;
		case COST_TYPE_CROSS_ENTROPY:
			return &MLPPCost::cross_entropy_derivm;
		case COST_TYPE_HINGE_LOSS:
			return &MLPPCost::hinge_loss_derivm;
		case COST_TYPE_WASSERSTEIN_LOSS:
			return &MLPPCost::wasserstein_loss_derivm;
		default:
			return NULL;
	}
}

real_t MLPPCost::run_cost_norm_vector(const CostTypes cost, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
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
real_t MLPPCost::run_cost_norm_matrix(const CostTypes cost, const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
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

Ref<MLPPVector> MLPPCost::run_cost_deriv_vector(const CostTypes cost, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y) {
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
Ref<MLPPMatrix> MLPPCost::run_cost_deriv_matrix(const CostTypes cost, const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
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

void MLPPCost::_bind_methods() {
	ClassDB::bind_method(D_METHOD("msev", "y_hat", "y"), &MLPPCost::msev);
	ClassDB::bind_method(D_METHOD("msem", "y_hat", "y"), &MLPPCost::msem);

	ClassDB::bind_method(D_METHOD("mse_derivv", "y_hat", "y"), &MLPPCost::mse_derivv);
	ClassDB::bind_method(D_METHOD("mse_derivm", "y_hat", "y"), &MLPPCost::mse_derivm);

	ClassDB::bind_method(D_METHOD("rmsev", "y_hat", "y"), &MLPPCost::rmsev);
	ClassDB::bind_method(D_METHOD("rmsem", "y_hat", "y"), &MLPPCost::rmsem);

	ClassDB::bind_method(D_METHOD("rmse_derivv", "y_hat", "y"), &MLPPCost::rmse_derivv);
	ClassDB::bind_method(D_METHOD("rmse_derivm", "y_hat", "y"), &MLPPCost::rmse_derivm);

	ClassDB::bind_method(D_METHOD("maev", "y_hat", "y"), &MLPPCost::maev);
	ClassDB::bind_method(D_METHOD("maem", "y_hat", "y"), &MLPPCost::maem);

	ClassDB::bind_method(D_METHOD("mae_derivv", "y_hat", "y"), &MLPPCost::mae_derivv);
	ClassDB::bind_method(D_METHOD("mae_derivm", "y_hat", "y"), &MLPPCost::mae_derivm);

	ClassDB::bind_method(D_METHOD("mbev", "y_hat", "y"), &MLPPCost::mbev);
	ClassDB::bind_method(D_METHOD("mbem", "y_hat", "y"), &MLPPCost::mbem);

	ClassDB::bind_method(D_METHOD("mbe_derivv", "y_hat", "y"), &MLPPCost::mbe_derivv);
	ClassDB::bind_method(D_METHOD("mbe_derivm", "y_hat", "y"), &MLPPCost::mbe_derivm);

	ClassDB::bind_method(D_METHOD("log_lossv", "y_hat", "y"), &MLPPCost::log_lossv);
	ClassDB::bind_method(D_METHOD("log_lossm", "y_hat", "y"), &MLPPCost::log_lossm);

	ClassDB::bind_method(D_METHOD("log_loss_derivv", "y_hat", "y"), &MLPPCost::log_loss_derivv);
	ClassDB::bind_method(D_METHOD("log_loss_derivm", "y_hat", "y"), &MLPPCost::log_loss_derivm);

	ClassDB::bind_method(D_METHOD("cross_entropyv", "y_hat", "y"), &MLPPCost::cross_entropyv);
	ClassDB::bind_method(D_METHOD("cross_entropym", "y_hat", "y"), &MLPPCost::cross_entropym);

	ClassDB::bind_method(D_METHOD("cross_entropy_derivv", "y_hat", "y"), &MLPPCost::cross_entropy_derivv);
	ClassDB::bind_method(D_METHOD("cross_entropy_derivm", "y_hat", "y"), &MLPPCost::cross_entropy_derivm);

	ClassDB::bind_method(D_METHOD("huber_lossv", "y_hat", "y"), &MLPPCost::huber_lossv);
	ClassDB::bind_method(D_METHOD("huber_lossm", "y_hat", "y"), &MLPPCost::huber_lossm);

	ClassDB::bind_method(D_METHOD("huber_loss_derivv", "y_hat", "y"), &MLPPCost::huber_loss_derivv);
	ClassDB::bind_method(D_METHOD("huber_loss_derivm", "y_hat", "y"), &MLPPCost::huber_loss_derivm);

	ClassDB::bind_method(D_METHOD("hinge_lossv", "y_hat", "y"), &MLPPCost::hinge_lossv);
	ClassDB::bind_method(D_METHOD("hinge_lossm", "y_hat", "y"), &MLPPCost::hinge_lossm);

	ClassDB::bind_method(D_METHOD("hinge_loss_derivv", "y_hat", "y"), &MLPPCost::hinge_loss_derivv);
	ClassDB::bind_method(D_METHOD("hinge_loss_derivm", "y_hat", "y"), &MLPPCost::hinge_loss_derivm);

	ClassDB::bind_method(D_METHOD("hinge_losswv", "y_hat", "y"), &MLPPCost::hinge_losswv);
	ClassDB::bind_method(D_METHOD("hinge_losswm", "y_hat", "y"), &MLPPCost::hinge_losswm);

	ClassDB::bind_method(D_METHOD("hinge_loss_derivwv", "y_hat", "y", "C"), &MLPPCost::hinge_loss_derivwv);
	ClassDB::bind_method(D_METHOD("hinge_loss_derivwm", "y_hat", "y", "C"), &MLPPCost::hinge_loss_derivwm);

	ClassDB::bind_method(D_METHOD("wasserstein_lossv", "y_hat", "y"), &MLPPCost::wasserstein_lossv);
	ClassDB::bind_method(D_METHOD("wasserstein_lossm", "y_hat", "y"), &MLPPCost::wasserstein_lossm);

	ClassDB::bind_method(D_METHOD("wasserstein_loss_derivv", "y_hat", "y"), &MLPPCost::wasserstein_loss_derivv);
	ClassDB::bind_method(D_METHOD("wasserstein_loss_derivm", "y_hat", "y"), &MLPPCost::wasserstein_loss_derivm);

	ClassDB::bind_method(D_METHOD("dual_form_svm", "alpha", "X", "y"), &MLPPCost::dual_form_svm);
	ClassDB::bind_method(D_METHOD("dual_form_svm_deriv", "alpha", "X", "y"), &MLPPCost::dual_form_svm_deriv);

	ClassDB::bind_method(D_METHOD("run_cost_norm_vector", "cost", "y_hat", "y"), &MLPPCost::run_cost_norm_vector);
	ClassDB::bind_method(D_METHOD("run_cost_norm_matrix", "cost", "y_hat", "y"), &MLPPCost::run_cost_norm_matrix);

	ClassDB::bind_method(D_METHOD("run_cost_deriv_vector", "cost", "y_hat", "y"), &MLPPCost::run_cost_deriv_vector);
	ClassDB::bind_method(D_METHOD("run_cost_deriv_matrix", "cost", "y_hat", "y"), &MLPPCost::run_cost_deriv_matrix);

	BIND_ENUM_CONSTANT(COST_TYPE_MSE);
	BIND_ENUM_CONSTANT(COST_TYPE_RMSE);
	BIND_ENUM_CONSTANT(COST_TYPE_MAE);
	BIND_ENUM_CONSTANT(COST_TYPE_MBE);
	BIND_ENUM_CONSTANT(COST_TYPE_LOGISTIC_LOSS);
	BIND_ENUM_CONSTANT(COST_TYPE_CROSS_ENTROPY);
	BIND_ENUM_CONSTANT(COST_TYPE_HINGE_LOSS);
	BIND_ENUM_CONSTANT(COST_TYPE_WASSERSTEIN_LOSS);
}
