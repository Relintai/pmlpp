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
	return alg.subtractionm(y_hat, y);
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

	return alg.scalar_multiplym(1 / (2.0 / Math::sqrt(msem(y_hat, y))), mse_derivm(y_hat, y));
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
	return alg.onevecv(y_hat->size());
}
Ref<MLPPMatrix> MLPPCost::mbe_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;
	return alg.onematm(y_hat->size().x, y_hat->size().y);
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
			alg.scalar_multiplynv(-1, alg.element_wise_division(y, y_hat)),
			alg.element_wise_division(alg.scalar_multiplynv(-1, alg.scalar_addnv(-1, y)), alg.scalar_multiplynv(-1, alg.scalar_addnv(-1, y_hat))));
}

Ref<MLPPMatrix> MLPPCost::log_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;
	return alg.additionm(
			alg.scalar_multiplym(-1, alg.element_wise_divisionm(y, y_hat)),
			alg.element_wise_divisionm(alg.scalar_multiplym(-1, alg.scalar_addm(-1, y)), alg.scalar_multiplym(-1, alg.scalar_addm(-1, y_hat))));
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
	return alg.scalar_multiplynv(-1, alg.element_wise_division(y, y_hat));
}
Ref<MLPPMatrix> MLPPCost::cross_entropy_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	MLPPLinAlg alg;
	return alg.scalar_multiplym(-1, alg.element_wise_divisionm(y, y_hat));
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
	real_t sum = 0;

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
	real_t sum = 0;

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

	return alg.scalar_multiplym(C, hinge_loss_derivm(y_hat, y));
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

	return alg.scalar_multiplym(-1, y); // Simple.
}

real_t MLPPCost::dual_form_svm(const Ref<MLPPVector> &alpha, const Ref<MLPPMatrix> &X, const Ref<MLPPVector> &y) {
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

Ref<MLPPVector> MLPPCost::dual_form_svm_deriv(const Ref<MLPPVector> &alpha, const Ref<MLPPMatrix> &X, const Ref<MLPPVector> &y) {
	MLPPLinAlg alg;

	Ref<MLPPMatrix> Y = alg.diagm(y); // Y is a diagnoal matrix. Y[i][j] = y[i] if i = i, else Y[i][j] = 0. Yt = Y.
	Ref<MLPPMatrix> K = alg.matmultm(X, alg.transposem(X)); // TO DO: DON'T forget to add non-linear kernelizations.
	Ref<MLPPMatrix> Q = alg.matmultm(alg.matmultm(alg.transposem(Y), K), Y);
	Ref<MLPPVector> alphaQDeriv = alg.mat_vec_multv(Q, alpha);
	Ref<MLPPVector> one = alg.onevecv(alpha->size());

	return alg.subtractionm(alphaQDeriv, one);
}

// ======  OLD  ======

real_t MLPPCost::MSE(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += (y_hat[i] - y[i]) * (y_hat[i] - y[i]);
	}
	return sum / 2 * y_hat.size();
}

real_t MLPPCost::MSE(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += (y_hat[i][j] - y[i][j]) * (y_hat[i][j] - y[i][j]);
		}
	}
	return sum / 2 * y_hat.size();
}

std::vector<real_t> MLPPCost::MSEDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPLinAlg alg;
	return alg.subtraction(y_hat, y);
}

std::vector<std::vector<real_t>> MLPPCost::MSEDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPLinAlg alg;
	return alg.subtraction(y_hat, y);
}

real_t MLPPCost::RMSE(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += (y_hat[i] - y[i]) * (y_hat[i] - y[i]);
	}
	return sqrt(sum / y_hat.size());
}

real_t MLPPCost::RMSE(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += (y_hat[i][j] - y[i][j]) * (y_hat[i][j] - y[i][j]);
		}
	}
	return sqrt(sum / y_hat.size());
}

std::vector<real_t> MLPPCost::RMSEDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPLinAlg alg;
	return alg.scalarMultiply(1 / (2 * sqrt(MSE(y_hat, y))), MSEDeriv(y_hat, y));
}

std::vector<std::vector<real_t>> MLPPCost::RMSEDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPLinAlg alg;
	return alg.scalarMultiply(1 / (2 / sqrt(MSE(y_hat, y))), MSEDeriv(y_hat, y));
}

real_t MLPPCost::MAE(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += abs((y_hat[i] - y[i]));
	}
	return sum / y_hat.size();
}

real_t MLPPCost::MAE(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += abs((y_hat[i][j] - y[i][j]));
		}
	}
	return sum / y_hat.size();
}

std::vector<real_t> MLPPCost::MAEDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	std::vector<real_t> deriv;
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

std::vector<std::vector<real_t>> MLPPCost::MAEDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	std::vector<std::vector<real_t>> deriv;
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

real_t MLPPCost::MBE(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += (y_hat[i] - y[i]);
	}
	return sum / y_hat.size();
}

real_t MLPPCost::MBE(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += (y_hat[i][j] - y[i][j]);
		}
	}
	return sum / y_hat.size();
}

std::vector<real_t> MLPPCost::MBEDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPLinAlg alg;
	return alg.onevec(y_hat.size());
}

std::vector<std::vector<real_t>> MLPPCost::MBEDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPLinAlg alg;
	return alg.onemat(y_hat.size(), y_hat[0].size());
}

real_t MLPPCost::LogLoss(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	real_t eps = 1e-8;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += -(y[i] * std::log(y_hat[i] + eps) + (1 - y[i]) * std::log(1 - y_hat[i] + eps));
	}

	return sum / y_hat.size();
}

real_t MLPPCost::LogLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	real_t eps = 1e-8;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += -(y[i][j] * std::log(y_hat[i][j] + eps) + (1 - y[i][j]) * std::log(1 - y_hat[i][j] + eps));
		}
	}

	return sum / y_hat.size();
}

std::vector<real_t> MLPPCost::LogLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPLinAlg alg;
	return alg.addition(alg.scalarMultiply(-1, alg.elementWiseDivision(y, y_hat)), alg.elementWiseDivision(alg.scalarMultiply(-1, alg.scalarAdd(-1, y)), alg.scalarMultiply(-1, alg.scalarAdd(-1, y_hat))));
}

std::vector<std::vector<real_t>> MLPPCost::LogLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPLinAlg alg;
	return alg.addition(alg.scalarMultiply(-1, alg.elementWiseDivision(y, y_hat)), alg.elementWiseDivision(alg.scalarMultiply(-1, alg.scalarAdd(-1, y)), alg.scalarMultiply(-1, alg.scalarAdd(-1, y_hat))));
}

real_t MLPPCost::CrossEntropy(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += y[i] * std::log(y_hat[i]);
	}

	return -1 * sum;
}

real_t MLPPCost::CrossEntropy(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += y[i][j] * std::log(y_hat[i][j]);
		}
	}

	return -1 * sum;
}

std::vector<real_t> MLPPCost::CrossEntropyDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPLinAlg alg;
	return alg.scalarMultiply(-1, alg.elementWiseDivision(y, y_hat));
}

std::vector<std::vector<real_t>> MLPPCost::CrossEntropyDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPLinAlg alg;
	return alg.scalarMultiply(-1, alg.elementWiseDivision(y, y_hat));
}

real_t MLPPCost::HuberLoss(std::vector<real_t> y_hat, std::vector<real_t> y, real_t delta) {
	MLPPLinAlg alg;
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		if (abs(y[i] - y_hat[i]) <= delta) {
			sum += (y[i] - y_hat[i]) * (y[i] - y_hat[i]);
		} else {
			sum += 2 * delta * abs(y[i] - y_hat[i]) - delta * delta;
		}
	}
	return sum;
}

real_t MLPPCost::HuberLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y, real_t delta) {
	MLPPLinAlg alg;
	real_t sum = 0;
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

std::vector<real_t> MLPPCost::HuberLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y, real_t delta) {
	MLPPLinAlg alg;
	real_t sum = 0;
	std::vector<real_t> deriv;
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

std::vector<std::vector<real_t>> MLPPCost::HuberLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y, real_t delta) {
	MLPPLinAlg alg;
	real_t sum = 0;
	std::vector<std::vector<real_t>> deriv;
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

real_t MLPPCost::HingeLoss(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += fmax(0, 1 - y[i] * y_hat[i]);
	}

	return sum / y_hat.size();
}

real_t MLPPCost::HingeLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += fmax(0, 1 - y[i][j] * y_hat[i][j]);
		}
	}

	return sum / y_hat.size();
}

std::vector<real_t> MLPPCost::HingeLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	std::vector<real_t> deriv;
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

std::vector<std::vector<real_t>> MLPPCost::HingeLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	std::vector<std::vector<real_t>> deriv;
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

real_t MLPPCost::WassersteinLoss(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		sum += y_hat[i] * y[i];
	}
	return -sum / y_hat.size();
}

real_t MLPPCost::WassersteinLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t sum = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		for (int j = 0; j < y_hat[i].size(); j++) {
			sum += y_hat[i][j] * y[i][j];
		}
	}
	return -sum / y_hat.size();
}

std::vector<real_t> MLPPCost::WassersteinLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y) {
	MLPPLinAlg alg;
	return alg.scalarMultiply(-1, y); // Simple.
}

std::vector<std::vector<real_t>> MLPPCost::WassersteinLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	MLPPLinAlg alg;
	return alg.scalarMultiply(-1, y); // Simple.
}

real_t MLPPCost::HingeLoss(std::vector<real_t> y_hat, std::vector<real_t> y, std::vector<real_t> weights, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	return C * HingeLoss(y_hat, y) + regularization.regTerm(weights, 1, 0, "Ridge");
}
real_t MLPPCost::HingeLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y, std::vector<std::vector<real_t>> weights, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	return C * HingeLoss(y_hat, y) + regularization.regTerm(weights, 1, 0, "Ridge");
}

std::vector<real_t> MLPPCost::HingeLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	return alg.scalarMultiply(C, HingeLossDeriv(y_hat, y));
}
std::vector<std::vector<real_t>> MLPPCost::HingeLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y, real_t C) {
	MLPPLinAlg alg;
	MLPPReg regularization;
	return alg.scalarMultiply(C, HingeLossDeriv(y_hat, y));
}

real_t MLPPCost::dualFormSVM(std::vector<real_t> alpha, std::vector<std::vector<real_t>> X, std::vector<real_t> y) {
	MLPPLinAlg alg;
	std::vector<std::vector<real_t>> Y = alg.diag(y); // Y is a diagnoal matrix. Y[i][j] = y[i] if i = i, else Y[i][j] = 0. Yt = Y.
	std::vector<std::vector<real_t>> K = alg.matmult(X, alg.transpose(X)); // TO DO: DON'T forget to add non-linear kernelizations.
	std::vector<std::vector<real_t>> Q = alg.matmult(alg.matmult(alg.transpose(Y), K), Y);
	real_t alphaQ = alg.matmult(alg.matmult({ alpha }, Q), alg.transpose({ alpha }))[0][0];
	std::vector<real_t> one = alg.onevec(alpha.size());

	return -alg.dot(one, alpha) + 0.5 * alphaQ;
}

std::vector<real_t> MLPPCost::dualFormSVMDeriv(std::vector<real_t> alpha, std::vector<std::vector<real_t>> X, std::vector<real_t> y) {
	MLPPLinAlg alg;
	std::vector<std::vector<real_t>> Y = alg.zeromat(y.size(), y.size());
	for (int i = 0; i < y.size(); i++) {
		Y[i][i] = y[i]; // Y is a diagnoal matrix. Y[i][j] = y[i] if i = i, else Y[i][j] = 0. Yt = Y.
	}
	std::vector<std::vector<real_t>> K = alg.matmult(X, alg.transpose(X)); // TO DO: DON'T forget to add non-linear kernelizations.
	std::vector<std::vector<real_t>> Q = alg.matmult(alg.matmult(alg.transpose(Y), K), Y);
	std::vector<real_t> alphaQDeriv = alg.mat_vec_mult(Q, alpha);
	std::vector<real_t> one = alg.onevec(alpha.size());

	return alg.subtraction(alphaQDeriv, one);
}
