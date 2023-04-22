
#ifndef MLPP_COST_H
#define MLPP_COST_H

//
//  Cost.hpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include <vector>

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

//void set_weights(const Ref<MLPPMatrix> &val);
//void set_bias(const Ref<MLPPVector> &val);

class MLPPCost : public Reference {
	GDCLASS(MLPPCost, Reference);

public:
	enum CostTypes {
		COST_TYPE_MSE = 0,
		COST_TYPE_RMSE,
		COST_TYPE_MAE,
		COST_TYPE_MBE,
		COST_TYPE_LOGISTIC_LOSS,
		COST_TYPE_CROSS_ENTROPY,
		COST_TYPE_HINGE_LOSS,
		COST_TYPE_WASSERSTEIN_LOSS,
	};

public:
	real_t msev(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	real_t msem(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	Ref<MLPPVector> mse_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	Ref<MLPPMatrix> mse_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	real_t rmsev(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	real_t rmsem(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	Ref<MLPPVector> rmse_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	Ref<MLPPMatrix> rmse_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	real_t maev(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	real_t maem(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	Ref<MLPPVector> mae_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	Ref<MLPPMatrix> mae_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	real_t mbev(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	real_t mbem(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	Ref<MLPPVector> mbe_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	Ref<MLPPMatrix> mbe_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	// Classification Costs
	real_t log_lossv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	real_t log_lossm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	Ref<MLPPVector> log_loss_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	Ref<MLPPMatrix> log_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	real_t cross_entropyv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	real_t cross_entropym(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	Ref<MLPPVector> cross_entropy_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	Ref<MLPPMatrix> cross_entropy_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	real_t huber_lossv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y, real_t delta);
	real_t huber_lossm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y, real_t delta);

	Ref<MLPPVector> huber_loss_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y, real_t delta);
	Ref<MLPPMatrix> huber_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y, real_t delta);

	real_t hinge_lossv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	real_t hinge_lossm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	Ref<MLPPVector> hinge_loss_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	Ref<MLPPMatrix> hinge_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	real_t hinge_losswv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y, const Ref<MLPPVector> &weights, real_t C);
	real_t hinge_losswm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y, const Ref<MLPPMatrix> &weights, real_t C);

	Ref<MLPPVector> hinge_loss_derivwv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y, real_t C);
	Ref<MLPPMatrix> hinge_loss_derivwm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y, real_t C);

	real_t wasserstein_lossv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	real_t wasserstein_lossm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	Ref<MLPPVector> wasserstein_loss_derivv(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	Ref<MLPPMatrix> wasserstein_loss_derivm(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	real_t dual_form_svm(const Ref<MLPPVector> &alpha, const Ref<MLPPMatrix> &X, const Ref<MLPPVector> &y); // TO DO: DON'T forget to add non-linear kernelizations.

	Ref<MLPPVector> dual_form_svm_deriv(const Ref<MLPPVector> &alpha, const Ref<MLPPMatrix> &X, const Ref<MLPPVector> &y);

	typedef real_t (MLPPCost::*VectorCostFunctionPointer)(const Ref<MLPPVector> &, const Ref<MLPPVector> &);
	typedef real_t (MLPPCost::*MatrixCostFunctionPointer)(const Ref<MLPPMatrix> &, const Ref<MLPPMatrix> &);

	typedef Ref<MLPPVector> (MLPPCost::*VectorDerivCostFunctionPointer)(const Ref<MLPPVector> &, const Ref<MLPPVector> &);
	typedef Ref<MLPPMatrix> (MLPPCost::*MatrixDerivCostFunctionPointer)(const Ref<MLPPMatrix> &, const Ref<MLPPMatrix> &);

	VectorCostFunctionPointer get_cost_function_ptr_normal_vector(const CostTypes cost);
	MatrixCostFunctionPointer get_cost_function_ptr_normal_matrix(const CostTypes cost);

	VectorDerivCostFunctionPointer get_cost_function_ptr_deriv_vector(const CostTypes cost);
	MatrixDerivCostFunctionPointer get_cost_function_ptr_deriv_matrix(const CostTypes cost);

	real_t run_cost_norm_vector(const CostTypes cost, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	real_t run_cost_norm_matrix(const CostTypes cost, const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

	Ref<MLPPVector> run_cost_deriv_vector(const CostTypes cost, const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);
	Ref<MLPPMatrix> run_cost_deriv_matrix(const CostTypes cost, const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y);

protected:
	static void _bind_methods();
};

VARIANT_ENUM_CAST(MLPPCost::CostTypes);

#endif /* Cost_hpp */
