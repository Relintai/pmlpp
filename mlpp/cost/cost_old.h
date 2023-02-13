
#ifndef MLPP_COST_OLD_H
#define MLPP_COST_OLD_H

//
//  Cost.hpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "core/math/math_defs.h"

#include <vector>

class MLPPCostOld {
public:
	// Regression Costs
	real_t MSE(std::vector<real_t> y_hat, std::vector<real_t> y);
	real_t MSE(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	std::vector<real_t> MSEDeriv(std::vector<real_t> y_hat, std::vector<real_t> y);
	std::vector<std::vector<real_t>> MSEDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	real_t RMSE(std::vector<real_t> y_hat, std::vector<real_t> y);
	real_t RMSE(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	std::vector<real_t> RMSEDeriv(std::vector<real_t> y_hat, std::vector<real_t> y);
	std::vector<std::vector<real_t>> RMSEDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	real_t MAE(std::vector<real_t> y_hat, std::vector<real_t> y);
	real_t MAE(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	std::vector<real_t> MAEDeriv(std::vector<real_t> y_hat, std::vector<real_t> y);
	std::vector<std::vector<real_t>> MAEDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	real_t MBE(std::vector<real_t> y_hat, std::vector<real_t> y);
	real_t MBE(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	std::vector<real_t> MBEDeriv(std::vector<real_t> y_hat, std::vector<real_t> y);
	std::vector<std::vector<real_t>> MBEDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	// Classification Costs
	real_t LogLoss(std::vector<real_t> y_hat, std::vector<real_t> y);
	real_t LogLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	std::vector<real_t> LogLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y);
	std::vector<std::vector<real_t>> LogLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	real_t CrossEntropy(std::vector<real_t> y_hat, std::vector<real_t> y);
	real_t CrossEntropy(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	std::vector<real_t> CrossEntropyDeriv(std::vector<real_t> y_hat, std::vector<real_t> y);
	std::vector<std::vector<real_t>> CrossEntropyDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	real_t HuberLoss(std::vector<real_t> y_hat, std::vector<real_t> y, real_t delta);
	real_t HuberLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y, real_t delta);

	std::vector<real_t> HuberLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y, real_t delta);
	std::vector<std::vector<real_t>> HuberLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y, real_t delta);

	real_t HingeLoss(std::vector<real_t> y_hat, std::vector<real_t> y);
	real_t HingeLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	std::vector<real_t> HingeLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y);
	std::vector<std::vector<real_t>> HingeLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	real_t HingeLoss(std::vector<real_t> y_hat, std::vector<real_t> y, std::vector<real_t> weights, real_t C);
	real_t HingeLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y, std::vector<std::vector<real_t>> weights, real_t C);

	std::vector<real_t> HingeLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y, real_t C);
	std::vector<std::vector<real_t>> HingeLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y, real_t C);

	real_t WassersteinLoss(std::vector<real_t> y_hat, std::vector<real_t> y);
	real_t WassersteinLoss(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	std::vector<real_t> WassersteinLossDeriv(std::vector<real_t> y_hat, std::vector<real_t> y);
	std::vector<std::vector<real_t>> WassersteinLossDeriv(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	real_t dualFormSVM(std::vector<real_t> alpha, std::vector<std::vector<real_t>> X, std::vector<real_t> y); // TO DO: DON'T forget to add non-linear kernelizations.

	std::vector<real_t> dualFormSVMDeriv(std::vector<real_t> alpha, std::vector<std::vector<real_t>> X, std::vector<real_t> y);
};

#endif /* Cost_hpp */
