
#ifndef MLPP_NUMERICAL_ANALYSIS_H
#define MLPP_NUMERICAL_ANALYSIS_H

//
//  NumericalAnalysis.hpp
//
//

#include "core/math/math_defs.h"

#include <string>
#include <vector>


class MLPPNumericalAnalysis {
public:
	/* A numerical method for derivatives is used. This may be subject to change,
	as an analytical method for calculating derivatives will most likely be used in
	the future.
	*/
	real_t numDiff(real_t (*function)(real_t), real_t x);
	real_t numDiff_2(real_t (*function)(real_t), real_t x);
	real_t numDiff_3(real_t (*function)(real_t), real_t x);

	real_t constantApproximation(real_t (*function)(real_t), real_t c);
	real_t linearApproximation(real_t (*function)(real_t), real_t c, real_t x);
	real_t quadraticApproximation(real_t (*function)(real_t), real_t c, real_t x);
	real_t cubicApproximation(real_t (*function)(real_t), real_t c, real_t x);

	real_t numDiff(real_t (*function)(std::vector<real_t>), std::vector<real_t> x, int axis);
	real_t numDiff_2(real_t (*function)(std::vector<real_t>), std::vector<real_t> x, int axis1, int axis2);
	real_t numDiff_3(real_t (*function)(std::vector<real_t>), std::vector<real_t> x, int axis1, int axis2, int axis3);

	real_t newtonRaphsonMethod(real_t (*function)(real_t), real_t x_0, real_t epoch_num);
	real_t halleyMethod(real_t (*function)(real_t), real_t x_0, real_t epoch_num);
	real_t invQuadraticInterpolation(real_t (*function)(real_t), std::vector<real_t> x_0, real_t epoch_num);

	real_t eulerianMethod(real_t (*derivative)(real_t), std::vector<real_t> q_0, real_t p, real_t h); // Euler's method for solving diffrential equations.
	real_t eulerianMethod(real_t (*derivative)(std::vector<real_t>), std::vector<real_t> q_0, real_t p, real_t h); // Euler's method for solving diffrential equations.

	real_t growthMethod(real_t C, real_t k, real_t t); // General growth-based diffrential equations can be solved by seperation of variables.

	std::vector<real_t> jacobian(real_t (*function)(std::vector<real_t>), std::vector<real_t> x); // Indeed, for functions with scalar outputs the Jacobians will be vectors.
	std::vector<std::vector<real_t>> hessian(real_t (*function)(std::vector<real_t>), std::vector<real_t> x);
	std::vector<std::vector<std::vector<real_t>>> thirdOrderTensor(real_t (*function)(std::vector<real_t>), std::vector<real_t> x);

	real_t constantApproximation(real_t (*function)(std::vector<real_t>), std::vector<real_t> c);
	real_t linearApproximation(real_t (*function)(std::vector<real_t>), std::vector<real_t> c, std::vector<real_t> x);
	real_t quadraticApproximation(real_t (*function)(std::vector<real_t>), std::vector<real_t> c, std::vector<real_t> x);
	real_t cubicApproximation(real_t (*function)(std::vector<real_t>), std::vector<real_t> c, std::vector<real_t> x);

	real_t laplacian(real_t (*function)(std::vector<real_t>), std::vector<real_t> x); // laplacian

	std::string secondPartialDerivativeTest(real_t (*function)(std::vector<real_t>), std::vector<real_t> x);
};


#endif /* NumericalAnalysis_hpp */
