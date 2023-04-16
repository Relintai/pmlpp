//
//  NumericalAnalysis.cpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "numerical_analysis.h"
#include "../lin_alg/lin_alg.h"

#include <climits>
#include <cmath>
#include <iostream>
#include <string>

real_t MLPPNumericalAnalysis::numDiff(real_t (*function)(real_t), real_t x) {
	real_t eps = 1e-10;
	return (function(x + eps) - function(x)) / eps; // This is just the formal def. of the derivative.
}

real_t MLPPNumericalAnalysis::numDiff_2(real_t (*function)(real_t), real_t x) {
	real_t eps = 1e-5;
	return (function(x + 2 * eps) - 2 * function(x + eps) + function(x)) / (eps * eps);
}

real_t MLPPNumericalAnalysis::numDiff_3(real_t (*function)(real_t), real_t x) {
	real_t eps = 1e-5;
	real_t t1 = function(x + 3 * eps) - 2 * function(x + 2 * eps) + function(x + eps);
	real_t t2 = function(x + 2 * eps) - 2 * function(x + eps) + function(x);
	return (t1 - t2) / (eps * eps * eps);
}

real_t MLPPNumericalAnalysis::constantApproximation(real_t (*function)(real_t), real_t c) {
	return function(c);
}

real_t MLPPNumericalAnalysis::linearApproximation(real_t (*function)(real_t), real_t c, real_t x) {
	return constantApproximation(function, c) + numDiff(function, c) * (x - c);
}

real_t MLPPNumericalAnalysis::quadraticApproximation(real_t (*function)(real_t), real_t c, real_t x) {
	return linearApproximation(function, c, x) + 0.5 * numDiff_2(function, c) * (x - c) * (x - c);
}

real_t MLPPNumericalAnalysis::cubicApproximation(real_t (*function)(real_t), real_t c, real_t x) {
	return quadraticApproximation(function, c, x) + (1 / 6) * numDiff_3(function, c) * (x - c) * (x - c) * (x - c);
}

real_t MLPPNumericalAnalysis::numDiff(real_t (*function)(std::vector<real_t>), std::vector<real_t> x, int axis) {
	// For multivariable function analysis.
	// This will be used for calculating Jacobian vectors.
	// Diffrentiate with respect to indicated axis. (0, 1, 2 ...)
	real_t eps = 1e-10;
	std::vector<real_t> x_eps = x;
	x_eps[axis] += eps;

	return (function(x_eps) - function(x)) / eps;
}

real_t MLPPNumericalAnalysis::numDiff_2(real_t (*function)(std::vector<real_t>), std::vector<real_t> x, int axis1, int axis2) {
	//For Hessians.
	real_t eps = 1e-5;

	std::vector<real_t> x_pp = x;
	x_pp[axis1] += eps;
	x_pp[axis2] += eps;

	std::vector<real_t> x_np = x;
	x_np[axis2] += eps;

	std::vector<real_t> x_pn = x;
	x_pn[axis1] += eps;

	return (function(x_pp) - function(x_np) - function(x_pn) + function(x)) / (eps * eps);
}

real_t MLPPNumericalAnalysis::numDiff_3(real_t (*function)(std::vector<real_t>), std::vector<real_t> x, int axis1, int axis2, int axis3) {
	// For third order derivative tensors.
	// NOTE: Approximations do not appear to be accurate for sinusodial functions...
	// Should revisit this later.
	real_t eps = 1e-5;

	std::vector<real_t> x_ppp = x;
	x_ppp[axis1] += eps;
	x_ppp[axis2] += eps;
	x_ppp[axis3] += eps;

	std::vector<real_t> x_npp = x;
	x_npp[axis2] += eps;
	x_npp[axis3] += eps;

	std::vector<real_t> x_pnp = x;
	x_pnp[axis1] += eps;
	x_pnp[axis3] += eps;

	std::vector<real_t> x_nnp = x;
	x_nnp[axis3] += eps;

	std::vector<real_t> x_ppn = x;
	x_ppn[axis1] += eps;
	x_ppn[axis2] += eps;

	std::vector<real_t> x_npn = x;
	x_npn[axis2] += eps;

	std::vector<real_t> x_pnn = x;
	x_pnn[axis1] += eps;

	real_t thirdAxis = function(x_ppp) - function(x_npp) - function(x_pnp) + function(x_nnp);
	real_t noThirdAxis = function(x_ppn) - function(x_npn) - function(x_pnn) + function(x);
	return (thirdAxis - noThirdAxis) / (eps * eps * eps);
}

real_t MLPPNumericalAnalysis::newtonRaphsonMethod(real_t (*function)(real_t), real_t x_0, real_t epoch_num) {
	real_t x = x_0;
	for (int i = 0; i < epoch_num; i++) {
		x -= function(x) / numDiff(function, x);
	}
	return x;
}

real_t MLPPNumericalAnalysis::halleyMethod(real_t (*function)(real_t), real_t x_0, real_t epoch_num) {
	real_t x = x_0;
	for (int i = 0; i < epoch_num; i++) {
		x -= ((2 * function(x) * numDiff(function, x)) / (2 * numDiff(function, x) * numDiff(function, x) - function(x) * numDiff_2(function, x)));
	}
	return x;
}

real_t MLPPNumericalAnalysis::invQuadraticInterpolation(real_t (*function)(real_t), std::vector<real_t> x_0, int epoch_num) {
	real_t x = 0;
	std::vector<real_t> currentThree = x_0;
	for (int i = 0; i < epoch_num; i++) {
		real_t t1 = ((function(currentThree[1]) * function(currentThree[2])) / ((function(currentThree[0]) - function(currentThree[1])) * (function(currentThree[0]) - function(currentThree[2])))) * currentThree[0];
		real_t t2 = ((function(currentThree[0]) * function(currentThree[2])) / ((function(currentThree[1]) - function(currentThree[0])) * (function(currentThree[1]) - function(currentThree[2])))) * currentThree[1];
		real_t t3 = ((function(currentThree[0]) * function(currentThree[1])) / ((function(currentThree[2]) - function(currentThree[0])) * (function(currentThree[2]) - function(currentThree[1])))) * currentThree[2];
		x = t1 + t2 + t3;

		currentThree.erase(currentThree.begin());
		currentThree.push_back(x);
	}
	return x;
}

real_t MLPPNumericalAnalysis::eulerianMethod(real_t (*derivative)(real_t), std::vector<real_t> q_0, real_t p, real_t h) {
	int max_epoch = static_cast<int>((p - q_0[0]) / h);
	real_t x = q_0[0];
	real_t y = q_0[1];
	for (int i = 0; i < max_epoch; i++) {
		y = y + h * derivative(x);
		x += h;
	}
	return y;
}

real_t MLPPNumericalAnalysis::eulerianMethod(real_t (*derivative)(std::vector<real_t>), std::vector<real_t> q_0, real_t p, real_t h) {
	int max_epoch = static_cast<int>((p - q_0[0]) / h);
	real_t x = q_0[0];
	real_t y = q_0[1];
	for (int i = 0; i < max_epoch; i++) {
		y = y + h * derivative({ x, y });
		x += h;
	}
	return y;
}

real_t MLPPNumericalAnalysis::growthMethod(real_t C, real_t k, real_t t) {
	/*
	dP/dt = kP
	dP/P = kdt
	integral(1/P)dP = integral(k) dt
	ln|P| = kt + C_initial
	|P| = e^(kt + C_initial)
	|P| = e^(C_initial) * e^(kt)
	P = +/- e^(C_initial) * e^(kt)
	P = C * e^(kt)
	*/

	// auto growthFunction = [&C, &k](real_t t) { return C * exp(k * t); };
	return C * std::exp(k * t);
}

std::vector<real_t> MLPPNumericalAnalysis::jacobian(real_t (*function)(std::vector<real_t>), std::vector<real_t> x) {
	std::vector<real_t> jacobian;
	jacobian.resize(x.size());
	for (uint32_t i = 0; i < jacobian.size(); i++) {
		jacobian[i] = numDiff(function, x, i); // Derivative w.r.t axis i evaluated at x. For all x_i.
	}
	return jacobian;
}
std::vector<std::vector<real_t>> MLPPNumericalAnalysis::hessian(real_t (*function)(std::vector<real_t>), std::vector<real_t> x) {
	std::vector<std::vector<real_t>> hessian;
	hessian.resize(x.size());

	for (uint32_t i = 0; i < hessian.size(); i++) {
		hessian[i].resize(x.size());
	}

	for (uint32_t i = 0; i < hessian.size(); i++) {
		for (uint32_t j = 0; j < hessian[i].size(); j++) {
			hessian[i][j] = numDiff_2(function, x, i, j);
		}
	}

	return hessian;
}

std::vector<std::vector<std::vector<real_t>>> MLPPNumericalAnalysis::thirdOrderTensor(real_t (*function)(std::vector<real_t>), std::vector<real_t> x) {
	std::vector<std::vector<std::vector<real_t>>> tensor;
	tensor.resize(x.size());

	for (uint32_t i = 0; i < tensor.size(); i++) {
		tensor[i].resize(x.size());
		for (uint32_t j = 0; j < tensor[i].size(); j++) {
			tensor[i][j].resize(x.size());
		}
	}

	for (uint32_t i = 0; i < tensor.size(); i++) { // O(n^3) time complexity :(
		for (uint32_t j = 0; j < tensor[i].size(); j++) {
			for (uint32_t k = 0; k < tensor[i][j].size(); k++)
				tensor[i][j][k] = numDiff_3(function, x, i, j, k);
		}
	}

	return tensor;
}

real_t MLPPNumericalAnalysis::constantApproximation(real_t (*function)(std::vector<real_t>), std::vector<real_t> c) {
	return function(c);
}

real_t MLPPNumericalAnalysis::linearApproximation(real_t (*function)(std::vector<real_t>), std::vector<real_t> c, std::vector<real_t> x) {
	MLPPLinAlg alg;
	return constantApproximation(function, c) + alg.matmult(alg.transpose({ jacobian(function, c) }), { alg.subtraction(x, c) })[0][0];
}

real_t MLPPNumericalAnalysis::quadraticApproximation(real_t (*function)(std::vector<real_t>), std::vector<real_t> c, std::vector<real_t> x) {
	MLPPLinAlg alg;
	return linearApproximation(function, c, x) + 0.5 * alg.matmult({ (alg.subtraction(x, c)) }, alg.matmult(hessian(function, c), alg.transpose({ alg.subtraction(x, c) })))[0][0];
}

real_t MLPPNumericalAnalysis::cubicApproximation(real_t (*function)(std::vector<real_t>), std::vector<real_t> c, std::vector<real_t> x) {
	/*
	Not completely sure as the literature seldom discusses the third order taylor approximation,
	in particular for multivariate cases, but ostensibly, the matrix/tensor/vector multiplies
	should look something like this:

	(N x N x N) (N x 1) [tensor vector mult] => (N x N x 1) => (N x N)
	Perform remaining multiplies as done for the 2nd order approximation.
	Result is a scalar.
	*/
	MLPPLinAlg alg;
	std::vector<std::vector<real_t>> resultMat = alg.tensor_vec_mult(thirdOrderTensor(function, c), alg.subtraction(x, c));
	real_t resultScalar = alg.matmult({ (alg.subtraction(x, c)) }, alg.matmult(resultMat, alg.transpose({ alg.subtraction(x, c) })))[0][0];

	return quadraticApproximation(function, c, x) + (1 / 6) * resultScalar;
}

real_t MLPPNumericalAnalysis::laplacian(real_t (*function)(std::vector<real_t>), std::vector<real_t> x) {
	std::vector<std::vector<real_t>> hessian_matrix = hessian(function, x);
	real_t laplacian = 0;

	for (uint32_t i = 0; i < hessian_matrix.size(); i++) {
		laplacian += hessian_matrix[i][i]; // homogenous 2nd derivs w.r.t i, then i
	}

	return laplacian;
}

std::string MLPPNumericalAnalysis::secondPartialDerivativeTest(real_t (*function)(std::vector<real_t>), std::vector<real_t> x) {
	MLPPLinAlg alg;
	std::vector<std::vector<real_t>> hessianMatrix = hessian(function, x);
	/*
	The reason we do this is because the 2nd partial derivative test is less conclusive for functions of variables greater than
	2, and the calculations specific to the bivariate case are less computationally intensive.
	*/
	if (x.size() == 2) {
		real_t det = alg.det(hessianMatrix, hessianMatrix.size());
		real_t secondDerivative = numDiff_2(function, x, 0, 0);
		if (secondDerivative > 0 && det > 0) {
			return "min";
		} else if (secondDerivative < 0 && det > 0) {
			return "max";
		} else if (det < 0) {
			return "saddle";
		} else {
			return "test was inconclusive";
		}
	} else {
		if (alg.positiveDefiniteChecker(hessianMatrix)) {
			return "min";
		} else if (alg.negativeDefiniteChecker(hessianMatrix)) {
			return "max";
		} else if (!alg.zeroEigenvalue(hessianMatrix)) {
			return "saddle";
		} else {
			return "test was inconclusive";
		}
	}
}

void MLPPNumericalAnalysis::_bind_methods() {
}
