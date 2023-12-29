
#ifndef MLPP_NUMERICAL_ANALYSIS_H
#define MLPP_NUMERICAL_ANALYSIS_H

//
//  NumericalAnalysis.hpp
//
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "core/containers/vector.h"
#include "core/string/ustring.h"

class MLPPVector;
class MLPPMatrix;
class MLPPTensor3;

class MLPPNumericalAnalysis : public Reference {
	GDCLASS(MLPPNumericalAnalysis, Reference);

public:
	/* A numerical method for derivatives is used. This may be subject to change,
	as an analytical method for calculating derivatives will most likely be used in
	the future.
	*/

	real_t num_diffr(real_t (*function)(real_t), real_t x);
	real_t num_diff_2r(real_t (*function)(real_t), real_t x);
	real_t num_diff_3r(real_t (*function)(real_t), real_t x);

	real_t constant_approximationr(real_t (*function)(real_t), real_t c);
	real_t linear_approximationr(real_t (*function)(real_t), real_t c, real_t x);
	real_t quadratic_approximationr(real_t (*function)(real_t), real_t c, real_t x);
	real_t cubic_approximationr(real_t (*function)(real_t), real_t c, real_t x);

	real_t num_diffv(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &, int axis);
	real_t num_diff_2v(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &, int axis1, int axis2);
	real_t num_diff_3v(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &, int axis1, int axis2, int axis3);

	real_t newton_raphson_method(real_t (*function)(real_t), real_t x_0, real_t epoch_num);
	real_t halley_method(real_t (*function)(real_t), real_t x_0, real_t epoch_num);
	real_t inv_quadratic_interpolation(real_t (*function)(real_t), const Ref<MLPPVector> &x_0, int epoch_num);

	real_t eulerian_methodr(real_t (*derivative)(real_t), real_t q_0, real_t q_1, real_t p, real_t h); // Euler's method for solving diffrential equations.
	real_t eulerian_methodv(real_t (*derivative)(const Ref<MLPPVector> &), real_t q_0, real_t q_1, real_t p, real_t h); // Euler's method for solving diffrential equations.

	real_t growth_method(real_t C, real_t k, real_t t); // General growth-based diffrential equations can be solved by seperation of variables.

	Ref<MLPPVector> jacobian(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x); // Indeed, for functions with scalar outputs the Jacobians will be vectors.
	Ref<MLPPMatrix> hessian(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x);
	Ref<MLPPTensor3> third_order_tensor(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x);
	Vector<Ref<MLPPMatrix>> third_order_tensorvt(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x);

	real_t constant_approximationv(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &c);
	real_t linear_approximationv(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &c, const Ref<MLPPVector> &x);
	real_t quadratic_approximationv(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &c, const Ref<MLPPVector> &x);
	real_t cubic_approximationv(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &c, const Ref<MLPPVector> &x);

	real_t laplacian(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x); // laplacian

	String second_partial_derivative_test(real_t (*function)(const Ref<MLPPVector> &), const Ref<MLPPVector> &x);

protected:
	static void _bind_methods();
};

#endif /* NumericalAnalysis_hpp */
