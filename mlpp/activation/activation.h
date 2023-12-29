
#ifndef MLPP_ACTIVATION_H
#define MLPP_ACTIVATION_H



#include "core/math/math_defs.h"

#include "core/object/func_ref.h"
#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <vector>

//TODO Activation functions should either have a variant which does not allocate, or they should just be reworked altogether
//TODO Methods here should probably use error macros, in a way where they get disabled in non-tools(?) (maybe release?) builds

class MLPPActivation : public Reference {
	GDCLASS(MLPPActivation, Reference);

public:
	enum ActivationFunction {
		ACTIVATION_FUNCTION_LINEAR = 0,
		ACTIVATION_FUNCTION_SIGMOID,
		ACTIVATION_FUNCTION_SWISH,
		ACTIVATION_FUNCTION_MISH,
		ACTIVATION_FUNCTION_SIN_C,
		ACTIVATION_FUNCTION_SOFTMAX,
		ACTIVATION_FUNCTION_SOFTPLUS,
		ACTIVATION_FUNCTION_SOFTSIGN,
		ACTIVATION_FUNCTION_ADJ_SOFTMAX,
		ACTIVATION_FUNCTION_C_LOG_LOG,
		ACTIVATION_FUNCTION_LOGIT,
		ACTIVATION_FUNCTION_GAUSSIAN_CDF,
		ACTIVATION_FUNCTION_RELU,
		ACTIVATION_FUNCTION_GELU,
		ACTIVATION_FUNCTION_SIGN,
		ACTIVATION_FUNCTION_UNIT_STEP,
		ACTIVATION_FUNCTION_SINH,
		ACTIVATION_FUNCTION_COSH,
		ACTIVATION_FUNCTION_TANH,
		ACTIVATION_FUNCTION_CSCH,
		ACTIVATION_FUNCTION_SECH,
		ACTIVATION_FUNCTION_COTH,
		ACTIVATION_FUNCTION_ARSINH,
		ACTIVATION_FUNCTION_ARCOSH,
		ACTIVATION_FUNCTION_ARTANH,
		ACTIVATION_FUNCTION_ARCSCH,
		ACTIVATION_FUNCTION_ARSECH,
		ACTIVATION_FUNCTION_ARCOTH,
	};

public:
	typedef real_t (MLPPActivation::*RealActivationFunctionPointer)(real_t);
	typedef Ref<MLPPVector> (MLPPActivation::*VectorActivationFunctionPointer)(const Ref<MLPPVector> &);
	typedef Ref<MLPPMatrix> (MLPPActivation::*MatrixActivationFunctionPointer)(const Ref<MLPPMatrix> &);

	RealActivationFunctionPointer get_activation_function_ptr_real(const ActivationFunction func, const bool deriv = false);
	VectorActivationFunctionPointer get_activation_function_ptr_vector(const ActivationFunction func, const bool deriv = false);
	MatrixActivationFunctionPointer get_activation_function_ptr_matrix(const ActivationFunction func, const bool deriv = false);

	RealActivationFunctionPointer get_activation_function_ptr_normal_real(const ActivationFunction func);
	VectorActivationFunctionPointer get_activation_function_ptr_normal_vector(const ActivationFunction func);
	MatrixActivationFunctionPointer get_activation_function_ptr_normal_matrix(const ActivationFunction func);

	RealActivationFunctionPointer get_activation_function_ptr_deriv_real(const ActivationFunction func);
	VectorActivationFunctionPointer get_activation_function_ptr_deriv_vector(const ActivationFunction func);
	MatrixActivationFunctionPointer get_activation_function_ptr_deriv_matrix(const ActivationFunction func);

	real_t run_activation_real(const ActivationFunction func, const real_t z, const bool deriv = false);
	Ref<MLPPVector> run_activation_vector(const ActivationFunction func, const Ref<MLPPVector> &z, const bool deriv = false);
	Ref<MLPPMatrix> run_activation_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z, const bool deriv = false);

	real_t run_activation_norm_real(const ActivationFunction func, const real_t z);
	Ref<MLPPVector> run_activation_norm_vector(const ActivationFunction func, const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> run_activation_norm_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z);

	real_t run_activation_deriv_real(const ActivationFunction func, const real_t z);
	Ref<MLPPVector> run_activation_deriv_vector(const ActivationFunction func, const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> run_activation_deriv_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z);

	Ref<MLPPVector> activationr(const Ref<MLPPVector> &z, real_t (*function)(real_t));

	//ACTIVATION FUNCTIONS

	//LINEAR

	real_t linear_normr(real_t z);
	Ref<MLPPVector> linear_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> linear_normm(const Ref<MLPPMatrix> &z);

	real_t linear_derivr(real_t z);
	Ref<MLPPVector> linear_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> linear_derivm(const Ref<MLPPMatrix> &z);

	//SIGMOID

	real_t sigmoid_normr(real_t z);
	Ref<MLPPVector> sigmoid_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sigmoid_normm(const Ref<MLPPMatrix> &z);

	real_t sigmoid_derivr(real_t z);
	Ref<MLPPVector> sigmoid_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sigmoid_derivm(const Ref<MLPPMatrix> &z);

	//SOFTMAX

	real_t softmax_normr(real_t z);
	Ref<MLPPVector> softmax_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> softmax_normm(const Ref<MLPPMatrix> &z);

	real_t softmax_derivr(real_t z);
	Ref<MLPPVector> softmax_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> softmax_derivm(const Ref<MLPPMatrix> &z);

	//ADJ_SOFTMAX

	real_t adj_softmax_normr(real_t z);
	Ref<MLPPVector> adj_softmax_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> adj_softmax_normm(const Ref<MLPPMatrix> &z);

	real_t adj_softmax_derivr(real_t z);
	Ref<MLPPVector> adj_softmax_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> adj_softmax_derivm(const Ref<MLPPMatrix> &z);

	//SOFTMAX DERIV

	Ref<MLPPMatrix> softmax_deriv_normv(const Ref<MLPPVector> &z);
	Vector<Ref<MLPPMatrix>> softmax_deriv_normm(const Ref<MLPPMatrix> &z);

	Ref<MLPPMatrix> softmax_deriv_derivv(const Ref<MLPPVector> &z);
	Vector<Ref<MLPPMatrix>> softmax_deriv_derivm(const Ref<MLPPMatrix> &z);

	//SOFTPLUS

	real_t softplus_normr(real_t z);
	Ref<MLPPVector> softplus_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> softplus_normm(const Ref<MLPPMatrix> &z);

	real_t softplus_derivr(real_t z);
	Ref<MLPPVector> softplus_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> softplus_derivm(const Ref<MLPPMatrix> &z);

	//SOFTSIGN

	real_t softsign_normr(real_t z);
	Ref<MLPPVector> softsign_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> softsign_normm(const Ref<MLPPMatrix> &z);

	real_t softsign_derivr(real_t z);
	Ref<MLPPVector> softsign_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> softsign_derivm(const Ref<MLPPMatrix> &z);

	//GAUSSIANCDF

	real_t gaussian_cdf_normr(real_t z);
	Ref<MLPPVector> gaussian_cdf_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> gaussian_cdf_normm(const Ref<MLPPMatrix> &z);

	real_t gaussian_cdf_derivr(real_t z);
	Ref<MLPPVector> gaussian_cdf_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> gaussian_cdf_derivm(const Ref<MLPPMatrix> &z);

	//CLOGLOG

	real_t cloglog_normr(real_t z);
	Ref<MLPPVector> cloglog_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> cloglog_normm(const Ref<MLPPMatrix> &z);

	real_t cloglog_derivr(real_t z);
	Ref<MLPPVector> cloglog_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> cloglog_derivm(const Ref<MLPPMatrix> &z);

	//LOGIT

	real_t logit_normr(real_t z);
	Ref<MLPPVector> logit_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> logit_normm(const Ref<MLPPMatrix> &z);

	real_t logit_derivr(real_t z);
	Ref<MLPPVector> logit_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> logit_derivm(const Ref<MLPPMatrix> &z);

	//UNITSTEP

	real_t unit_step_normr(real_t z);
	Ref<MLPPVector> unit_step_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> unit_step_normm(const Ref<MLPPMatrix> &z);

	real_t unit_step_derivr(real_t z);
	Ref<MLPPVector> unit_step_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> unit_step_derivm(const Ref<MLPPMatrix> &z);

	//SWISH

	real_t swish_normr(real_t z);
	Ref<MLPPVector> swish_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> swish_normm(const Ref<MLPPMatrix> &z);

	real_t swish_derivr(real_t z);
	Ref<MLPPVector> swish_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> swish_derivm(const Ref<MLPPMatrix> &z);

	//MISH

	real_t mish_normr(real_t z);
	Ref<MLPPVector> mish_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> mish_normm(const Ref<MLPPMatrix> &z);

	real_t mish_derivr(real_t z);
	Ref<MLPPVector> mish_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> mish_derivm(const Ref<MLPPMatrix> &z);

	//SINC

	real_t sinc_normr(real_t z);
	Ref<MLPPVector> sinc_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sinc_normm(const Ref<MLPPMatrix> &z);

	real_t sinc_derivr(real_t z);
	Ref<MLPPVector> sinc_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sinc_derivm(const Ref<MLPPMatrix> &z);

	//RELU

	real_t relu_normr(real_t z);
	Ref<MLPPVector> relu_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> relu_normm(const Ref<MLPPMatrix> &z);

	real_t relu_derivr(real_t z);
	Ref<MLPPVector> relu_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> relu_derivm(const Ref<MLPPMatrix> &z);

	//LEAKYRELU

	real_t leaky_relu_normr(real_t z, real_t c);
	Ref<MLPPVector> leaky_relu_normv(const Ref<MLPPVector> &z, real_t c);
	Ref<MLPPMatrix> leaky_relu_normm(const Ref<MLPPMatrix> &z, real_t c);

	real_t leaky_relu_derivr(real_t z, real_t c);
	Ref<MLPPVector> leaky_relu_derivv(const Ref<MLPPVector> &z, real_t c);
	Ref<MLPPMatrix> leaky_relu_derivm(const Ref<MLPPMatrix> &z, real_t c);

	//ELU

	real_t elu_normr(real_t z, real_t c);
	Ref<MLPPVector> elu_normv(const Ref<MLPPVector> &z, real_t c);
	Ref<MLPPMatrix> elu_normm(const Ref<MLPPMatrix> &z, real_t c);

	real_t elu_derivr(real_t z, real_t c);
	Ref<MLPPVector> elu_derivv(const Ref<MLPPVector> &z, real_t c);
	Ref<MLPPMatrix> elu_derivm(const Ref<MLPPMatrix> &z, real_t c);

	//SELU

	real_t selu_normr(real_t z, real_t lambda, real_t c);
	Ref<MLPPVector> selu_normv(const Ref<MLPPVector> &z, real_t lambda, real_t c);
	Ref<MLPPMatrix> selu_normm(const Ref<MLPPMatrix> &z, real_t lambda, real_t c);

	real_t selu_derivr(real_t z, real_t lambda, real_t c);
	Ref<MLPPVector> selu_derivv(const Ref<MLPPVector> &z, real_t lambda, real_t c);
	Ref<MLPPMatrix> selu_derivm(const Ref<MLPPMatrix> &z, real_t lambda, real_t c);

	//GELU

	real_t gelu_normr(real_t z);
	Ref<MLPPVector> gelu_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> gelu_normm(const Ref<MLPPMatrix> &z);

	real_t gelu_derivr(real_t z);
	Ref<MLPPVector> gelu_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> gelu_derivm(const Ref<MLPPMatrix> &z);

	//SIGN

	real_t sign_normr(real_t z);
	Ref<MLPPVector> sign_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sign_normm(const Ref<MLPPMatrix> &z);

	real_t sign_derivr(real_t z);
	Ref<MLPPVector> sign_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sign_derivm(const Ref<MLPPMatrix> &z);

	//SINH

	real_t sinh_normr(real_t z);
	Ref<MLPPVector> sinh_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sinh_normm(const Ref<MLPPMatrix> &z);

	real_t sinh_derivr(real_t z);
	Ref<MLPPVector> sinh_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sinh_derivm(const Ref<MLPPMatrix> &z);

	//COSH

	real_t cosh_normr(real_t z);
	Ref<MLPPVector> cosh_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> cosh_normm(const Ref<MLPPMatrix> &z);

	real_t cosh_derivr(real_t z);
	Ref<MLPPVector> cosh_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> cosh_derivm(const Ref<MLPPMatrix> &z);

	//TANH

	real_t tanh_normr(real_t z);
	Ref<MLPPVector> tanh_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> tanh_normm(const Ref<MLPPMatrix> &z);

	real_t tanh_derivr(real_t z);
	Ref<MLPPVector> tanh_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> tanh_derivm(const Ref<MLPPMatrix> &z);

	//CSCH

	real_t csch_normr(real_t z);
	Ref<MLPPVector> csch_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> csch_normm(const Ref<MLPPMatrix> &z);

	real_t csch_derivr(real_t z);
	Ref<MLPPVector> csch_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> csch_derivm(const Ref<MLPPMatrix> &z);

	//SECH

	real_t sech_normr(real_t z);
	Ref<MLPPVector> sech_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sech_normm(const Ref<MLPPMatrix> &z);

	real_t sech_derivr(real_t z);
	Ref<MLPPVector> sech_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sech_derivm(const Ref<MLPPMatrix> &z);

	//COTH

	real_t coth_normr(real_t z);
	Ref<MLPPVector> coth_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> coth_normm(const Ref<MLPPMatrix> &z);

	real_t coth_derivr(real_t z);
	Ref<MLPPVector> coth_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> coth_derivm(const Ref<MLPPMatrix> &z);

	//ARSINH

	real_t arsinh_normr(real_t z);
	Ref<MLPPVector> arsinh_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arsinh_normm(const Ref<MLPPMatrix> &z);

	real_t arsinh_derivr(real_t z);
	Ref<MLPPVector> arsinh_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arsinh_derivm(const Ref<MLPPMatrix> &z);

	//ARCOSH

	real_t arcosh_normr(real_t z);
	Ref<MLPPVector> arcosh_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arcosh_normm(const Ref<MLPPMatrix> &z);

	real_t arcosh_derivr(real_t z);
	Ref<MLPPVector> arcosh_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arcosh_derivm(const Ref<MLPPMatrix> &z);

	//ARTANH

	real_t artanh_normr(real_t z);
	Ref<MLPPVector> artanh_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> artanh_normm(const Ref<MLPPMatrix> &z);

	real_t artanh_derivr(real_t z);
	Ref<MLPPVector> artanh_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> artanh_derivm(const Ref<MLPPMatrix> &z);

	//ARCSCH

	real_t arcsch_normr(real_t z);
	Ref<MLPPVector> arcsch_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arcsch_normm(const Ref<MLPPMatrix> &z);

	real_t arcsch_derivr(real_t z);
	Ref<MLPPVector> arcsch_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arcsch_derivm(const Ref<MLPPMatrix> &z);

	//ARSECH

	real_t arsech_normr(real_t z);
	Ref<MLPPVector> arsech_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arsech_normm(const Ref<MLPPMatrix> &z);

	real_t arsech_derivr(real_t z);
	Ref<MLPPVector> arsech_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arsech_derivm(const Ref<MLPPMatrix> &z);

	//ARCOTH

	real_t arcoth_normr(real_t z);
	Ref<MLPPVector> arcoth_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arcoth_normm(const Ref<MLPPMatrix> &z);

	real_t arcoth_derivr(real_t z);
	Ref<MLPPVector> arcoth_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arcoth_derivm(const Ref<MLPPMatrix> &z);

protected:
	static void _bind_methods();
};

VARIANT_ENUM_CAST(MLPPActivation::ActivationFunction);

#endif /* Activation_hpp */
