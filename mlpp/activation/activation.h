
#ifndef MLPP_ACTIVATION_H
#define MLPP_ACTIVATION_H

//
//  Activation.hpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <vector>

//TODO this should probably be a singleton
//TODO Activation functions should either have a variant which does not allocate, or they should just be reworked altogether

class MLPPActivation : public Reference {
	GDCLASS(MLPPActivation, Reference);

public:
	enum ActivationFunction {
		ACTIVATION_FUNCTION_LINEAR = 0,
		ACTIVATION_FUNCTION_SIGMOID,
		ACTIVATION_FUNCTION_SWISH,
		ACTIVATION_FUNCTION_MISH,
		ACTIVATION_FUNCTION_SIN_C,
		ACTIVATION_FUNCTION_SOFTPLUS,
		ACTIVATION_FUNCTION_SOFTSIGN,
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
	typedef Ref<MLPPMatrix> (MLPPActivation::*ActivationFunctionPointer)(const Ref<MLPPMatrix> &);
	ActivationFunctionPointer get_activation_function_ptr(const ActivationFunction func, const bool deriv = false);

	Ref<MLPPVector> run_activation_vector(const ActivationFunction func, const Ref<MLPPVector> &z, const bool deriv = false);
	Ref<MLPPMatrix> run_activation_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z, const bool deriv = false);

	Ref<MLPPVector> run_activation_norm_vector(const ActivationFunction func, const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> run_activation_norm_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z);

	Ref<MLPPVector> run_activation_deriv_vector(const ActivationFunction func, const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> run_activation_deriv_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z);

	Ref<MLPPVector> activation(const Ref<MLPPVector> &z, real_t (*function)(real_t), const bool deriv = false);
	Ref<MLPPVector> activation_norm(const Ref<MLPPVector> &z, real_t (*function)(real_t));
	Ref<MLPPVector> activation_deriv(const Ref<MLPPVector> &z, real_t (*function)(real_t));

	//ACTIVATION FUNCTIONS

	//LINEAR
	real_t linear_norm(real_t z);
	Ref<MLPPVector> linear_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> linear_norm(const Ref<MLPPMatrix> &z);

	real_t linear_deriv(real_t z);
	Ref<MLPPVector> linear_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> linear_deriv(const Ref<MLPPMatrix> &z);

	//SIGMOID
	real_t sigmoid_norm(real_t z);
	Ref<MLPPVector> sigmoid_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sigmoid_norm(const Ref<MLPPMatrix> &z);

	real_t sigmoid_deriv(real_t z);
	Ref<MLPPVector> sigmoid_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sigmoid_deriv(const Ref<MLPPMatrix> &z);

	//SOFTMAX
	Ref<MLPPVector> softmax_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> softmax_norm(const Ref<MLPPMatrix> &z);

	Ref<MLPPVector> softmax_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> softmax_deriv(const Ref<MLPPMatrix> &z);

	//ADJ_SOFTMAX

	Ref<MLPPVector> adj_softmax_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> adj_softmax_norm(const Ref<MLPPMatrix> &z);

	Ref<MLPPVector> adj_softmax(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> adj_softmax(const Ref<MLPPMatrix> &z);

	//SOFTMAX DERIV

	Ref<MLPPMatrix> softmax_deriv_norm(const Ref<MLPPVector> &z);
	std::vector<Ref<MLPPMatrix>> softmax_deriv_norm(const Ref<MLPPMatrix> &z);

	Ref<MLPPMatrix> softmax_deriv_deriv(const Ref<MLPPVector> &z);
	std::vector<Ref<MLPPMatrix>> softmax_deriv_deriv(const Ref<MLPPMatrix> &z);

	//SOFTPLUS

	real_t softplus_norm(real_t z);
	Ref<MLPPVector> softplus_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> softplus_norm(const Ref<MLPPMatrix> &z);

	real_t softplus_deriv(real_t z);
	Ref<MLPPVector> softplus_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> softplus_deriv(const Ref<MLPPMatrix> &z);

	//SOFTSIGN

	real_t softsign_norm(real_t z);
	Ref<MLPPVector> softsign_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> softsign_norm(const Ref<MLPPMatrix> &z);

	real_t softsign_deriv(real_t z);
	Ref<MLPPVector> softsign_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> softsign_deriv(const Ref<MLPPMatrix> &z);

	//GAUSSIANCDF

	real_t gaussian_cdf_norm(real_t z);
	Ref<MLPPVector> gaussian_cdf_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> gaussian_cdf_norm(const Ref<MLPPMatrix> &z);

	real_t gaussian_cdf_deriv(real_t z);
	Ref<MLPPVector> gaussian_cdf_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> gaussian_cdf_deriv(const Ref<MLPPMatrix> &z);

	//CLOGLOG

	real_t cloglog_norm(real_t z);
	Ref<MLPPVector> cloglog_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> cloglog_norm(const Ref<MLPPMatrix> &z);

	real_t cloglog_deriv(real_t z);
	Ref<MLPPVector> cloglog_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> cloglog_deriv(const Ref<MLPPMatrix> &z);

	//LOGIT

	real_t logit_norm(real_t z);
	Ref<MLPPVector> logit_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> logit_norm(const Ref<MLPPMatrix> &z);

	real_t logit_deriv(real_t z);
	Ref<MLPPVector> logit_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> logit_deriv(const Ref<MLPPMatrix> &z);

	//UNITSTEP

	real_t unit_step_norm(real_t z);
	Ref<MLPPVector> unit_step_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> unit_step_norm(const Ref<MLPPMatrix> &z);

	real_t unit_step_deriv(real_t z);
	Ref<MLPPVector> unit_step_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> unit_step_deriv(const Ref<MLPPMatrix> &z);

	//SWISH

	real_t swish_norm(real_t z);
	Ref<MLPPVector> swish_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> swish_norm(const Ref<MLPPMatrix> &z);

	real_t swish_deriv(real_t z);
	Ref<MLPPVector> swish_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> swish_deriv(const Ref<MLPPMatrix> &z);

	//MISH

	real_t mish_norm(real_t z);
	Ref<MLPPVector> mish_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> mish_norm(const Ref<MLPPMatrix> &z);

	real_t mish_deriv(real_t z);
	Ref<MLPPVector> mish_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> mish_deriv(const Ref<MLPPMatrix> &z);

	//SINC

	real_t sinc_norm(real_t z);
	Ref<MLPPVector> sinc_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sinc_norm(const Ref<MLPPMatrix> &z);

	real_t sinc_deriv(real_t z);
	Ref<MLPPVector> sinc_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sinc_deriv(const Ref<MLPPMatrix> &z);

	//RELU

	real_t relu_norm(real_t z);
	Ref<MLPPVector> relu_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> relu_norm(const Ref<MLPPMatrix> &z);

	real_t relu_deriv(real_t z);
	Ref<MLPPVector> relu_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> relu_deriv(const Ref<MLPPMatrix> &z);

	//LEAKYRELU

	real_t leaky_relu_norm(real_t z, real_t c);
	Ref<MLPPVector> leaky_relu_norm(const Ref<MLPPVector> &z, real_t c);
	Ref<MLPPMatrix> leaky_relu_norm(const Ref<MLPPMatrix> &z, real_t c);

	real_t leaky_relu_deriv(real_t z, real_t c);
	Ref<MLPPVector> leaky_relu_deriv(const Ref<MLPPVector> &z, real_t c);
	Ref<MLPPMatrix> leaky_relu_deriv(const Ref<MLPPMatrix> &z, real_t c);

	//ELU

	real_t elu_norm(real_t z, real_t c);
	Ref<MLPPVector> elu_norm(const Ref<MLPPVector> &z, real_t c);
	Ref<MLPPMatrix> elu_norm(const Ref<MLPPMatrix> &z, real_t c);

	real_t elu_deriv(real_t z, real_t c);
	Ref<MLPPVector> elu_deriv(const Ref<MLPPVector> &z, real_t c);
	Ref<MLPPMatrix> elu_deriv(const Ref<MLPPMatrix> &z, real_t c);

	//SELU

	real_t selu_norm(real_t z, real_t lambda, real_t c);
	Ref<MLPPVector> selu_norm(const Ref<MLPPVector> &z, real_t lambda, real_t c);
	Ref<MLPPMatrix> selu_norm(Ref<MLPPMatrix>, real_t lambda, real_t c);

	real_t selu_deriv(real_t z, real_t lambda, real_t c);
	Ref<MLPPVector> selu_deriv(const Ref<MLPPVector> &z, real_t lambda, real_t c);
	Ref<MLPPMatrix> selu_deriv(Ref<MLPPMatrix>, real_t lambda, real_t c);

	//GELU

	real_t gelu_norm(real_t z);
	Ref<MLPPVector> gelu_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> gelu_norm(const Ref<MLPPMatrix> &z);

	real_t gelu_deriv(real_t z);
	Ref<MLPPVector> gelu_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> gelu_deriv(const Ref<MLPPMatrix> &z);

	//SIGN

	real_t sign_norm(real_t z);
	Ref<MLPPVector> sign_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sign_norm(const Ref<MLPPMatrix> &z);

	real_t sign_deriv(real_t z);
	Ref<MLPPVector> sign_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sign_deriv(const Ref<MLPPMatrix> &z);

	//SINH

	real_t sinh_norm(real_t z);
	Ref<MLPPVector> sinh_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sinh_norm(const Ref<MLPPMatrix> &z);

	real_t sinh_deriv(real_t z);
	Ref<MLPPVector> sinh_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sinh_deriv(const Ref<MLPPMatrix> &z);

	//COSH

	real_t cosh_norm(real_t z);
	Ref<MLPPVector> cosh_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> cosh_norm(const Ref<MLPPMatrix> &z);

	real_t cosh_deriv(real_t z);
	Ref<MLPPVector> cosh_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> cosh_deriv(const Ref<MLPPMatrix> &z);

	//TANH

	real_t tanh_norm(real_t z);
	Ref<MLPPVector> tanh_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> tanh_norm(const Ref<MLPPMatrix> &z);

	real_t tanh_deriv(real_t z);
	Ref<MLPPVector> tanh_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> tanh_deriv(const Ref<MLPPMatrix> &z);

	//CSCH

	real_t csch_norm(real_t z);
	Ref<MLPPVector> csch_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> csch_norm(const Ref<MLPPMatrix> &z);

	real_t csch_deriv(real_t z);
	Ref<MLPPVector> csch_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> csch_deriv(const Ref<MLPPMatrix> &z);

	//SECH

	real_t sech_norm(real_t z);
	Ref<MLPPVector> sech_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sech_norm(const Ref<MLPPMatrix> &z);

	real_t sech_deriv(real_t z);
	Ref<MLPPVector> sech_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> sech_deriv(const Ref<MLPPMatrix> &z);

	//COTH

	real_t coth_norm(real_t z);
	Ref<MLPPVector> coth_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> coth_norm(const Ref<MLPPMatrix> &z);

	real_t coth_deriv(real_t z);
	Ref<MLPPVector> coth_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> coth_deriv(const Ref<MLPPMatrix> &z);

	//ARSINH

	real_t arsinh_norm(real_t z);
	Ref<MLPPVector> arsinh_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arsinh_norm(const Ref<MLPPMatrix> &z);

	real_t arsinh_deriv(real_t z);
	Ref<MLPPVector> arsinh_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arsinh_deriv(const Ref<MLPPMatrix> &z);

	//ARCOSH

	real_t arcosh_norm(real_t z);
	Ref<MLPPVector> arcosh_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arcosh_norm(const Ref<MLPPMatrix> &z);

	real_t arcosh_deriv(real_t z);
	Ref<MLPPVector> arcosh_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arcosh_deriv(const Ref<MLPPMatrix> &z);

	//ARTANH

	real_t artanh_norm(real_t z);
	Ref<MLPPVector> artanh_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> artanh_norm(const Ref<MLPPMatrix> &z);

	real_t artanh_deriv(real_t z);
	Ref<MLPPVector> artanh_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> artanh_deriv(const Ref<MLPPMatrix> &z);

	//ARCSCH

	real_t arcsch_norm(real_t z);
	Ref<MLPPVector> arcsch_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arcsch_norm(const Ref<MLPPMatrix> &z);

	real_t arcsch_deriv(real_t z);
	Ref<MLPPVector> arcsch_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arcsch_deriv(const Ref<MLPPMatrix> &z);

	//ARSECH

	real_t arsech_norm(real_t z);
	Ref<MLPPVector> arsech_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arsech_norm(const Ref<MLPPMatrix> &z);

	real_t arsech_deriv(real_t z);
	Ref<MLPPVector> arsech_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arsech_deriv(const Ref<MLPPMatrix> &z);

	//ARCOTH

	real_t arcoth_norm(real_t z);
	Ref<MLPPVector> arcoth_norm(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arcoth_norm(const Ref<MLPPMatrix> &z);

	real_t arcoth_deriv(real_t z);
	Ref<MLPPVector> arcoth_deriv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> arcoth_deriv(const Ref<MLPPMatrix> &z);

	// =========    OLD    ===========

	real_t linear(real_t z, bool deriv = false);
	std::vector<real_t> linear(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> linear(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t sigmoid(real_t z, bool deriv = false);
	std::vector<real_t> sigmoid(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> sigmoid(std::vector<std::vector<real_t>> z, bool deriv = false);

	std::vector<real_t> softmax(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> softmax(std::vector<std::vector<real_t>> z, bool deriv = false);

	std::vector<real_t> adjSoftmax(std::vector<real_t> z);
	std::vector<std::vector<real_t>> adjSoftmax(std::vector<std::vector<real_t>> z);

	std::vector<std::vector<real_t>> softmaxDeriv(std::vector<real_t> z);
	std::vector<std::vector<std::vector<real_t>>> softmaxDeriv(std::vector<std::vector<real_t>> z);

	real_t softplus(real_t z, bool deriv = false);
	std::vector<real_t> softplus(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> softplus(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t softsign(real_t z, bool deriv = false);
	std::vector<real_t> softsign(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> softsign(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t gaussianCDF(real_t z, bool deriv = false);
	std::vector<real_t> gaussianCDF(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> gaussianCDF(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t cloglog(real_t z, bool deriv = false);
	std::vector<real_t> cloglog(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> cloglog(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t logit(real_t z, bool deriv = false);
	std::vector<real_t> logit(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> logit(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t unitStep(real_t z, bool deriv = false);
	std::vector<real_t> unitStep(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> unitStep(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t swish(real_t z, bool deriv = false);
	std::vector<real_t> swish(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> swish(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t mish(real_t z, bool deriv = false);
	std::vector<real_t> mish(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> mish(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t sinc(real_t z, bool deriv = false);
	std::vector<real_t> sinc(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> sinc(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t RELU(real_t z, bool deriv = false);
	std::vector<real_t> RELU(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> RELU(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t leakyReLU(real_t z, real_t c, bool deriv = false);
	std::vector<real_t> leakyReLU(std::vector<real_t> z, real_t c, bool deriv = false);
	std::vector<std::vector<real_t>> leakyReLU(std::vector<std::vector<real_t>> z, real_t c, bool deriv = false);

	real_t ELU(real_t z, real_t c, bool deriv = false);
	std::vector<real_t> ELU(std::vector<real_t> z, real_t c, bool deriv = false);
	std::vector<std::vector<real_t>> ELU(std::vector<std::vector<real_t>> z, real_t c, bool deriv = false);

	real_t SELU(real_t z, real_t lambda, real_t c, bool deriv = false);
	std::vector<real_t> SELU(std::vector<real_t> z, real_t lambda, real_t c, bool deriv = false);
	std::vector<std::vector<real_t>> SELU(std::vector<std::vector<real_t>>, real_t lambda, real_t c, bool deriv = false);

	real_t GELU(real_t z, bool deriv = false);
	std::vector<real_t> GELU(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> GELU(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t sign(real_t z, bool deriv = false);
	std::vector<real_t> sign(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> sign(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t sinh(real_t z, bool deriv = false);
	std::vector<real_t> sinh(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> sinh(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t cosh(real_t z, bool deriv = false);
	std::vector<real_t> cosh(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> cosh(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t tanh(real_t z, bool deriv = false);
	std::vector<real_t> tanh(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> tanh(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t csch(real_t z, bool deriv = false);
	std::vector<real_t> csch(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> csch(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t sech(real_t z, bool deriv = false);
	std::vector<real_t> sech(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> sech(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t coth(real_t z, bool deriv = false);
	std::vector<real_t> coth(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> coth(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t arsinh(real_t z, bool deriv = false);
	std::vector<real_t> arsinh(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> arsinh(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t arcosh(real_t z, bool deriv = false);
	std::vector<real_t> arcosh(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> arcosh(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t artanh(real_t z, bool deriv = false);
	std::vector<real_t> artanh(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> artanh(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t arcsch(real_t z, bool deriv = false);
	std::vector<real_t> arcsch(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> arcsch(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t arsech(real_t z, bool deriv = false);
	std::vector<real_t> arsech(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> arsech(std::vector<std::vector<real_t>> z, bool deriv = false);

	real_t arcoth(real_t z, bool deriv = false);
	std::vector<real_t> arcoth(std::vector<real_t> z, bool deriv = false);
	std::vector<std::vector<real_t>> arcoth(std::vector<std::vector<real_t>> z, bool deriv = false);

	std::vector<real_t> activation(std::vector<real_t> z, bool deriv, real_t (*function)(real_t, bool));

private:
};

VARIANT_ENUM_CAST(MLPPActivation::ActivationFunction);

#endif /* Activation_hpp */
