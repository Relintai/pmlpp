//
//  Activation.cpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "activation.h"
#include "../lin_alg/lin_alg.h"

#include "core/math/math_defs.h"

#include <algorithm>
#include <cmath>
#include <iostream>

MLPPActivation::RealActivationFunctionPointer MLPPActivation::get_activation_function_ptr_real(const ActivationFunction func, const bool deriv) {
	if (deriv) {
		return get_activation_function_ptr_normal_real(func);
	} else {
		return get_activation_function_ptr_deriv_real(func);
	}
}
MLPPActivation::VectorActivationFunctionPointer MLPPActivation::get_activation_function_ptr_vector(const ActivationFunction func, const bool deriv) {
	if (deriv) {
		return get_activation_function_ptr_normal_vector(func);
	} else {
		return get_activation_function_ptr_deriv_vector(func);
	}
}
MLPPActivation::MatrixActivationFunctionPointer MLPPActivation::get_activation_function_ptr_matrix(const ActivationFunction func, const bool deriv) {
	if (deriv) {
		return get_activation_function_ptr_normal_matrix(func);
	} else {
		return get_activation_function_ptr_deriv_matrix(func);
	}
}

MLPPActivation::RealActivationFunctionPointer MLPPActivation::get_activation_function_ptr_normal_real(const ActivationFunction func) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return &MLPPActivation::linear_normr;
		case ACTIVATION_FUNCTION_SIGMOID:
			return &MLPPActivation::sigmoid_normr;
		case ACTIVATION_FUNCTION_SWISH:
			return &MLPPActivation::swish_normr;
		case ACTIVATION_FUNCTION_MISH:
			return &MLPPActivation::mish_normr;
		case ACTIVATION_FUNCTION_SIN_C:
			return &MLPPActivation::sinc_normr;
		case ACTIVATION_FUNCTION_SOFTMAX:
			return &MLPPActivation::softmax_normr;
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return &MLPPActivation::softplus_normr;
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return &MLPPActivation::softsign_normr;
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return &MLPPActivation::adj_softmax_normr;
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return &MLPPActivation::cloglog_normr;
		case ACTIVATION_FUNCTION_LOGIT:
			return &MLPPActivation::logit_normr;
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return &MLPPActivation::gaussian_cdf_normr;
		case ACTIVATION_FUNCTION_RELU:
			return &MLPPActivation::relu_normr;
		case ACTIVATION_FUNCTION_GELU:
			return &MLPPActivation::gelu_normr;
		case ACTIVATION_FUNCTION_SIGN:
			return &MLPPActivation::sign_normr;
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return &MLPPActivation::unit_step_normr;
		case ACTIVATION_FUNCTION_SINH:
			return &MLPPActivation::sinh_normr;
		case ACTIVATION_FUNCTION_COSH:
			return &MLPPActivation::cosh_normr;
		case ACTIVATION_FUNCTION_TANH:
			return &MLPPActivation::tanh_normr;
		case ACTIVATION_FUNCTION_CSCH:
			return &MLPPActivation::csch_normr;
		case ACTIVATION_FUNCTION_SECH:
			return &MLPPActivation::sech_normr;
		case ACTIVATION_FUNCTION_COTH:
			return &MLPPActivation::coth_normr;
		case ACTIVATION_FUNCTION_ARSINH:
			return &MLPPActivation::arsinh_normr;
		case ACTIVATION_FUNCTION_ARCOSH:
			return &MLPPActivation::arcosh_normr;
		case ACTIVATION_FUNCTION_ARTANH:
			return &MLPPActivation::artanh_normr;
		case ACTIVATION_FUNCTION_ARCSCH:
			return &MLPPActivation::arcsch_normr;
		case ACTIVATION_FUNCTION_ARSECH:
			return &MLPPActivation::arsech_normr;
		case ACTIVATION_FUNCTION_ARCOTH:
			return &MLPPActivation::arcoth_normr;
		default:
			return NULL;
	}
}
MLPPActivation::VectorActivationFunctionPointer MLPPActivation::get_activation_function_ptr_normal_vector(const ActivationFunction func) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return &MLPPActivation::linear_normv;
		case ACTIVATION_FUNCTION_SIGMOID:
			return &MLPPActivation::sigmoid_normv;
		case ACTIVATION_FUNCTION_SWISH:
			return &MLPPActivation::swish_normv;
		case ACTIVATION_FUNCTION_MISH:
			return &MLPPActivation::mish_normv;
		case ACTIVATION_FUNCTION_SIN_C:
			return &MLPPActivation::sinc_normv;
		case ACTIVATION_FUNCTION_SOFTMAX:
			return &MLPPActivation::softmax_normv;
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return &MLPPActivation::softplus_normv;
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return &MLPPActivation::softsign_normv;
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return &MLPPActivation::adj_softmax_normv;
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return &MLPPActivation::cloglog_normv;
		case ACTIVATION_FUNCTION_LOGIT:
			return &MLPPActivation::logit_normv;
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return &MLPPActivation::gaussian_cdf_normv;
		case ACTIVATION_FUNCTION_RELU:
			return &MLPPActivation::relu_normv;
		case ACTIVATION_FUNCTION_GELU:
			return &MLPPActivation::gelu_normv;
		case ACTIVATION_FUNCTION_SIGN:
			return &MLPPActivation::sign_normv;
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return &MLPPActivation::unit_step_normv;
		case ACTIVATION_FUNCTION_SINH:
			return &MLPPActivation::sinh_normv;
		case ACTIVATION_FUNCTION_COSH:
			return &MLPPActivation::cosh_normv;
		case ACTIVATION_FUNCTION_TANH:
			return &MLPPActivation::tanh_normv;
		case ACTIVATION_FUNCTION_CSCH:
			return &MLPPActivation::csch_normv;
		case ACTIVATION_FUNCTION_SECH:
			return &MLPPActivation::sech_normv;
		case ACTIVATION_FUNCTION_COTH:
			return &MLPPActivation::coth_normv;
		case ACTIVATION_FUNCTION_ARSINH:
			return &MLPPActivation::arsinh_normv;
		case ACTIVATION_FUNCTION_ARCOSH:
			return &MLPPActivation::arcosh_normv;
		case ACTIVATION_FUNCTION_ARTANH:
			return &MLPPActivation::artanh_normv;
		case ACTIVATION_FUNCTION_ARCSCH:
			return &MLPPActivation::arcsch_normv;
		case ACTIVATION_FUNCTION_ARSECH:
			return &MLPPActivation::arsech_normv;
		case ACTIVATION_FUNCTION_ARCOTH:
			return &MLPPActivation::arcoth_normv;
		default:
			return NULL;
	}
}
MLPPActivation::MatrixActivationFunctionPointer MLPPActivation::get_activation_function_ptr_normal_matrix(const ActivationFunction func) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return &MLPPActivation::linear_normm;
		case ACTIVATION_FUNCTION_SIGMOID:
			return &MLPPActivation::sigmoid_normm;
		case ACTIVATION_FUNCTION_SWISH:
			return &MLPPActivation::swish_normm;
		case ACTIVATION_FUNCTION_MISH:
			return &MLPPActivation::mish_normm;
		case ACTIVATION_FUNCTION_SIN_C:
			return &MLPPActivation::sinc_normm;
		case ACTIVATION_FUNCTION_SOFTMAX:
			return &MLPPActivation::softmax_normm;
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return &MLPPActivation::softplus_normm;
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return &MLPPActivation::softsign_normm;
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return &MLPPActivation::adj_softmax_normm;
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return &MLPPActivation::cloglog_normm;
		case ACTIVATION_FUNCTION_LOGIT:
			return &MLPPActivation::logit_normm;
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return &MLPPActivation::gaussian_cdf_normm;
		case ACTIVATION_FUNCTION_RELU:
			return &MLPPActivation::relu_normm;
		case ACTIVATION_FUNCTION_GELU:
			return &MLPPActivation::gelu_normm;
		case ACTIVATION_FUNCTION_SIGN:
			return &MLPPActivation::sign_normm;
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return &MLPPActivation::unit_step_normm;
		case ACTIVATION_FUNCTION_SINH:
			return &MLPPActivation::sinh_normm;
		case ACTIVATION_FUNCTION_COSH:
			return &MLPPActivation::cosh_normm;
		case ACTIVATION_FUNCTION_TANH:
			return &MLPPActivation::tanh_normm;
		case ACTIVATION_FUNCTION_CSCH:
			return &MLPPActivation::csch_normm;
		case ACTIVATION_FUNCTION_SECH:
			return &MLPPActivation::sech_normm;
		case ACTIVATION_FUNCTION_COTH:
			return &MLPPActivation::coth_normm;
		case ACTIVATION_FUNCTION_ARSINH:
			return &MLPPActivation::arsinh_normm;
		case ACTIVATION_FUNCTION_ARCOSH:
			return &MLPPActivation::arcosh_normm;
		case ACTIVATION_FUNCTION_ARTANH:
			return &MLPPActivation::artanh_normm;
		case ACTIVATION_FUNCTION_ARCSCH:
			return &MLPPActivation::arcsch_normm;
		case ACTIVATION_FUNCTION_ARSECH:
			return &MLPPActivation::arsech_normm;
		case ACTIVATION_FUNCTION_ARCOTH:
			return &MLPPActivation::arcoth_normm;
		default:
			return NULL;
	}
}

MLPPActivation::RealActivationFunctionPointer MLPPActivation::get_activation_function_ptr_deriv_real(const ActivationFunction func) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return &MLPPActivation::linear_normr;
		case ACTIVATION_FUNCTION_SIGMOID:
			return &MLPPActivation::sigmoid_normr;
		case ACTIVATION_FUNCTION_SWISH:
			return &MLPPActivation::swish_normr;
		case ACTIVATION_FUNCTION_MISH:
			return &MLPPActivation::mish_normr;
		case ACTIVATION_FUNCTION_SIN_C:
			return &MLPPActivation::sinc_normr;
		case ACTIVATION_FUNCTION_SOFTMAX:
			return &MLPPActivation::softmax_normr;
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return &MLPPActivation::softplus_normr;
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return &MLPPActivation::softsign_normr;
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return &MLPPActivation::adj_softmax_normr;
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return &MLPPActivation::cloglog_normr;
		case ACTIVATION_FUNCTION_LOGIT:
			return &MLPPActivation::logit_normr;
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return &MLPPActivation::gaussian_cdf_normr;
		case ACTIVATION_FUNCTION_RELU:
			return &MLPPActivation::relu_normr;
		case ACTIVATION_FUNCTION_GELU:
			return &MLPPActivation::gelu_normr;
		case ACTIVATION_FUNCTION_SIGN:
			return &MLPPActivation::sign_normr;
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return &MLPPActivation::unit_step_normr;
		case ACTIVATION_FUNCTION_SINH:
			return &MLPPActivation::sinh_normr;
		case ACTIVATION_FUNCTION_COSH:
			return &MLPPActivation::cosh_normr;
		case ACTIVATION_FUNCTION_TANH:
			return &MLPPActivation::tanh_normr;
		case ACTIVATION_FUNCTION_CSCH:
			return &MLPPActivation::csch_normr;
		case ACTIVATION_FUNCTION_SECH:
			return &MLPPActivation::sech_normr;
		case ACTIVATION_FUNCTION_COTH:
			return &MLPPActivation::coth_normr;
		case ACTIVATION_FUNCTION_ARSINH:
			return &MLPPActivation::arsinh_normr;
		case ACTIVATION_FUNCTION_ARCOSH:
			return &MLPPActivation::arcosh_normr;
		case ACTIVATION_FUNCTION_ARTANH:
			return &MLPPActivation::artanh_normr;
		case ACTIVATION_FUNCTION_ARCSCH:
			return &MLPPActivation::arcsch_normr;
		case ACTIVATION_FUNCTION_ARSECH:
			return &MLPPActivation::arsech_normr;
		case ACTIVATION_FUNCTION_ARCOTH:
			return &MLPPActivation::arcoth_normr;
		default:
			return NULL;
	}
}
MLPPActivation::VectorActivationFunctionPointer MLPPActivation::get_activation_function_ptr_deriv_vector(const ActivationFunction func) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return &MLPPActivation::linear_derivv;
		case ACTIVATION_FUNCTION_SIGMOID:
			return &MLPPActivation::sigmoid_derivv;
		case ACTIVATION_FUNCTION_SWISH:
			return &MLPPActivation::swish_derivv;
		case ACTIVATION_FUNCTION_MISH:
			return &MLPPActivation::mish_derivv;
		case ACTIVATION_FUNCTION_SIN_C:
			return &MLPPActivation::sinc_derivv;
		case ACTIVATION_FUNCTION_SOFTMAX:
			return &MLPPActivation::softmax_derivv;
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return &MLPPActivation::softplus_derivv;
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return &MLPPActivation::softsign_derivv;
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return &MLPPActivation::adj_softmax_derivv;
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return &MLPPActivation::cloglog_derivv;
		case ACTIVATION_FUNCTION_LOGIT:
			return &MLPPActivation::logit_derivv;
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return &MLPPActivation::gaussian_cdf_derivv;
		case ACTIVATION_FUNCTION_RELU:
			return &MLPPActivation::relu_derivv;
		case ACTIVATION_FUNCTION_GELU:
			return &MLPPActivation::gelu_derivv;
		case ACTIVATION_FUNCTION_SIGN:
			return &MLPPActivation::sign_derivv;
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return &MLPPActivation::unit_step_derivv;
		case ACTIVATION_FUNCTION_SINH:
			return &MLPPActivation::sinh_derivv;
		case ACTIVATION_FUNCTION_COSH:
			return &MLPPActivation::cosh_derivv;
		case ACTIVATION_FUNCTION_TANH:
			return &MLPPActivation::tanh_derivv;
		case ACTIVATION_FUNCTION_CSCH:
			return &MLPPActivation::csch_derivv;
		case ACTIVATION_FUNCTION_SECH:
			return &MLPPActivation::sech_derivv;
		case ACTIVATION_FUNCTION_COTH:
			return &MLPPActivation::coth_derivv;
		case ACTIVATION_FUNCTION_ARSINH:
			return &MLPPActivation::arsinh_derivv;
		case ACTIVATION_FUNCTION_ARCOSH:
			return &MLPPActivation::arcosh_derivv;
		case ACTIVATION_FUNCTION_ARTANH:
			return &MLPPActivation::artanh_derivv;
		case ACTIVATION_FUNCTION_ARCSCH:
			return &MLPPActivation::arcsch_derivv;
		case ACTIVATION_FUNCTION_ARSECH:
			return &MLPPActivation::arsech_derivv;
		case ACTIVATION_FUNCTION_ARCOTH:
			return &MLPPActivation::arcoth_derivv;
		default:
			return NULL;
	}
}
MLPPActivation::MatrixActivationFunctionPointer MLPPActivation::get_activation_function_ptr_deriv_matrix(const ActivationFunction func) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return &MLPPActivation::linear_derivm;
		case ACTIVATION_FUNCTION_SIGMOID:
			return &MLPPActivation::sigmoid_derivm;
		case ACTIVATION_FUNCTION_SWISH:
			return &MLPPActivation::swish_derivm;
		case ACTIVATION_FUNCTION_MISH:
			return &MLPPActivation::mish_derivm;
		case ACTIVATION_FUNCTION_SIN_C:
			return &MLPPActivation::sinc_derivm;
		case ACTIVATION_FUNCTION_SOFTMAX:
			return &MLPPActivation::softmax_derivm;
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return &MLPPActivation::softplus_derivm;
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return &MLPPActivation::softsign_derivm;
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return &MLPPActivation::adj_softmax_derivm;
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return &MLPPActivation::cloglog_derivm;
		case ACTIVATION_FUNCTION_LOGIT:
			return &MLPPActivation::logit_derivm;
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return &MLPPActivation::gaussian_cdf_derivm;
		case ACTIVATION_FUNCTION_RELU:
			return &MLPPActivation::relu_derivm;
		case ACTIVATION_FUNCTION_GELU:
			return &MLPPActivation::gelu_derivm;
		case ACTIVATION_FUNCTION_SIGN:
			return &MLPPActivation::sign_derivm;
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return &MLPPActivation::unit_step_derivm;
		case ACTIVATION_FUNCTION_SINH:
			return &MLPPActivation::sinh_derivm;
		case ACTIVATION_FUNCTION_COSH:
			return &MLPPActivation::cosh_derivm;
		case ACTIVATION_FUNCTION_TANH:
			return &MLPPActivation::tanh_derivm;
		case ACTIVATION_FUNCTION_CSCH:
			return &MLPPActivation::csch_derivm;
		case ACTIVATION_FUNCTION_SECH:
			return &MLPPActivation::sech_derivm;
		case ACTIVATION_FUNCTION_COTH:
			return &MLPPActivation::coth_derivm;
		case ACTIVATION_FUNCTION_ARSINH:
			return &MLPPActivation::arsinh_derivm;
		case ACTIVATION_FUNCTION_ARCOSH:
			return &MLPPActivation::arcosh_derivm;
		case ACTIVATION_FUNCTION_ARTANH:
			return &MLPPActivation::artanh_derivm;
		case ACTIVATION_FUNCTION_ARCSCH:
			return &MLPPActivation::arcsch_derivm;
		case ACTIVATION_FUNCTION_ARSECH:
			return &MLPPActivation::arsech_derivm;
		case ACTIVATION_FUNCTION_ARCOTH:
			return &MLPPActivation::arcoth_derivm;
		default:
			return NULL;
	}
}

real_t MLPPActivation::run_activation_real(const ActivationFunction func, const real_t z, const bool deriv) {
	if (deriv) {
		return run_activation_norm_real(func, z);
	} else {
		return run_activation_deriv_real(func, z);
	}
}
Ref<MLPPVector> MLPPActivation::run_activation_vector(const ActivationFunction func, const Ref<MLPPVector> &z, const bool deriv) {
	if (deriv) {
		return run_activation_norm_vector(func, z);
	} else {
		return run_activation_deriv_vector(func, z);
	}
}
Ref<MLPPMatrix> MLPPActivation::run_activation_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z, const bool deriv) {
	if (deriv) {
		return run_activation_norm_matrix(func, z);
	} else {
		return run_activation_deriv_matrix(func, z);
	}
}

real_t MLPPActivation::run_activation_norm_real(const ActivationFunction func, const real_t z) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return linear_normr(z);
		case ACTIVATION_FUNCTION_SIGMOID:
			return sigmoid_normr(z);
		case ACTIVATION_FUNCTION_SWISH:
			return swish_normr(z);
		case ACTIVATION_FUNCTION_MISH:
			return mish_normr(z);
		case ACTIVATION_FUNCTION_SIN_C:
			return sinc_normr(z);
		case ACTIVATION_FUNCTION_SOFTMAX:
			return softmax_normr(z);
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return softplus_normr(z);
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return softsign_normr(z);
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return adj_softmax_normr(z);
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return cloglog_normr(z);
		case ACTIVATION_FUNCTION_LOGIT:
			return logit_normr(z);
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return gaussian_cdf_normr(z);
		case ACTIVATION_FUNCTION_RELU:
			return relu_normr(z);
		case ACTIVATION_FUNCTION_GELU:
			return gelu_normr(z);
		case ACTIVATION_FUNCTION_SIGN:
			return sign_normr(z);
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return unit_step_normr(z);
		case ACTIVATION_FUNCTION_SINH:
			return sinh_normr(z);
		case ACTIVATION_FUNCTION_COSH:
			return cosh_normr(z);
		case ACTIVATION_FUNCTION_TANH:
			return tanh_normr(z);
		case ACTIVATION_FUNCTION_CSCH:
			return csch_normr(z);
		case ACTIVATION_FUNCTION_SECH:
			return sech_normr(z);
		case ACTIVATION_FUNCTION_COTH:
			return coth_normr(z);
		case ACTIVATION_FUNCTION_ARSINH:
			return arsinh_normr(z);
		case ACTIVATION_FUNCTION_ARCOSH:
			return arcosh_normr(z);
		case ACTIVATION_FUNCTION_ARTANH:
			return artanh_normr(z);
		case ACTIVATION_FUNCTION_ARCSCH:
			return arcsch_normr(z);
		case ACTIVATION_FUNCTION_ARSECH:
			return arsech_normr(z);
		case ACTIVATION_FUNCTION_ARCOTH:
			return arcoth_normr(z);
		default:
			ERR_FAIL_V(0);
	}
}
Ref<MLPPVector> MLPPActivation::run_activation_norm_vector(const ActivationFunction func, const Ref<MLPPVector> &z) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return linear_normv(z);
		case ACTIVATION_FUNCTION_SIGMOID:
			return sigmoid_normv(z);
		case ACTIVATION_FUNCTION_SWISH:
			return swish_normv(z);
		case ACTIVATION_FUNCTION_MISH:
			return mish_normv(z);
		case ACTIVATION_FUNCTION_SIN_C:
			return sinc_normv(z);
		case ACTIVATION_FUNCTION_SOFTMAX:
			return softmax_normv(z);
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return softplus_normv(z);
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return softsign_normv(z);
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return adj_softmax_normv(z);
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return cloglog_normv(z);
		case ACTIVATION_FUNCTION_LOGIT:
			return logit_normv(z);
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return gaussian_cdf_normv(z);
		case ACTIVATION_FUNCTION_RELU:
			return relu_normv(z);
		case ACTIVATION_FUNCTION_GELU:
			return gelu_normv(z);
		case ACTIVATION_FUNCTION_SIGN:
			return sign_normv(z);
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return unit_step_normv(z);
		case ACTIVATION_FUNCTION_SINH:
			return sinh_normv(z);
		case ACTIVATION_FUNCTION_COSH:
			return cosh_normv(z);
		case ACTIVATION_FUNCTION_TANH:
			return tanh_normv(z);
		case ACTIVATION_FUNCTION_CSCH:
			return csch_normv(z);
		case ACTIVATION_FUNCTION_SECH:
			return sech_normv(z);
		case ACTIVATION_FUNCTION_COTH:
			return coth_normv(z);
		case ACTIVATION_FUNCTION_ARSINH:
			return arsinh_normv(z);
		case ACTIVATION_FUNCTION_ARCOSH:
			return arcosh_normv(z);
		case ACTIVATION_FUNCTION_ARTANH:
			return artanh_normv(z);
		case ACTIVATION_FUNCTION_ARCSCH:
			return arcsch_normv(z);
		case ACTIVATION_FUNCTION_ARSECH:
			return arsech_normv(z);
		case ACTIVATION_FUNCTION_ARCOTH:
			return arcoth_normv(z);
		default:
			ERR_FAIL_V(Ref<MLPPVector>());
	}
}
Ref<MLPPMatrix> MLPPActivation::run_activation_norm_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return linear_normm(z);
		case ACTIVATION_FUNCTION_SIGMOID:
			return sigmoid_normm(z);
		case ACTIVATION_FUNCTION_SWISH:
			return swish_normm(z);
		case ACTIVATION_FUNCTION_MISH:
			return mish_normm(z);
		case ACTIVATION_FUNCTION_SIN_C:
			return sinc_normm(z);
		case ACTIVATION_FUNCTION_SOFTMAX:
			return softmax_normm(z);
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return softplus_normm(z);
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return softsign_normm(z);
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return adj_softmax_normm(z);
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return cloglog_normm(z);
		case ACTIVATION_FUNCTION_LOGIT:
			return logit_normm(z);
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return gaussian_cdf_normm(z);
		case ACTIVATION_FUNCTION_RELU:
			return relu_normm(z);
		case ACTIVATION_FUNCTION_GELU:
			return gelu_normm(z);
		case ACTIVATION_FUNCTION_SIGN:
			return sign_normm(z);
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return unit_step_normm(z);
		case ACTIVATION_FUNCTION_SINH:
			return sinh_normm(z);
		case ACTIVATION_FUNCTION_COSH:
			return cosh_normm(z);
		case ACTIVATION_FUNCTION_TANH:
			return tanh_normm(z);
		case ACTIVATION_FUNCTION_CSCH:
			return csch_normm(z);
		case ACTIVATION_FUNCTION_SECH:
			return sech_normm(z);
		case ACTIVATION_FUNCTION_COTH:
			return coth_normm(z);
		case ACTIVATION_FUNCTION_ARSINH:
			return arsinh_normm(z);
		case ACTIVATION_FUNCTION_ARCOSH:
			return arcosh_normm(z);
		case ACTIVATION_FUNCTION_ARTANH:
			return artanh_normm(z);
		case ACTIVATION_FUNCTION_ARCSCH:
			return arcsch_normm(z);
		case ACTIVATION_FUNCTION_ARSECH:
			return arsech_normm(z);
		case ACTIVATION_FUNCTION_ARCOTH:
			return arcoth_normm(z);
		default:
			ERR_FAIL_V(Ref<MLPPMatrix>());
	}
}

real_t MLPPActivation::run_activation_deriv_real(const ActivationFunction func, const real_t z) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return linear_normr(z);
		case ACTIVATION_FUNCTION_SIGMOID:
			return sigmoid_normr(z);
		case ACTIVATION_FUNCTION_SWISH:
			return swish_normr(z);
		case ACTIVATION_FUNCTION_MISH:
			return mish_normr(z);
		case ACTIVATION_FUNCTION_SIN_C:
			return sinc_normr(z);
		case ACTIVATION_FUNCTION_SOFTMAX:
			return softmax_normr(z);
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return softplus_normr(z);
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return softsign_normr(z);
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return adj_softmax_normr(z);
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return cloglog_normr(z);
		case ACTIVATION_FUNCTION_LOGIT:
			return logit_normr(z);
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return gaussian_cdf_normr(z);
		case ACTIVATION_FUNCTION_RELU:
			return relu_normr(z);
		case ACTIVATION_FUNCTION_GELU:
			return gelu_normr(z);
		case ACTIVATION_FUNCTION_SIGN:
			return sign_normr(z);
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return unit_step_normr(z);
		case ACTIVATION_FUNCTION_SINH:
			return sinh_normr(z);
		case ACTIVATION_FUNCTION_COSH:
			return cosh_normr(z);
		case ACTIVATION_FUNCTION_TANH:
			return tanh_normr(z);
		case ACTIVATION_FUNCTION_CSCH:
			return csch_normr(z);
		case ACTIVATION_FUNCTION_SECH:
			return sech_normr(z);
		case ACTIVATION_FUNCTION_COTH:
			return coth_normr(z);
		case ACTIVATION_FUNCTION_ARSINH:
			return arsinh_normr(z);
		case ACTIVATION_FUNCTION_ARCOSH:
			return arcosh_normr(z);
		case ACTIVATION_FUNCTION_ARTANH:
			return artanh_normr(z);
		case ACTIVATION_FUNCTION_ARCSCH:
			return arcsch_normr(z);
		case ACTIVATION_FUNCTION_ARSECH:
			return arsech_normr(z);
		case ACTIVATION_FUNCTION_ARCOTH:
			return arcoth_normr(z);
		default:
			ERR_FAIL_V(0);
	}
}
Ref<MLPPVector> MLPPActivation::run_activation_deriv_vector(const ActivationFunction func, const Ref<MLPPVector> &z) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return linear_derivv(z);
		case ACTIVATION_FUNCTION_SIGMOID:
			return sigmoid_derivv(z);
		case ACTIVATION_FUNCTION_SWISH:
			return swish_derivv(z);
		case ACTIVATION_FUNCTION_MISH:
			return mish_derivv(z);
		case ACTIVATION_FUNCTION_SIN_C:
			return sinc_derivv(z);
		case ACTIVATION_FUNCTION_SOFTMAX:
			return softmax_derivv(z);
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return softplus_derivv(z);
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return softsign_derivv(z);
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return adj_softmax_derivv(z);
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return cloglog_derivv(z);
		case ACTIVATION_FUNCTION_LOGIT:
			return logit_derivv(z);
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return gaussian_cdf_derivv(z);
		case ACTIVATION_FUNCTION_RELU:
			return relu_derivv(z);
		case ACTIVATION_FUNCTION_GELU:
			return gelu_derivv(z);
		case ACTIVATION_FUNCTION_SIGN:
			return sign_derivv(z);
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return unit_step_derivv(z);
		case ACTIVATION_FUNCTION_SINH:
			return sinh_derivv(z);
		case ACTIVATION_FUNCTION_COSH:
			return cosh_derivv(z);
		case ACTIVATION_FUNCTION_TANH:
			return tanh_derivv(z);
		case ACTIVATION_FUNCTION_CSCH:
			return csch_derivv(z);
		case ACTIVATION_FUNCTION_SECH:
			return sech_derivv(z);
		case ACTIVATION_FUNCTION_COTH:
			return coth_derivv(z);
		case ACTIVATION_FUNCTION_ARSINH:
			return arsinh_derivv(z);
		case ACTIVATION_FUNCTION_ARCOSH:
			return arcosh_derivv(z);
		case ACTIVATION_FUNCTION_ARTANH:
			return artanh_derivv(z);
		case ACTIVATION_FUNCTION_ARCSCH:
			return arcsch_derivv(z);
		case ACTIVATION_FUNCTION_ARSECH:
			return arsech_derivv(z);
		case ACTIVATION_FUNCTION_ARCOTH:
			return arcoth_derivv(z);
		default:
			ERR_FAIL_V(Ref<MLPPVector>());
	}
}
Ref<MLPPMatrix> MLPPActivation::run_activation_deriv_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return linear_derivm(z);
		case ACTIVATION_FUNCTION_SIGMOID:
			return sigmoid_derivm(z);
		case ACTIVATION_FUNCTION_SWISH:
			return swish_derivm(z);
		case ACTIVATION_FUNCTION_MISH:
			return mish_derivm(z);
		case ACTIVATION_FUNCTION_SIN_C:
			return sinc_derivm(z);
		case ACTIVATION_FUNCTION_SOFTMAX:
			return softmax_derivm(z);
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return softplus_derivm(z);
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return softsign_derivm(z);
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return adj_softmax_derivm(z);
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return cloglog_derivm(z);
		case ACTIVATION_FUNCTION_LOGIT:
			return logit_derivm(z);
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return gaussian_cdf_derivm(z);
		case ACTIVATION_FUNCTION_RELU:
			return relu_derivm(z);
		case ACTIVATION_FUNCTION_GELU:
			return gelu_derivm(z);
		case ACTIVATION_FUNCTION_SIGN:
			return sign_derivm(z);
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return unit_step_derivm(z);
		case ACTIVATION_FUNCTION_SINH:
			return sinh_derivm(z);
		case ACTIVATION_FUNCTION_COSH:
			return cosh_derivm(z);
		case ACTIVATION_FUNCTION_TANH:
			return tanh_derivm(z);
		case ACTIVATION_FUNCTION_CSCH:
			return csch_derivm(z);
		case ACTIVATION_FUNCTION_SECH:
			return sech_derivm(z);
		case ACTIVATION_FUNCTION_COTH:
			return coth_derivm(z);
		case ACTIVATION_FUNCTION_ARSINH:
			return arsinh_derivm(z);
		case ACTIVATION_FUNCTION_ARCOSH:
			return arcosh_derivm(z);
		case ACTIVATION_FUNCTION_ARTANH:
			return artanh_derivm(z);
		case ACTIVATION_FUNCTION_ARCSCH:
			return arcsch_derivm(z);
		case ACTIVATION_FUNCTION_ARSECH:
			return arsech_derivm(z);
		case ACTIVATION_FUNCTION_ARCOTH:
			return arcoth_derivm(z);
		default:
			ERR_FAIL_V(Ref<MLPPMatrix>());
	}
}

Ref<MLPPVector> MLPPActivation::activationr(const Ref<MLPPVector> &z, real_t (*function)(real_t)) {
	Ref<MLPPVector> a;
	a.instance();

	int size = z->size();

	a->resize(size);

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < size; ++i) {
		a_ptr[i] = function(z_ptr[i]);
	}

	return a;
}

//ACTIVATION FUNCTIONS

//LINEAR
real_t MLPPActivation::linear_normr(real_t z) {
	return z;
}
Ref<MLPPVector> MLPPActivation::linear_normv(const Ref<MLPPVector> &z) {
	return z->duplicate_fast();
}
Ref<MLPPMatrix> MLPPActivation::linear_normm(const Ref<MLPPMatrix> &z) {
	return z->duplicate_fast();
}

real_t MLPPActivation::linear_derivr(real_t z) {
	return 1;
}
Ref<MLPPVector> MLPPActivation::linear_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;
	return alg.onevecnv(z->size());
}
Ref<MLPPMatrix> MLPPActivation::linear_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;
	return alg.onematnm(z->size().x, z->size().y);
}

//SIGMOID
real_t MLPPActivation::sigmoid_normr(real_t z) {
	return 1 / (1 + exp(-z));
}
Ref<MLPPVector> MLPPActivation::sigmoid_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;
	return alg.element_wise_divisionnv(alg.onevecnv(z->size()), alg.additionnv(alg.onevecnv(z->size()), alg.expnv(alg.scalar_multiplynv(-1, z))));
}
Ref<MLPPMatrix> MLPPActivation::sigmoid_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;
	return alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y), alg.additionnm(alg.onematnm(z->size().x, z->size().y), alg.expnm(alg.scalar_multiplynm(-1, z))));
}

real_t MLPPActivation::sigmoid_derivr(real_t z) {
	real_t sig_norm = sigmoid_normr(z);

	return sig_norm * (1 - sig_norm);
}

Ref<MLPPVector> MLPPActivation::sigmoid_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	Ref<MLPPVector> sig_norm = sigmoid_normv(z);

	return alg.subtractionnv(sig_norm, alg.hadamard_productnv(sig_norm, sig_norm));
}
Ref<MLPPMatrix> MLPPActivation::sigmoid_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	Ref<MLPPMatrix> sig_norm = sigmoid_normm(z);

	return alg.subtractionnm(sig_norm, alg.hadamard_productnm(sig_norm, sig_norm));
}

//SOFTMAX

real_t MLPPActivation::softmax_normr(real_t z) {
	return z;
}
Ref<MLPPVector> MLPPActivation::softmax_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	int z_size = z->size();

	Ref<MLPPVector> a;
	a.instance();
	a->resize(z_size);

	Ref<MLPPVector> exp_z = alg.expnv(z);
	real_t sum = 0;

	const real_t *exp_z_ptr = exp_z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		sum += exp_z_ptr[i];
	}

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = exp_z_ptr[i] / sum;
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::softmax_normm(const Ref<MLPPMatrix> &z) {
	Size2i z_size = z->size();

	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z_size);

	Ref<MLPPVector> row_tmp;
	row_tmp.instance();
	row_tmp->resize(z_size.x);

	for (int i = 0; i < z_size.y; ++i) {
		z->get_row_into_mlpp_vector(i, row_tmp);

		Ref<MLPPVector> sfn = softmax_normv(row_tmp);

		a->set_row_mlpp_vector(i, sfn);
	}

	return a;
}

real_t MLPPActivation::softmax_derivr(real_t z) {
	return z;
}
Ref<MLPPVector> MLPPActivation::softmax_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	int z_size = z->size();

	Ref<MLPPVector> a;
	a.instance();
	a->resize(z_size);

	Ref<MLPPVector> exp_z = alg.expnv(z);
	real_t sum = 0;

	const real_t *exp_z_ptr = exp_z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		sum += exp_z_ptr[i];
	}

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = exp_z_ptr[i] / sum;
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::softmax_derivm(const Ref<MLPPMatrix> &z) {
	Size2i z_size = z->size();

	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z_size);

	Ref<MLPPVector> row_tmp;
	row_tmp.instance();
	row_tmp->resize(z_size.x);

	for (int i = 0; i < z_size.y; ++i) {
		z->get_row_into_mlpp_vector(i, row_tmp);

		Ref<MLPPVector> sfn = softmax_derivm(z);

		a->set_row_mlpp_vector(i, sfn);
	}

	return a;
}

//ADJ_SOFTMAX

real_t MLPPActivation::adj_softmax_normr(real_t z) {
	return 0;
}

Ref<MLPPVector> MLPPActivation::adj_softmax_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	int size = z->size();
	const real_t *z_ptr = z->ptr();
	real_t c = -Math_INF;

	for (int i = 0; i < size; ++i) {
		int zpi = z_ptr[i];

		if (c < zpi) {
			c = zpi;
		}
	}

	c = -c;

	Ref<MLPPVector> n = alg.scalar_addnv(c, z);

	return softmax_normv(n);
}
Ref<MLPPMatrix> MLPPActivation::adj_softmax_normm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> n = z->duplicate_fast();

	Size2i size = z->size();

	Ref<MLPPVector> row_rmp;
	row_rmp.instance();
	row_rmp->resize(size.x);

	for (int i = 0; i < size.y; ++i) {
		z->get_row_into_mlpp_vector(i, row_rmp);

		Ref<MLPPVector> nv = adj_softmax_normv(row_rmp);

		n->set_row_mlpp_vector(i, nv);
	}

	return n;
}

real_t MLPPActivation::adj_softmax_derivr(real_t z) {
	return 0;
}

Ref<MLPPVector> MLPPActivation::adj_softmax_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	int size = z->size();
	const real_t *z_ptr = z->ptr();
	real_t c = -Math_INF;

	for (int i = 0; i < size; ++i) {
		int zpi = z_ptr[i];

		if (c < zpi) {
			c = zpi;
		}
	}

	c = -c;

	Ref<MLPPVector> n = alg.scalar_addnv(c, z);

	return adj_softmax_normv(n);
}
Ref<MLPPMatrix> MLPPActivation::adj_softmax_derivm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> n = z->duplicate_fast();

	Size2i size = z->size();

	Ref<MLPPVector> row_rmp;
	row_rmp.instance();
	row_rmp->resize(size.x);

	for (int i = 0; i < size.y; ++i) {
		z->get_row_into_mlpp_vector(i, row_rmp);

		Ref<MLPPVector> nv = adj_softmax_derivv(row_rmp);

		n->set_row_mlpp_vector(i, nv);
	}

	return n;
}

//SOFTMAX DERIV

Ref<MLPPMatrix> MLPPActivation::softmax_deriv_normv(const Ref<MLPPVector> &z) {
	Ref<MLPPVector> a = softmax_normv(z);

	int z_size = z->size();
	int a_size = a->size();

	Ref<MLPPMatrix> deriv;
	deriv.instance();
	deriv->resize(Size2i(a_size, a_size));

	const real_t *a_ptr = a->ptr();

	for (int i = 0; i < z_size; ++i) {
		for (int j = 0; j < z_size; ++j) {
			if (i == j) {
				deriv->element_set(i, j, a_ptr[i] * (1 - a_ptr[i]));
			} else {
				deriv->element_set(i, j, -a_ptr[i] * a_ptr[j]);
			}
		}
	}

	return deriv;
}
Vector<Ref<MLPPMatrix>> MLPPActivation::softmax_deriv_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	int z_size_y = z->size().y;

	Ref<MLPPMatrix> a = softmax_normm(z);
	int a_size_y = a->size().y;
	int a_size_x = a->size().x;

	Vector<Ref<MLPPMatrix>> deriv;
	deriv.resize(a_size_y);

	Ref<MLPPVector> a_i_tmp;
	a_i_tmp.instance();
	a_i_tmp->resize(a_size_x);

	Ref<MLPPVector> a_j_tmp;
	a_j_tmp.instance();
	a_j_tmp->resize(a_size_x);

	for (int i = 0; i < deriv.size(); ++i) {
		Ref<MLPPMatrix> d;
		d.instance();
		d->resize(Size2i(a_size_x, z_size_y));

		for (int j = 0; j < z_size_y; ++j) {
			a->get_row_into_mlpp_vector(i, a_i_tmp);

			if (i == j) {
				Ref<MLPPVector> d_j = alg.subtractionnv(a_i_tmp, alg.hadamard_productnv(a_i_tmp, a_i_tmp));
				d->set_row_mlpp_vector(j, d_j);
			} else {
				a->get_row_into_mlpp_vector(j, a_j_tmp);
				Ref<MLPPVector> d_j = alg.scalar_multiplynv(-1, alg.hadamard_productnv(a_i_tmp, a_j_tmp));
				d->set_row_mlpp_vector(j, d_j);
			}
		}

		deriv.write[i] = d;
	}

	return deriv;
}

Ref<MLPPMatrix> MLPPActivation::softmax_deriv_derivv(const Ref<MLPPVector> &z) {
	Ref<MLPPVector> a = softmax_normv(z);

	int z_size = z->size();
	int a_size = a->size();

	Ref<MLPPMatrix> deriv;
	deriv.instance();
	deriv->resize(Size2i(a_size, a_size));

	const real_t *a_ptr = a->ptr();

	for (int i = 0; i < z_size; ++i) {
		for (int j = 0; j < z_size; ++j) {
			if (i == j) {
				deriv->element_set(i, j, a_ptr[i] * (1 - a_ptr[i]));
			} else {
				deriv->element_set(i, j, -a_ptr[i] * a_ptr[j]);
			}
		}
	}

	return deriv;
}
Vector<Ref<MLPPMatrix>> MLPPActivation::softmax_deriv_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	int z_size_y = z->size().y;

	Ref<MLPPMatrix> a = softmax_normm(z);
	int a_size_y = a->size().y;
	int a_size_x = a->size().x;

	Vector<Ref<MLPPMatrix>> deriv;
	deriv.resize(a_size_y);

	Ref<MLPPVector> a_i_tmp;
	a_i_tmp.instance();
	a_i_tmp->resize(a_size_x);

	Ref<MLPPVector> a_j_tmp;
	a_j_tmp.instance();
	a_j_tmp->resize(a_size_x);

	for (int i = 0; i < deriv.size(); ++i) {
		Ref<MLPPMatrix> d;
		d.instance();
		d->resize(Size2i(a_size_x, z_size_y));

		for (int j = 0; j < z_size_y; ++j) {
			a->get_row_into_mlpp_vector(i, a_i_tmp);

			if (i == j) {
				Ref<MLPPVector> d_j = alg.subtractionnv(a_i_tmp, alg.hadamard_productnv(a_i_tmp, a_i_tmp));
				d->set_row_mlpp_vector(j, d_j);
			} else {
				a->get_row_into_mlpp_vector(j, a_j_tmp);
				Ref<MLPPVector> d_j = alg.scalar_multiplynv(-1, alg.hadamard_productnv(a_i_tmp, a_j_tmp));
				d->set_row_mlpp_vector(j, d_j);
			}
		}

		deriv.write[i] = d;
	}

	return deriv;
}

//SOFTPLUS

real_t MLPPActivation::softplus_normr(real_t z) {
	return Math::log(1 + exp(z));
}
Ref<MLPPVector> MLPPActivation::softplus_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.lognv(alg.additionnv(alg.onevecnv(z->size()), alg.expnv(z)));
}
Ref<MLPPMatrix> MLPPActivation::softplus_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.lognv(alg.additionnv(alg.onematnm(z->size().x, z->size().y), alg.expnv(z)));
}

real_t MLPPActivation::softplus_derivr(real_t z) {
	return sigmoid_normr(z);
}
Ref<MLPPVector> MLPPActivation::softplus_derivv(const Ref<MLPPVector> &z) {
	return sigmoid_normv(z);
}
Ref<MLPPMatrix> MLPPActivation::softplus_derivm(const Ref<MLPPMatrix> &z) {
	return sigmoid_normm(z);
}

//SOFTSIGN

real_t MLPPActivation::softsign_normr(real_t z) {
	return z / (1 + abs(z));
}
Ref<MLPPVector> MLPPActivation::softsign_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(z, alg.additionnv(alg.onevecnv(z->size()), alg.absv(z)));
}
Ref<MLPPMatrix> MLPPActivation::softsign_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(z, alg.additionnv(alg.onematnm(z->size().x, z->size().y), alg.absnm(z)));
}

real_t MLPPActivation::softsign_derivr(real_t z) {
	return 1 / ((1 + abs(z)) * (1 + abs(z)));
}
Ref<MLPPVector> MLPPActivation::softsign_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(alg.onevecnv(z->size()), alg.exponentiatenv(alg.additionnv(alg.onevecnv(z->size()), alg.absv(z)), 2));
}
Ref<MLPPMatrix> MLPPActivation::softsign_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y), alg.exponentiatenv(alg.additionnm(alg.onematnm(z->size().x, z->size().y), alg.absnm(z)), 2));
}

//GAUSSIANCDF

real_t MLPPActivation::gaussian_cdf_normr(real_t z) {
	return 0.5 * (1 + erf(z / sqrt(2)));
}
Ref<MLPPVector> MLPPActivation::gaussian_cdf_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(0.5, alg.additionnv(alg.onevecnv(z->size()), alg.erfnv(alg.scalar_multiplynv(1 / sqrt(2), z))));
}

Ref<MLPPMatrix> MLPPActivation::gaussian_cdf_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynm(0.5, alg.additionnm(alg.onematnm(z->size().x, z->size().y), alg.erfnm(alg.scalar_multiplynm(1 / sqrt(2), z))));
}

real_t MLPPActivation::gaussian_cdf_derivr(real_t z) {
	return (1 / sqrt(2 * Math_PI)) * exp(-z * z / 2);
}
Ref<MLPPVector> MLPPActivation::gaussian_cdf_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(1 / Math::sqrt(2 * Math_PI), alg.expnv(alg.scalar_multiplynv(-1 / 2.0, alg.hadamard_productnv(z, z))));
}

Ref<MLPPMatrix> MLPPActivation::gaussian_cdf_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynm(1 / Math::sqrt(2 * Math_PI), alg.expnm(alg.scalar_multiplynm(-1 / 2.0, alg.hadamard_productnm(z, z))));
}

//CLOGLOG

real_t MLPPActivation::cloglog_normr(real_t z) {
	return 1 - exp(-exp(z));
}
Ref<MLPPVector> MLPPActivation::cloglog_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(-1, alg.scalar_addnv(-1, alg.expnv(alg.scalar_multiplynv(-1, alg.expnv(z)))));
}

Ref<MLPPMatrix> MLPPActivation::cloglog_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynm(-1, alg.scalar_addnm(-1, alg.expnm(alg.scalar_multiplynm(-1, alg.expnm(z)))));
}

real_t MLPPActivation::cloglog_derivr(real_t z) {
	return exp(z - exp(z));
}
Ref<MLPPVector> MLPPActivation::cloglog_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.expnv(alg.scalar_multiplynv(-1, alg.expnv(z)));
}

Ref<MLPPMatrix> MLPPActivation::cloglog_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.expnm(alg.scalar_multiplynm(-1, alg.expnm(z)));
}

//LOGIT

real_t MLPPActivation::logit_normr(real_t z) {
	return Math::log(z / (1 - z));
}
Ref<MLPPVector> MLPPActivation::logit_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.lognv(alg.element_wise_divisionnv(z, alg.subtractionnv(alg.onevecnv(z->size()), z)));
}
Ref<MLPPMatrix> MLPPActivation::logit_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.lognm(alg.element_wise_divisionnvnm(z, alg.subtractionnm(alg.onematnm(z->size().x, z->size().y), z)));
}

real_t MLPPActivation::logit_derivr(real_t z) {
	return 1 / z - 1 / (z - 1);
}
Ref<MLPPVector> MLPPActivation::logit_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.subtractionnv(
			alg.element_wise_divisionnv(alg.onevecnv(z->size()), z),
			alg.element_wise_divisionnv(alg.onevecnv(z->size()), alg.subtractionnv(z, alg.onevecnv(z->size()))));
}
Ref<MLPPMatrix> MLPPActivation::logit_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.subtractionnm(
			alg.element_wise_divisionnvnm(
					alg.onematnm(z->size().x, z->size().y), z),
			alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y),
					alg.subtractionnm(z, alg.onematnm(z->size().x, z->size().y))));
}

//UNITSTEP

real_t MLPPActivation::unit_step_normr(real_t z) {
	return z < 0 ? 0 : 1;
}
Ref<MLPPVector> MLPPActivation::unit_step_normv(const Ref<MLPPVector> &z) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = unit_step_normr(z_ptr[i]);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::unit_step_normm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = unit_step_normr(z_ptr[i]);
	}

	return a;
}

real_t MLPPActivation::unit_step_derivr(real_t z) {
	return 0;
}
Ref<MLPPVector> MLPPActivation::unit_step_derivv(const Ref<MLPPVector> &z) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());
	a->fill(0);

	return a;
}
Ref<MLPPMatrix> MLPPActivation::unit_step_derivm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());
	a->fill(0);

	return a;
}

//SWISH

real_t MLPPActivation::swish_normr(real_t z) {
	return z * sigmoid_normr(z);
}
Ref<MLPPVector> MLPPActivation::swish_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnv(z, sigmoid_normv(z));
}
Ref<MLPPMatrix> MLPPActivation::swish_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnv(z, sigmoid_normm(z));
}

real_t MLPPActivation::swish_derivr(real_t z) {
	return swish_normr(z) + sigmoid_normr(z) * (1 - swish_normr(z));
}
Ref<MLPPVector> MLPPActivation::swish_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.additionnv(swish_normv(z), alg.subtractionnv(sigmoid_normv(z), alg.hadamard_productnv(sigmoid_normv(z), swish_normv(z))));
}
Ref<MLPPMatrix> MLPPActivation::swish_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.additionnv(swish_normm(z), alg.subtractionnv(sigmoid_normm(z), alg.hadamard_productnm(sigmoid_normm(z), swish_normm(z))));
}

//MISH

real_t MLPPActivation::mish_normr(real_t z) {
	return z * tanh(softplus_normr(z));
}
Ref<MLPPVector> MLPPActivation::mish_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnv(z, tanh_normv(softplus_normv(z)));
}
Ref<MLPPMatrix> MLPPActivation::mish_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnm(z, tanh_normm(softplus_normm(z)));
}

real_t MLPPActivation::mish_derivr(real_t z) {
	return sech_normr(softplus_normr(z)) * sech_normr(softplus_normr(z)) * z * sigmoid_normr(z) + mish_normr(z) / z;
}
Ref<MLPPVector> MLPPActivation::mish_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.additionnv(
			alg.hadamard_productnv(
					alg.hadamard_productnv(
							alg.hadamard_productnv(
									sech_normv(softplus_normv(z)), sech_normv(softplus_normv(z))),
							z),
					sigmoid_normv(z)),
			alg.element_wise_divisionnv(mish_normv(z), z));
}
Ref<MLPPMatrix> MLPPActivation::mish_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.additionnv(
			alg.hadamard_productnm(
					alg.hadamard_productnm(
							alg.hadamard_productnm(
									sech_normm(softplus_normm(z)), sech_normm(softplus_normm(z))),
							z),
					sigmoid_normm(z)),
			alg.element_wise_divisionnvnm(mish_normm(z), z));
}

//SINC

real_t MLPPActivation::sinc_normr(real_t z) {
	return Math::sin(z) / z;
}
Ref<MLPPVector> MLPPActivation::sinc_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(alg.sinnv(z), z);
}
Ref<MLPPMatrix> MLPPActivation::sinc_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(alg.sinnm(z), z);
}

real_t MLPPActivation::sinc_derivr(real_t z) {
	return (z * Math::cos(z) - Math::sin(z)) / (z * z);
}
Ref<MLPPVector> MLPPActivation::sinc_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(alg.subtractionnv(alg.hadamard_productnv(z, alg.cosnv(z)), alg.sinnv(z)), alg.hadamard_productnv(z, z));
}
Ref<MLPPMatrix> MLPPActivation::sinc_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(alg.subtractionnm(alg.hadamard_productnm(z, alg.cosnm(z)), alg.sinnm(z)), alg.hadamard_productnm(z, z));
}

//RELU

real_t MLPPActivation::relu_normr(real_t z) {
	return fmax(0, z);
}
Ref<MLPPVector> MLPPActivation::relu_normv(const Ref<MLPPVector> &z) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = relu_normr(z_ptr[i]);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::relu_normm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = relu_normr(z_ptr[i]);
	}

	return a;
}

real_t MLPPActivation::relu_derivr(real_t z) {
	if (z <= 0) {
		return 0;
	} else {
		return 1;
	}
}
Ref<MLPPVector> MLPPActivation::relu_derivv(const Ref<MLPPVector> &z) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = relu_derivr(z_ptr[i]);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::relu_derivm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = relu_derivr(z_ptr[i]);
	}

	return a;
}

//LEAKYRELU

real_t MLPPActivation::leaky_relu_normr(real_t z, real_t c) {
	return fmax(c * z, z);
}
Ref<MLPPVector> MLPPActivation::leaky_relu_normv(const Ref<MLPPVector> &z, real_t c) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = leaky_relu_normr(z_ptr[i], c);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::leaky_relu_normm(const Ref<MLPPMatrix> &z, real_t c) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = leaky_relu_normr(z_ptr[i], c);
	}

	return a;
}

real_t MLPPActivation::leaky_relu_derivr(real_t z, real_t c) {
	if (z <= 0) {
		return c;
	} else {
		return 1;
	}
}
Ref<MLPPVector> MLPPActivation::leaky_relu_derivv(const Ref<MLPPVector> &z, real_t c) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = leaky_relu_derivr(z_ptr[i], c);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::leaky_relu_derivm(const Ref<MLPPMatrix> &z, real_t c) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = leaky_relu_derivr(z_ptr[i], c);
	}

	return a;
}

//ELU

real_t MLPPActivation::elu_normr(real_t z, real_t c) {
	if (z >= 0) {
		return z;
	} else {
		return c * (exp(z) - 1);
	}
}
Ref<MLPPVector> MLPPActivation::elu_normv(const Ref<MLPPVector> &z, real_t c) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = elu_normr(z_ptr[i], c);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::elu_normm(const Ref<MLPPMatrix> &z, real_t c) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = elu_normr(z_ptr[i], c);
	}

	return a;
}

real_t MLPPActivation::elu_derivr(real_t z, real_t c) {
	if (z <= 0) {
		return c * exp(z);
	} else {
		return 1;
	}
}
Ref<MLPPVector> MLPPActivation::elu_derivv(const Ref<MLPPVector> &z, real_t c) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = elu_derivr(z_ptr[i], c);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::elu_derivm(const Ref<MLPPMatrix> &z, real_t c) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = elu_derivr(z_ptr[i], c);
	}

	return a;
}

//SELU

real_t MLPPActivation::selu_normr(real_t z, real_t lambda, real_t c) {
	return lambda * elu_normr(z, c);
}
Ref<MLPPVector> MLPPActivation::selu_normv(const Ref<MLPPVector> &z, real_t lambda, real_t c) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = selu_normr(z_ptr[i], lambda, c);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::selu_normm(const Ref<MLPPMatrix> &z, real_t lambda, real_t c) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = selu_normr(z_ptr[i], lambda, c);
	}

	return a;
}

real_t MLPPActivation::selu_derivr(real_t z, real_t lambda, real_t c) {
	return elu_derivr(z, c);
}
Ref<MLPPVector> MLPPActivation::selu_derivv(const Ref<MLPPVector> &z, real_t lambda, real_t c) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = selu_derivr(z_ptr[i], lambda, c);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::selu_derivm(const Ref<MLPPMatrix> &z, real_t lambda, real_t c) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = selu_derivr(z_ptr[i], lambda, c);
	}

	return a;
}

//GELU

real_t MLPPActivation::gelu_normr(real_t z) {
	return 0.5 * z * (1 + tanh(sqrt(2 / Math_PI) * (z + 0.044715 * Math::pow(z, 3))));
}
Ref<MLPPVector> MLPPActivation::gelu_normv(const Ref<MLPPVector> &z) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = gelu_normr(z_ptr[i]);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::gelu_normm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = gelu_normr(z_ptr[i]);
	}

	return a;
}

real_t MLPPActivation::gelu_derivr(real_t z) {
	return 0.5 * tanh_normr(0.0356774 * Math::pow(z, 3) + 0.797885 * z) + (0.0535161 * Math::pow(z, 3) + 0.398942 * z) * Math::pow(sech_normr(0.0356774 * Math::pow(z, 3) + 0.797885 * z), 2) + 0.5;
}
Ref<MLPPVector> MLPPActivation::gelu_derivv(const Ref<MLPPVector> &z) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = gelu_derivr(z_ptr[i]);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::gelu_derivm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = gelu_derivr(z_ptr[i]);
	}

	return a;
}

//SIGN

real_t MLPPActivation::sign_normr(real_t z) {
	if (z < 0) {
		return -1;
	} else if (z == 0) {
		return 0;
	} else {
		return 1;
	}
}
Ref<MLPPVector> MLPPActivation::sign_normv(const Ref<MLPPVector> &z) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = sign_normr(z_ptr[i]);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::sign_normm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = sign_normr(z_ptr[i]);
	}

	return a;
}

real_t MLPPActivation::sign_derivr(real_t z) {
	return 0;
}
Ref<MLPPVector> MLPPActivation::sign_derivv(const Ref<MLPPVector> &z) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());

	int z_size = z->size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_size; ++i) {
		a_ptr[i] = sign_derivr(z_ptr[i]);
	}

	return a;
}
Ref<MLPPMatrix> MLPPActivation::sign_derivm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());

	int z_data_size = z->data_size();

	const real_t *z_ptr = z->ptr();
	real_t *a_ptr = a->ptrw();

	for (int i = 0; i < z_data_size; ++i) {
		a_ptr[i] = sign_derivr(z_ptr[i]);
	}

	return a;
}

//SINH

real_t MLPPActivation::sinh_normr(real_t z) {
	return 0.5 * (Math::exp(z) - Math::exp(-z));
}
Ref<MLPPVector> MLPPActivation::sinh_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;
	return alg.scalar_multiplynv(0.5, alg.subtractionnv(alg.expnv(z), alg.expnv(alg.scalar_multiplynv(-1, z))));
}
Ref<MLPPMatrix> MLPPActivation::sinh_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;
	return alg.scalar_multiplynm(0.5, alg.subtractionnm(alg.expnm(z), alg.expnm(alg.scalar_multiplynm(-1, z))));
}

real_t MLPPActivation::sinh_derivr(real_t z) {
	return cosh_normr(z);
}
Ref<MLPPVector> MLPPActivation::sinh_derivv(const Ref<MLPPVector> &z) {
	return cosh_normv(z);
}
Ref<MLPPMatrix> MLPPActivation::sinh_derivm(const Ref<MLPPMatrix> &z) {
	return cosh_normm(z);
}

//COSH

real_t MLPPActivation::cosh_normr(real_t z) {
	return 0.5 * (Math::exp(z) + Math::exp(-z));
}
Ref<MLPPVector> MLPPActivation::cosh_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;
	return alg.scalar_multiplynv(0.5, alg.additionnv(alg.expnv(z), alg.expnv(alg.scalar_multiplynv(-1, z))));
}
Ref<MLPPMatrix> MLPPActivation::cosh_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;
	return alg.scalar_multiplynm(0.5, alg.additionnv(alg.expnm(z), alg.expnm(alg.scalar_multiplynm(-1, z))));
}

real_t MLPPActivation::cosh_derivr(real_t z) {
	return sinh_normr(z);
}
Ref<MLPPVector> MLPPActivation::cosh_derivv(const Ref<MLPPVector> &z) {
	return sinh_normv(z);
}
Ref<MLPPMatrix> MLPPActivation::cosh_derivm(const Ref<MLPPMatrix> &z) {
	return sinh_normm(z);
}

//TANH

real_t MLPPActivation::tanh_normr(real_t z) {
	return (Math::exp(z) - Math::exp(-z)) / (Math::exp(z) + Math::exp(-z));
}
Ref<MLPPVector> MLPPActivation::tanh_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(alg.subtractionnv(alg.expnv(z), alg.expnv(alg.scalar_multiplynv(-1, z))), alg.additionnv(alg.expnv(z), alg.expnv(alg.scalar_multiplynv(-1, z))));
}
Ref<MLPPMatrix> MLPPActivation::tanh_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(alg.subtractionnm(alg.expnm(z), alg.expnm(alg.scalar_multiplynm(-1, z))), alg.additionnm(alg.expnm(z), alg.expnm(alg.scalar_multiplynm(-1, z))));
}

real_t MLPPActivation::tanh_derivr(real_t z) {
	return 1 - tanh(z) * tanh(z);
}
Ref<MLPPVector> MLPPActivation::tanh_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(-1, alg.scalar_addnv(-1, alg.hadamard_productnv(tanh_normv(z), tanh_normv(z))));
}
Ref<MLPPMatrix> MLPPActivation::tanh_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynm(-1, alg.scalar_addnm(-1, alg.hadamard_productnm(tanh_normm(z), tanh_normm(z))));
}

//CSCH

real_t MLPPActivation::csch_normr(real_t z) {
	return 1 / sinh(z);
}
Ref<MLPPVector> MLPPActivation::csch_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(alg.onevecnv(z->size()), sinh_normv(z));
}

Ref<MLPPMatrix> MLPPActivation::csch_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y), sinh_normm(z));
}

real_t MLPPActivation::csch_derivr(real_t z) {
	return -csch_normr(z) * coth_normr(z);
}
Ref<MLPPVector> MLPPActivation::csch_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnv(alg.scalar_multiplynv(-1, csch_normv(z)), coth_normv(z));
}

Ref<MLPPMatrix> MLPPActivation::csch_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnm(alg.scalar_multiplynm(-1, csch_normm(z)), coth_normm(z));
}

//SECH

real_t MLPPActivation::sech_normr(real_t z) {
	return 1 / cosh(z);
}

Ref<MLPPVector> MLPPActivation::sech_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(alg.onevecnv(z->size()), cosh_normv(z));

	// return activation(z, deriv, static_cast<void (*)(real_t, bool)>(&sech));
}
Ref<MLPPMatrix> MLPPActivation::sech_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y), cosh_normm(z));

	// return activation(z, deriv, static_cast<void (*)(real_t, bool)>(&sech));
}

real_t MLPPActivation::sech_derivr(real_t z) {
	return -sech_normr(z) * tanh_normr(z);
}

Ref<MLPPVector> MLPPActivation::sech_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnv(alg.scalar_multiplynv(-1, sech_normv(z)), tanh_normv(z));
}
Ref<MLPPMatrix> MLPPActivation::sech_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnm(alg.scalar_multiplynm(-1, sech_normm(z)), tanh_normm(z));
}

//COTH

real_t MLPPActivation::coth_normr(real_t z) {
	return 1 / tanh(z);
}
Ref<MLPPVector> MLPPActivation::coth_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(alg.onevecnv(z->size()), tanh_normv(z));
}
Ref<MLPPMatrix> MLPPActivation::coth_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y), tanh_normm(z));
}

real_t MLPPActivation::coth_derivr(real_t z) {
	return -csch_normr(z) * csch_normr(z);
}
Ref<MLPPVector> MLPPActivation::coth_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnv(alg.scalar_multiplynv(-1, csch_normv(z)), csch_normv(z));
}
Ref<MLPPMatrix> MLPPActivation::coth_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnm(alg.scalar_multiplynm(-1, csch_normm(z)), csch_normm(z));
}

//ARSINH

real_t MLPPActivation::arsinh_normr(real_t z) {
	return Math::log(z + sqrt(z * z + 1));
}

Ref<MLPPVector> MLPPActivation::arsinh_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.lognv(alg.additionnv(z, alg.sqrtnv(alg.additionnv(alg.hadamard_productnv(z, z), alg.onevecnv(z->size())))));
}

Ref<MLPPMatrix> MLPPActivation::arsinh_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.lognm(alg.additionnm(z, alg.sqrtnm(alg.additionnm(alg.hadamard_productnm(z, z), alg.onematnm(z->size().x, z->size().y)))));
}

real_t MLPPActivation::arsinh_derivr(real_t z) {
	return 1 / sqrt(z * z + 1);
}

Ref<MLPPVector> MLPPActivation::arsinh_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(alg.onevecnv(z->size()), alg.sqrtnv(alg.additionnv(alg.hadamard_productnv(z, z), alg.onevecnv(z->size()))));
}

Ref<MLPPMatrix> MLPPActivation::arsinh_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y), alg.sqrtnm(alg.additionnm(alg.hadamard_productnm(z, z), alg.onematnm(z->size().x, z->size().y))));
}

//ARCOSH

real_t MLPPActivation::arcosh_normr(real_t z) {
	return Math::log(z + sqrt(z * z - 1));
}
Ref<MLPPVector> MLPPActivation::arcosh_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.lognv(alg.additionnv(z, alg.sqrtnv(alg.subtractionnv(alg.hadamard_productnv(z, z), alg.onevecnv(z->size())))));
}

Ref<MLPPMatrix> MLPPActivation::arcosh_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.lognm(alg.additionnm(z, alg.sqrtnm(alg.subtractionnm(alg.hadamard_productnm(z, z), alg.onematnm(z->size().x, z->size().y)))));
}

real_t MLPPActivation::arcosh_derivr(real_t z) {
	return 1 / sqrt(z * z - 1);
}
Ref<MLPPVector> MLPPActivation::arcosh_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(alg.onevecnv(z->size()), alg.sqrtnv(alg.subtractionnv(alg.hadamard_productnv(z, z), alg.onevecnv(z->size()))));
}

Ref<MLPPMatrix> MLPPActivation::arcosh_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y), alg.sqrtnm(alg.subtractionnm(alg.hadamard_productnm(z, z), alg.onematnm(z->size().x, z->size().y))));
}

//ARTANH

real_t MLPPActivation::artanh_normr(real_t z) {
	return 0.5 * Math::log((1 + z) / (1 - z));
}
Ref<MLPPVector> MLPPActivation::artanh_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(0.5, alg.lognv(alg.element_wise_divisionnv(alg.additionnv(alg.onevecnv(z->size()), z), alg.subtractionnv(alg.onevecnv(z->size()), z))));
}

Ref<MLPPMatrix> MLPPActivation::artanh_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynm(0.5, alg.lognm(alg.element_wise_divisionnvnm(alg.additionnm(alg.onematnm(z->size().x, z->size().y), z), alg.subtractionnm(alg.onematnm(z->size().x, z->size().y), z))));
}

real_t MLPPActivation::artanh_derivr(real_t z) {
	return 1 / (1 - z * z);
}
Ref<MLPPVector> MLPPActivation::artanh_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(alg.onevecnv(z->size()), alg.subtractionnv(alg.onevecnv(z->size()), alg.hadamard_productnv(z, z)));
}

Ref<MLPPMatrix> MLPPActivation::artanh_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y), alg.subtractionnv(alg.onematnm(z->size().x, z->size().y), alg.hadamard_productnm(z, z)));
}

//ARCSCH

real_t MLPPActivation::arcsch_normr(real_t z) {
	return Math::log(sqrt(1 + (1 / (z * z))) + (1 / z));
}
Ref<MLPPVector> MLPPActivation::arcsch_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.lognv(
			alg.additionnv(
					alg.sqrtnv(
							alg.additionnv(
									alg.onevecnv(z->size()),
									alg.element_wise_divisionnv(alg.onevecnv(z->size()), alg.hadamard_productnv(z, z)))),
					alg.element_wise_divisionnv(alg.onevecnv(z->size()), z)));
}
Ref<MLPPMatrix> MLPPActivation::arcsch_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.lognm(
			alg.additionnm(
					alg.sqrtnm(
							alg.additionnm(alg.onematnm(z->size().x, z->size().y),
									alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y), alg.hadamard_productnm(z, z)))),
					alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y), z)));
}

real_t MLPPActivation::arcsch_derivr(real_t z) {
	return -1 / ((z * z) * sqrt(1 + (1 / (z * z))));
}
Ref<MLPPVector> MLPPActivation::arcsch_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(
			alg.fullnv(z->size(), -1),
			alg.hadamard_productnm(
					alg.hadamard_productnv(z, z),
					alg.sqrtnv(alg.additionnv(alg.onevecnv(z->size()), alg.element_wise_divisionnv(alg.onevecnv(z->size()), alg.hadamard_productnv(z, z))))));
}
Ref<MLPPMatrix> MLPPActivation::arcsch_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(
			alg.fullnm(z->size().x, z->size().y, -1),
			alg.hadamard_productnm(alg.hadamard_productnm(z, z),
					alg.sqrtnm(alg.additionnm(alg.onematnm(z->size().x, z->size().y),
							alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y), alg.hadamard_productnm(z, z))))));
}

//ARSECH

real_t MLPPActivation::arsech_normr(real_t z) {
	return Math::log((1 / z) + ((1 / z) + 1) * ((1 / z) - 1));
}

Ref<MLPPVector> MLPPActivation::arsech_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.lognv(
			alg.additionnv(
					alg.element_wise_divisionnv(
							alg.onevecnv(z->size()), z),
					alg.hadamard_productnv(
							alg.additionnv(alg.element_wise_divisionnv(alg.onevecnv(z->size()), z), alg.onevecnv(z->size())),
							alg.subtractionnv(alg.element_wise_divisionnv(alg.onevecnv(z->size()), z), alg.onevecnv(z->size())))));
}

Ref<MLPPMatrix> MLPPActivation::arsech_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.lognm(
			alg.additionnm(
					alg.element_wise_divisionnvnm(
							alg.onematnm(z->size().x, z->size().y), z),
					alg.hadamard_productnm(
							alg.additionnm(
									alg.element_wise_divisionnvnm(
											alg.onematnm(z->size().x, z->size().y), z),
									alg.onematnm(z->size().x, z->size().y)),
							alg.subtractionnm(
									alg.element_wise_divisionnvnm(
											alg.onematnm(z->size().x, z->size().y), z),
									alg.onematnm(z->size().x, z->size().y)))));
}

real_t MLPPActivation::arsech_derivr(real_t z) {
	return -1 / (z * sqrt(1 - z * z));
}

Ref<MLPPVector> MLPPActivation::arsech_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(
			alg.fullnv(z->size(), -1),
			alg.hadamard_productnv(
					z,
					alg.sqrtnv(
							alg.subtractionnv(alg.onevecnv(z->size()), alg.hadamard_productnv(z, z)))));
}

Ref<MLPPMatrix> MLPPActivation::arsech_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(
			alg.fullnm(z->size().x, z->size().y, -1),
			alg.hadamard_productnm(
					z,
					alg.sqrtnm(alg.subtractionnm(alg.onematnm(z->size().x, z->size().y), alg.hadamard_productnm(z, z)))));
}

//ARCOTH

real_t MLPPActivation::arcoth_normr(real_t z) {
	return 0.5 * Math::log((1 + z) / (z - 1));
}
Ref<MLPPVector> MLPPActivation::arcoth_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(
			0.5,
			alg.lognv(alg.element_wise_divisionnv(alg.additionnv(alg.onevecnv(z->size()), z), alg.subtractionnv(z, alg.onevecnv(z->size())))));
}

Ref<MLPPMatrix> MLPPActivation::arcoth_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynm(
			0.5,
			alg.lognm(alg.element_wise_divisionnvnm(alg.additionnm(alg.onematnm(z->size().x, z->size().y), z), alg.subtractionnm(z, alg.onematnm(z->size().x, z->size().y)))));
}

real_t MLPPActivation::arcoth_derivr(real_t z) {
	return 1 / (1 - z * z);
}
Ref<MLPPVector> MLPPActivation::arcoth_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnv(alg.onevecnv(z->size()), alg.subtractionnv(alg.onevecnv(z->size()), alg.hadamard_productnv(z, z)));
}

Ref<MLPPMatrix> MLPPActivation::arcoth_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionnvnm(alg.onematnm(z->size().x, z->size().y), alg.subtractionnm(alg.onematnm(z->size().x, z->size().y), alg.hadamard_productnm(z, z)));
}

void MLPPActivation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("run_activation_real", "func", "z", "deriv"), &MLPPActivation::run_activation_real, false);
	ClassDB::bind_method(D_METHOD("run_activation_vector", "func", "z", "deriv"), &MLPPActivation::run_activation_vector, false);
	ClassDB::bind_method(D_METHOD("run_activation_matrix", "func", "z", "deriv"), &MLPPActivation::run_activation_matrix, false);

	ClassDB::bind_method(D_METHOD("run_activation_norm_real", "func", "z"), &MLPPActivation::run_activation_norm_real);
	ClassDB::bind_method(D_METHOD("run_activation_norm_vector", "func", "z"), &MLPPActivation::run_activation_norm_vector);
	ClassDB::bind_method(D_METHOD("run_activation_norm_matrix", "func", "z"), &MLPPActivation::run_activation_norm_matrix);

	real_t run_activation_norm_real(const ActivationFunction func, const real_t z);
	Ref<MLPPVector> run_activation_norm_vector(const ActivationFunction func, const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> run_activation_norm_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z);

	ClassDB::bind_method(D_METHOD("run_activation_deriv_real", "func", "z"), &MLPPActivation::run_activation_deriv_real);
	ClassDB::bind_method(D_METHOD("run_activation_deriv_vector", "func", "z"), &MLPPActivation::run_activation_deriv_vector);
	ClassDB::bind_method(D_METHOD("run_activation_deriv_matrix", "func", "z"), &MLPPActivation::run_activation_deriv_matrix);

	//LINEAR

	ClassDB::bind_method(D_METHOD("linear_normr", "z"), &MLPPActivation::linear_normr);
	ClassDB::bind_method(D_METHOD("linear_normv", "z"), &MLPPActivation::linear_normv);
	ClassDB::bind_method(D_METHOD("linear_normm", "z"), &MLPPActivation::linear_normm);

	ClassDB::bind_method(D_METHOD("linear_derivr", "z"), &MLPPActivation::linear_derivr);
	ClassDB::bind_method(D_METHOD("linear_derivv", "z"), &MLPPActivation::linear_derivv);
	ClassDB::bind_method(D_METHOD("linear_derivm", "z"), &MLPPActivation::linear_derivm);

	//SIGMOID

	ClassDB::bind_method(D_METHOD("sigmoid_normr", "z"), &MLPPActivation::sigmoid_normr);
	ClassDB::bind_method(D_METHOD("sigmoid_normv", "z"), &MLPPActivation::sigmoid_normv);
	ClassDB::bind_method(D_METHOD("sigmoid_normm", "z"), &MLPPActivation::sigmoid_normm);

	ClassDB::bind_method(D_METHOD("sigmoid_derivr", "z"), &MLPPActivation::sigmoid_derivr);
	ClassDB::bind_method(D_METHOD("sigmoid_derivv", "z"), &MLPPActivation::sigmoid_derivv);
	ClassDB::bind_method(D_METHOD("sigmoid_derivm", "z"), &MLPPActivation::sigmoid_derivm);

	//SOFTMAX

	ClassDB::bind_method(D_METHOD("softmax_normr", "z"), &MLPPActivation::softmax_normr);
	ClassDB::bind_method(D_METHOD("softmax_normv", "z"), &MLPPActivation::softmax_normv);
	ClassDB::bind_method(D_METHOD("softmax_normm", "z"), &MLPPActivation::softmax_normm);

	ClassDB::bind_method(D_METHOD("softmax_derivr", "z"), &MLPPActivation::softmax_derivr);
	ClassDB::bind_method(D_METHOD("softmax_derivv", "z"), &MLPPActivation::softmax_derivv);
	ClassDB::bind_method(D_METHOD("softmax_derivm", "z"), &MLPPActivation::softmax_derivm);

	//ADJ_SOFTMAX

	real_t adj_softmax_normr(real_t z);
	Ref<MLPPVector> adj_softmax_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> adj_softmax_normm(const Ref<MLPPMatrix> &z);

	real_t adj_softmax_derivr(real_t z);
	Ref<MLPPVector> adj_softmax_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> adj_softmax_derivm(const Ref<MLPPMatrix> &z);

	//SOFTPLUS

	ClassDB::bind_method(D_METHOD("softplus_normr", "z"), &MLPPActivation::softplus_normr);
	ClassDB::bind_method(D_METHOD("softplus_normv", "z"), &MLPPActivation::softplus_normv);
	ClassDB::bind_method(D_METHOD("softplus_normm", "z"), &MLPPActivation::softplus_normm);

	ClassDB::bind_method(D_METHOD("softplus_derivr", "z"), &MLPPActivation::softplus_derivr);
	ClassDB::bind_method(D_METHOD("softplus_derivv", "z"), &MLPPActivation::softplus_derivv);
	ClassDB::bind_method(D_METHOD("softplus_derivm", "z"), &MLPPActivation::softplus_derivm);

	//SOFTSIGN

	ClassDB::bind_method(D_METHOD("softsign_normr", "z"), &MLPPActivation::softsign_normr);
	ClassDB::bind_method(D_METHOD("softsign_normv", "z"), &MLPPActivation::softsign_normv);
	ClassDB::bind_method(D_METHOD("softsign_normm", "z"), &MLPPActivation::softsign_normm);

	ClassDB::bind_method(D_METHOD("softsign_derivr", "z"), &MLPPActivation::softsign_derivr);
	ClassDB::bind_method(D_METHOD("softsign_derivv", "z"), &MLPPActivation::softsign_derivv);
	ClassDB::bind_method(D_METHOD("softsign_derivm", "z"), &MLPPActivation::softsign_derivm);

	//GAUSSIANCDF

	ClassDB::bind_method(D_METHOD("gaussian_cdf_normr", "z"), &MLPPActivation::gaussian_cdf_normr);
	ClassDB::bind_method(D_METHOD("gaussian_cdf_normv", "z"), &MLPPActivation::gaussian_cdf_normv);
	ClassDB::bind_method(D_METHOD("gaussian_cdf_normm", "z"), &MLPPActivation::gaussian_cdf_normm);

	ClassDB::bind_method(D_METHOD("gaussian_cdf_derivr", "z"), &MLPPActivation::gaussian_cdf_derivr);
	ClassDB::bind_method(D_METHOD("gaussian_cdf_derivv", "z"), &MLPPActivation::gaussian_cdf_derivv);
	ClassDB::bind_method(D_METHOD("gaussian_cdf_derivm", "z"), &MLPPActivation::gaussian_cdf_derivm);

	//CLOGLOG

	ClassDB::bind_method(D_METHOD("cloglog_normr", "z"), &MLPPActivation::cloglog_normr);
	ClassDB::bind_method(D_METHOD("cloglog_normv", "z"), &MLPPActivation::cloglog_normv);
	ClassDB::bind_method(D_METHOD("cloglog_normm", "z"), &MLPPActivation::cloglog_normm);

	ClassDB::bind_method(D_METHOD("cloglog_derivr", "z"), &MLPPActivation::cloglog_derivr);
	ClassDB::bind_method(D_METHOD("cloglog_derivv", "z"), &MLPPActivation::cloglog_derivv);
	ClassDB::bind_method(D_METHOD("cloglog_derivm", "z"), &MLPPActivation::cloglog_derivm);

	//LOGIT

	ClassDB::bind_method(D_METHOD("logit_normr", "z"), &MLPPActivation::logit_normr);
	ClassDB::bind_method(D_METHOD("logit_normv", "z"), &MLPPActivation::logit_normv);
	ClassDB::bind_method(D_METHOD("logit_normm", "z"), &MLPPActivation::logit_normm);

	ClassDB::bind_method(D_METHOD("logit_derivr", "z"), &MLPPActivation::logit_derivr);
	ClassDB::bind_method(D_METHOD("logit_derivv", "z"), &MLPPActivation::logit_derivv);
	ClassDB::bind_method(D_METHOD("logit_derivm", "z"), &MLPPActivation::logit_derivm);

	//UNITSTEP

	ClassDB::bind_method(D_METHOD("unit_step_normr", "z"), &MLPPActivation::unit_step_normr);
	ClassDB::bind_method(D_METHOD("unit_step_normv", "z"), &MLPPActivation::unit_step_normv);
	ClassDB::bind_method(D_METHOD("unit_step_normm", "z"), &MLPPActivation::unit_step_normm);

	ClassDB::bind_method(D_METHOD("unit_step_derivr", "z"), &MLPPActivation::unit_step_derivr);
	ClassDB::bind_method(D_METHOD("unit_step_derivv", "z"), &MLPPActivation::unit_step_derivv);
	ClassDB::bind_method(D_METHOD("unit_step_derivm", "z"), &MLPPActivation::unit_step_derivm);

	//SWISH

	ClassDB::bind_method(D_METHOD("swish_normr", "z"), &MLPPActivation::swish_normr);
	ClassDB::bind_method(D_METHOD("swish_normv", "z"), &MLPPActivation::swish_normv);
	ClassDB::bind_method(D_METHOD("swish_normm", "z"), &MLPPActivation::swish_normm);

	ClassDB::bind_method(D_METHOD("swish_derivr", "z"), &MLPPActivation::swish_derivr);
	ClassDB::bind_method(D_METHOD("swish_derivv", "z"), &MLPPActivation::swish_derivv);
	ClassDB::bind_method(D_METHOD("swish_derivm", "z"), &MLPPActivation::swish_derivm);

	//MISH

	ClassDB::bind_method(D_METHOD("mish_normr", "z"), &MLPPActivation::mish_normr);
	ClassDB::bind_method(D_METHOD("mish_normv", "z"), &MLPPActivation::mish_normv);
	ClassDB::bind_method(D_METHOD("mish_normm", "z"), &MLPPActivation::mish_normm);

	ClassDB::bind_method(D_METHOD("mish_derivr", "z"), &MLPPActivation::mish_derivr);
	ClassDB::bind_method(D_METHOD("mish_derivv", "z"), &MLPPActivation::mish_derivv);
	ClassDB::bind_method(D_METHOD("mish_derivm", "z"), &MLPPActivation::mish_derivm);

	//SINC

	ClassDB::bind_method(D_METHOD("sinc_normr", "z"), &MLPPActivation::sinc_normr);
	ClassDB::bind_method(D_METHOD("sinc_normv", "z"), &MLPPActivation::sinc_normv);
	ClassDB::bind_method(D_METHOD("sinc_normm", "z"), &MLPPActivation::sinc_normm);

	ClassDB::bind_method(D_METHOD("sinc_derivr", "z"), &MLPPActivation::sinc_derivr);
	ClassDB::bind_method(D_METHOD("sinc_derivv", "z"), &MLPPActivation::sinc_derivv);
	ClassDB::bind_method(D_METHOD("sinc_derivm", "z"), &MLPPActivation::sinc_derivm);

	//RELU

	ClassDB::bind_method(D_METHOD("relu_normr", "z"), &MLPPActivation::relu_normr);
	ClassDB::bind_method(D_METHOD("relu_normv", "z"), &MLPPActivation::relu_normv);
	ClassDB::bind_method(D_METHOD("relu_normm", "z"), &MLPPActivation::relu_normm);

	ClassDB::bind_method(D_METHOD("relu_derivr", "z"), &MLPPActivation::relu_derivr);
	ClassDB::bind_method(D_METHOD("relu_derivv", "z"), &MLPPActivation::relu_derivv);
	ClassDB::bind_method(D_METHOD("relu_derivm", "z"), &MLPPActivation::relu_derivm);

	//LEAKYRELU

	ClassDB::bind_method(D_METHOD("leaky_relu_normr", "z"), &MLPPActivation::leaky_relu_normr);
	ClassDB::bind_method(D_METHOD("leaky_relu_normv", "z"), &MLPPActivation::leaky_relu_normv);
	ClassDB::bind_method(D_METHOD("leaky_relu_normm", "z"), &MLPPActivation::leaky_relu_normm);

	ClassDB::bind_method(D_METHOD("leaky_relu_derivr", "z"), &MLPPActivation::leaky_relu_derivr);
	ClassDB::bind_method(D_METHOD("leaky_relu_derivv", "z"), &MLPPActivation::leaky_relu_derivv);
	ClassDB::bind_method(D_METHOD("leaky_relu_derivm", "z"), &MLPPActivation::leaky_relu_derivm);

	//ELU

	ClassDB::bind_method(D_METHOD("elu_normr", "z"), &MLPPActivation::elu_normr);
	ClassDB::bind_method(D_METHOD("elu_normv", "z"), &MLPPActivation::elu_normv);
	ClassDB::bind_method(D_METHOD("elu_normm", "z"), &MLPPActivation::elu_normm);

	ClassDB::bind_method(D_METHOD("elu_derivr", "z"), &MLPPActivation::elu_derivr);
	ClassDB::bind_method(D_METHOD("elu_derivv", "z"), &MLPPActivation::elu_derivv);
	ClassDB::bind_method(D_METHOD("elu_derivm", "z"), &MLPPActivation::elu_derivm);

	//SELU

	ClassDB::bind_method(D_METHOD("selu_normr", "z"), &MLPPActivation::selu_normr);
	ClassDB::bind_method(D_METHOD("selu_normv", "z"), &MLPPActivation::selu_normv);
	ClassDB::bind_method(D_METHOD("selu_normm", "z"), &MLPPActivation::selu_normm);

	ClassDB::bind_method(D_METHOD("selu_derivr", "z"), &MLPPActivation::selu_derivr);
	ClassDB::bind_method(D_METHOD("selu_derivv", "z"), &MLPPActivation::selu_derivv);
	ClassDB::bind_method(D_METHOD("selu_derivm", "z"), &MLPPActivation::selu_derivm);

	//GELU

	ClassDB::bind_method(D_METHOD("gelu_normr", "z"), &MLPPActivation::gelu_normr);
	ClassDB::bind_method(D_METHOD("gelu_normv", "z"), &MLPPActivation::gelu_normv);
	ClassDB::bind_method(D_METHOD("gelu_normm", "z"), &MLPPActivation::gelu_normm);

	ClassDB::bind_method(D_METHOD("gelu_derivr", "z"), &MLPPActivation::gelu_derivr);
	ClassDB::bind_method(D_METHOD("gelu_derivv", "z"), &MLPPActivation::gelu_derivv);
	ClassDB::bind_method(D_METHOD("gelu_derivm", "z"), &MLPPActivation::gelu_derivm);

	//SIGN

	ClassDB::bind_method(D_METHOD("sign_normr", "z"), &MLPPActivation::sign_normr);
	ClassDB::bind_method(D_METHOD("sign_normv", "z"), &MLPPActivation::sign_normv);
	ClassDB::bind_method(D_METHOD("sign_normm", "z"), &MLPPActivation::sign_normm);

	ClassDB::bind_method(D_METHOD("sign_derivr", "z"), &MLPPActivation::sign_derivr);
	ClassDB::bind_method(D_METHOD("sign_derivv", "z"), &MLPPActivation::sign_derivv);
	ClassDB::bind_method(D_METHOD("sign_derivm", "z"), &MLPPActivation::sign_derivm);

	//SINH

	ClassDB::bind_method(D_METHOD("sinh_normr", "z"), &MLPPActivation::sinh_normr);
	ClassDB::bind_method(D_METHOD("sinh_normv", "z"), &MLPPActivation::sinh_normv);
	ClassDB::bind_method(D_METHOD("sinh_normm", "z"), &MLPPActivation::sinh_normm);

	ClassDB::bind_method(D_METHOD("sinh_derivr", "z"), &MLPPActivation::sinh_derivr);
	ClassDB::bind_method(D_METHOD("sinh_derivv", "z"), &MLPPActivation::sinh_derivv);
	ClassDB::bind_method(D_METHOD("sinh_derivm", "z"), &MLPPActivation::sinh_derivm);

	//COSH

	ClassDB::bind_method(D_METHOD("cosh_normr", "z"), &MLPPActivation::cosh_normr);
	ClassDB::bind_method(D_METHOD("cosh_normv", "z"), &MLPPActivation::cosh_normv);
	ClassDB::bind_method(D_METHOD("cosh_normm", "z"), &MLPPActivation::cosh_normm);

	ClassDB::bind_method(D_METHOD("cosh_derivr", "z"), &MLPPActivation::cosh_derivr);
	ClassDB::bind_method(D_METHOD("cosh_derivv", "z"), &MLPPActivation::cosh_derivv);
	ClassDB::bind_method(D_METHOD("cosh_derivm", "z"), &MLPPActivation::cosh_derivm);

	//TANH

	ClassDB::bind_method(D_METHOD("tanh_normr", "z"), &MLPPActivation::tanh_normr);
	ClassDB::bind_method(D_METHOD("tanh_normv", "z"), &MLPPActivation::tanh_normv);
	ClassDB::bind_method(D_METHOD("tanh_normm", "z"), &MLPPActivation::tanh_normm);

	ClassDB::bind_method(D_METHOD("tanh_derivr", "z"), &MLPPActivation::tanh_derivr);
	ClassDB::bind_method(D_METHOD("tanh_derivv", "z"), &MLPPActivation::tanh_derivv);
	ClassDB::bind_method(D_METHOD("tanh_derivm", "z"), &MLPPActivation::tanh_derivm);

	//CSCH

	ClassDB::bind_method(D_METHOD("csch_normr", "z"), &MLPPActivation::csch_normr);
	ClassDB::bind_method(D_METHOD("csch_normv", "z"), &MLPPActivation::csch_normv);
	ClassDB::bind_method(D_METHOD("csch_normm", "z"), &MLPPActivation::csch_normm);

	ClassDB::bind_method(D_METHOD("csch_derivr", "z"), &MLPPActivation::csch_derivr);
	ClassDB::bind_method(D_METHOD("csch_derivv", "z"), &MLPPActivation::csch_derivv);
	ClassDB::bind_method(D_METHOD("csch_derivm", "z"), &MLPPActivation::csch_derivm);

	//SECH

	ClassDB::bind_method(D_METHOD("sech_normr", "z"), &MLPPActivation::sech_normr);
	ClassDB::bind_method(D_METHOD("sech_normv", "z"), &MLPPActivation::sech_normv);
	ClassDB::bind_method(D_METHOD("sech_normm", "z"), &MLPPActivation::sech_normm);

	ClassDB::bind_method(D_METHOD("sech_derivr", "z"), &MLPPActivation::sech_derivr);
	ClassDB::bind_method(D_METHOD("sech_derivv", "z"), &MLPPActivation::sech_derivv);
	ClassDB::bind_method(D_METHOD("sech_derivm", "z"), &MLPPActivation::sech_derivm);

	//COTH

	ClassDB::bind_method(D_METHOD("coth_normr", "z"), &MLPPActivation::coth_normr);
	ClassDB::bind_method(D_METHOD("coth_normv", "z"), &MLPPActivation::coth_normv);
	ClassDB::bind_method(D_METHOD("coth_normm", "z"), &MLPPActivation::coth_normm);

	ClassDB::bind_method(D_METHOD("coth_derivr", "z"), &MLPPActivation::coth_derivr);
	ClassDB::bind_method(D_METHOD("coth_derivv", "z"), &MLPPActivation::coth_derivv);
	ClassDB::bind_method(D_METHOD("coth_derivm", "z"), &MLPPActivation::coth_derivm);

	//ARSINH

	ClassDB::bind_method(D_METHOD("arsinh_normr", "z"), &MLPPActivation::arsinh_normr);
	ClassDB::bind_method(D_METHOD("arsinh_normv", "z"), &MLPPActivation::arsinh_normv);
	ClassDB::bind_method(D_METHOD("arsinh_normm", "z"), &MLPPActivation::arsinh_normm);

	ClassDB::bind_method(D_METHOD("arsinh_derivr", "z"), &MLPPActivation::arsinh_derivr);
	ClassDB::bind_method(D_METHOD("arsinh_derivv", "z"), &MLPPActivation::arsinh_derivv);
	ClassDB::bind_method(D_METHOD("arsinh_derivm", "z"), &MLPPActivation::arsinh_derivm);

	//ARCOSH

	ClassDB::bind_method(D_METHOD("arcosh_normr", "z"), &MLPPActivation::arcosh_normr);
	ClassDB::bind_method(D_METHOD("arcosh_normv", "z"), &MLPPActivation::arcosh_normv);
	ClassDB::bind_method(D_METHOD("arcosh_normm", "z"), &MLPPActivation::arcosh_normm);

	ClassDB::bind_method(D_METHOD("arcosh_derivr", "z"), &MLPPActivation::arcosh_derivr);
	ClassDB::bind_method(D_METHOD("arcosh_derivv", "z"), &MLPPActivation::arcosh_derivv);
	ClassDB::bind_method(D_METHOD("arcosh_derivm", "z"), &MLPPActivation::arcosh_derivm);

	//ARTANH

	ClassDB::bind_method(D_METHOD("artanh_normr", "z"), &MLPPActivation::artanh_normr);
	ClassDB::bind_method(D_METHOD("artanh_normv", "z"), &MLPPActivation::artanh_normv);
	ClassDB::bind_method(D_METHOD("artanh_normm", "z"), &MLPPActivation::artanh_normm);

	ClassDB::bind_method(D_METHOD("artanh_derivr", "z"), &MLPPActivation::artanh_derivr);
	ClassDB::bind_method(D_METHOD("artanh_derivv", "z"), &MLPPActivation::artanh_derivv);
	ClassDB::bind_method(D_METHOD("artanh_derivm", "z"), &MLPPActivation::artanh_derivm);

	//ARCSCH

	ClassDB::bind_method(D_METHOD("arcsch_normr", "z"), &MLPPActivation::arcsch_normr);
	ClassDB::bind_method(D_METHOD("arcsch_normv", "z"), &MLPPActivation::arcsch_normv);
	ClassDB::bind_method(D_METHOD("arcsch_normm", "z"), &MLPPActivation::arcsch_normm);

	ClassDB::bind_method(D_METHOD("arcsch_derivr", "z"), &MLPPActivation::arcsch_derivr);
	ClassDB::bind_method(D_METHOD("arcsch_derivv", "z"), &MLPPActivation::arcsch_derivv);
	ClassDB::bind_method(D_METHOD("arcsch_derivm", "z"), &MLPPActivation::arcsch_derivm);

	//ARSECH

	ClassDB::bind_method(D_METHOD("arsech_normr", "z"), &MLPPActivation::arsech_normr);
	ClassDB::bind_method(D_METHOD("arsech_normv", "z"), &MLPPActivation::arsech_normv);
	ClassDB::bind_method(D_METHOD("arsech_normm", "z"), &MLPPActivation::arsech_normm);

	ClassDB::bind_method(D_METHOD("arsech_derivr", "z"), &MLPPActivation::arsech_derivr);
	ClassDB::bind_method(D_METHOD("arsech_derivv", "z"), &MLPPActivation::arsech_derivv);
	ClassDB::bind_method(D_METHOD("arsech_derivm", "z"), &MLPPActivation::arsech_derivm);

	//ARCOTH

	ClassDB::bind_method(D_METHOD("arcoth_normr", "z"), &MLPPActivation::arcoth_normr);
	ClassDB::bind_method(D_METHOD("arcoth_normv", "z"), &MLPPActivation::arcoth_normv);
	ClassDB::bind_method(D_METHOD("arcoth_normm", "z"), &MLPPActivation::arcoth_normm);

	ClassDB::bind_method(D_METHOD("arcoth_derivr", "z"), &MLPPActivation::arcoth_derivr);
	ClassDB::bind_method(D_METHOD("arcoth_derivv", "z"), &MLPPActivation::arcoth_derivv);
	ClassDB::bind_method(D_METHOD("arcoth_derivm", "z"), &MLPPActivation::arcoth_derivm);

	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_LINEAR);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_SIGMOID);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_SWISH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_MISH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_SIN_C);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_SOFTMAX);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_SOFTPLUS);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_SOFTSIGN);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_ADJ_SOFTMAX);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_C_LOG_LOG);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_LOGIT);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_GAUSSIAN_CDF);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_RELU);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_GELU);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_SIGN);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_UNIT_STEP);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_SINH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_COSH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_TANH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_CSCH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_SECH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_COTH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_ARSINH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_ARCOSH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_ARTANH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_ARCSCH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_ARSECH);
	BIND_ENUM_CONSTANT(ACTIVATION_FUNCTION_ARCOTH);
}
