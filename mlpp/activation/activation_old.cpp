//
//  Activation.cpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "activation_old.h"
#include "../lin_alg/lin_alg.h"
#include <algorithm>
#include <cmath>
#include <iostream>

MLPPActivationOld::RealActivationFunctionPointer MLPPActivationOld::get_activation_function_ptr_real(const ActivationFunction func, const bool deriv) {
	if (deriv) {
		return get_activation_function_ptr_normal_real(func);
	} else {
		return get_activation_function_ptr_deriv_real(func);
	}
}
MLPPActivationOld::VectorActivationFunctionPointer MLPPActivationOld::get_activation_function_ptr_vector(const ActivationFunction func, const bool deriv) {
	if (deriv) {
		return get_activation_function_ptr_normal_vector(func);
	} else {
		return get_activation_function_ptr_deriv_vector(func);
	}
}
MLPPActivationOld::MatrixActivationFunctionPointer MLPPActivationOld::get_activation_function_ptr_matrix(const ActivationFunction func, const bool deriv) {
	if (deriv) {
		return get_activation_function_ptr_normal_matrix(func);
	} else {
		return get_activation_function_ptr_deriv_matrix(func);
	}
}

MLPPActivationOld::RealActivationFunctionPointer MLPPActivationOld::get_activation_function_ptr_normal_real(const ActivationFunction func) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return &MLPPActivationOld::linear_normr;
		case ACTIVATION_FUNCTION_SIGMOID:
			return &MLPPActivationOld::sigmoid_normr;
		case ACTIVATION_FUNCTION_SWISH:
			return &MLPPActivationOld::swish_normr;
		case ACTIVATION_FUNCTION_MISH:
			return &MLPPActivationOld::mish_normr;
		case ACTIVATION_FUNCTION_SIN_C:
			return &MLPPActivationOld::sinc_normr;
		case ACTIVATION_FUNCTION_SOFTMAX:
			return &MLPPActivationOld::softmax_normr;
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return &MLPPActivationOld::softplus_normr;
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return &MLPPActivationOld::softsign_normr;
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return &MLPPActivationOld::adj_softmax_normr;
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return &MLPPActivationOld::cloglog_normr;
		case ACTIVATION_FUNCTION_LOGIT:
			return &MLPPActivationOld::logit_normr;
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return &MLPPActivationOld::gaussian_cdf_normr;
		case ACTIVATION_FUNCTION_RELU:
			return &MLPPActivationOld::relu_normr;
		case ACTIVATION_FUNCTION_GELU:
			return &MLPPActivationOld::gelu_normr;
		case ACTIVATION_FUNCTION_SIGN:
			return &MLPPActivationOld::sign_normr;
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return &MLPPActivationOld::unit_step_normr;
		case ACTIVATION_FUNCTION_SINH:
			return &MLPPActivationOld::sinh_normr;
		case ACTIVATION_FUNCTION_COSH:
			return &MLPPActivationOld::cosh_normr;
		case ACTIVATION_FUNCTION_TANH:
			return &MLPPActivationOld::tanh_normr;
		case ACTIVATION_FUNCTION_CSCH:
			return &MLPPActivationOld::csch_normr;
		case ACTIVATION_FUNCTION_SECH:
			return &MLPPActivationOld::sech_normr;
		case ACTIVATION_FUNCTION_COTH:
			return &MLPPActivationOld::coth_normr;
		case ACTIVATION_FUNCTION_ARSINH:
			return &MLPPActivationOld::arsinh_normr;
		case ACTIVATION_FUNCTION_ARCOSH:
			return &MLPPActivationOld::arcosh_normr;
		case ACTIVATION_FUNCTION_ARTANH:
			return &MLPPActivationOld::artanh_normr;
		case ACTIVATION_FUNCTION_ARCSCH:
			return &MLPPActivationOld::arcsch_normr;
		case ACTIVATION_FUNCTION_ARSECH:
			return &MLPPActivationOld::arsech_normr;
		case ACTIVATION_FUNCTION_ARCOTH:
			return &MLPPActivationOld::arcoth_normr;
		default:
			return NULL;
	}
}
MLPPActivationOld::VectorActivationFunctionPointer MLPPActivationOld::get_activation_function_ptr_normal_vector(const ActivationFunction func) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return &MLPPActivationOld::linear_normv;
		case ACTIVATION_FUNCTION_SIGMOID:
			return &MLPPActivationOld::sigmoid_normv;
		case ACTIVATION_FUNCTION_SWISH:
			return &MLPPActivationOld::swish_normv;
		case ACTIVATION_FUNCTION_MISH:
			return &MLPPActivationOld::mish_normv;
		case ACTIVATION_FUNCTION_SIN_C:
			return &MLPPActivationOld::sinc_normv;
		case ACTIVATION_FUNCTION_SOFTMAX:
			return &MLPPActivationOld::softmax_normv;
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return &MLPPActivationOld::softplus_normv;
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return &MLPPActivationOld::softsign_normv;
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return &MLPPActivationOld::adj_softmax_normv;
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return &MLPPActivationOld::cloglog_normv;
		case ACTIVATION_FUNCTION_LOGIT:
			return &MLPPActivationOld::logit_normv;
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return &MLPPActivationOld::gaussian_cdf_normv;
		case ACTIVATION_FUNCTION_RELU:
			return &MLPPActivationOld::relu_normv;
		case ACTIVATION_FUNCTION_GELU:
			return &MLPPActivationOld::gelu_normv;
		case ACTIVATION_FUNCTION_SIGN:
			return &MLPPActivationOld::sign_normv;
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return &MLPPActivationOld::unit_step_normv;
		case ACTIVATION_FUNCTION_SINH:
			return &MLPPActivationOld::sinh_normv;
		case ACTIVATION_FUNCTION_COSH:
			return &MLPPActivationOld::cosh_normv;
		case ACTIVATION_FUNCTION_TANH:
			return &MLPPActivationOld::tanh_normv;
		case ACTIVATION_FUNCTION_CSCH:
			return &MLPPActivationOld::csch_normv;
		case ACTIVATION_FUNCTION_SECH:
			return &MLPPActivationOld::sech_normv;
		case ACTIVATION_FUNCTION_COTH:
			return &MLPPActivationOld::coth_normv;
		case ACTIVATION_FUNCTION_ARSINH:
			return &MLPPActivationOld::arsinh_normv;
		case ACTIVATION_FUNCTION_ARCOSH:
			return &MLPPActivationOld::arcosh_normv;
		case ACTIVATION_FUNCTION_ARTANH:
			return &MLPPActivationOld::artanh_normv;
		case ACTIVATION_FUNCTION_ARCSCH:
			return &MLPPActivationOld::arcsch_normv;
		case ACTIVATION_FUNCTION_ARSECH:
			return &MLPPActivationOld::arsech_normv;
		case ACTIVATION_FUNCTION_ARCOTH:
			return &MLPPActivationOld::arcoth_normv;
		default:
			return NULL;
	}
}
MLPPActivationOld::MatrixActivationFunctionPointer MLPPActivationOld::get_activation_function_ptr_normal_matrix(const ActivationFunction func) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return &MLPPActivationOld::linear_normm;
		case ACTIVATION_FUNCTION_SIGMOID:
			return &MLPPActivationOld::sigmoid_normm;
		case ACTIVATION_FUNCTION_SWISH:
			return &MLPPActivationOld::swish_normm;
		case ACTIVATION_FUNCTION_MISH:
			return &MLPPActivationOld::mish_normm;
		case ACTIVATION_FUNCTION_SIN_C:
			return &MLPPActivationOld::sinc_normm;
		case ACTIVATION_FUNCTION_SOFTMAX:
			return &MLPPActivationOld::softmax_normm;
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return &MLPPActivationOld::softplus_normm;
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return &MLPPActivationOld::softsign_normm;
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return &MLPPActivationOld::adj_softmax_normm;
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return &MLPPActivationOld::cloglog_normm;
		case ACTIVATION_FUNCTION_LOGIT:
			return &MLPPActivationOld::logit_normm;
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return &MLPPActivationOld::gaussian_cdf_normm;
		case ACTIVATION_FUNCTION_RELU:
			return &MLPPActivationOld::relu_normm;
		case ACTIVATION_FUNCTION_GELU:
			return &MLPPActivationOld::gelu_normm;
		case ACTIVATION_FUNCTION_SIGN:
			return &MLPPActivationOld::sign_normm;
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return &MLPPActivationOld::unit_step_normm;
		case ACTIVATION_FUNCTION_SINH:
			return &MLPPActivationOld::sinh_normm;
		case ACTIVATION_FUNCTION_COSH:
			return &MLPPActivationOld::cosh_normm;
		case ACTIVATION_FUNCTION_TANH:
			return &MLPPActivationOld::tanh_normm;
		case ACTIVATION_FUNCTION_CSCH:
			return &MLPPActivationOld::csch_normm;
		case ACTIVATION_FUNCTION_SECH:
			return &MLPPActivationOld::sech_normm;
		case ACTIVATION_FUNCTION_COTH:
			return &MLPPActivationOld::coth_normm;
		case ACTIVATION_FUNCTION_ARSINH:
			return &MLPPActivationOld::arsinh_normm;
		case ACTIVATION_FUNCTION_ARCOSH:
			return &MLPPActivationOld::arcosh_normm;
		case ACTIVATION_FUNCTION_ARTANH:
			return &MLPPActivationOld::artanh_normm;
		case ACTIVATION_FUNCTION_ARCSCH:
			return &MLPPActivationOld::arcsch_normm;
		case ACTIVATION_FUNCTION_ARSECH:
			return &MLPPActivationOld::arsech_normm;
		case ACTIVATION_FUNCTION_ARCOTH:
			return &MLPPActivationOld::arcoth_normm;
		default:
			return NULL;
	}
}

MLPPActivationOld::RealActivationFunctionPointer MLPPActivationOld::get_activation_function_ptr_deriv_real(const ActivationFunction func) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return &MLPPActivationOld::linear_normr;
		case ACTIVATION_FUNCTION_SIGMOID:
			return &MLPPActivationOld::sigmoid_normr;
		case ACTIVATION_FUNCTION_SWISH:
			return &MLPPActivationOld::swish_normr;
		case ACTIVATION_FUNCTION_MISH:
			return &MLPPActivationOld::mish_normr;
		case ACTIVATION_FUNCTION_SIN_C:
			return &MLPPActivationOld::sinc_normr;
		case ACTIVATION_FUNCTION_SOFTMAX:
			return &MLPPActivationOld::softmax_normr;
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return &MLPPActivationOld::softplus_normr;
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return &MLPPActivationOld::softsign_normr;
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return &MLPPActivationOld::adj_softmax_normr;
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return &MLPPActivationOld::cloglog_normr;
		case ACTIVATION_FUNCTION_LOGIT:
			return &MLPPActivationOld::logit_normr;
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return &MLPPActivationOld::gaussian_cdf_normr;
		case ACTIVATION_FUNCTION_RELU:
			return &MLPPActivationOld::relu_normr;
		case ACTIVATION_FUNCTION_GELU:
			return &MLPPActivationOld::gelu_normr;
		case ACTIVATION_FUNCTION_SIGN:
			return &MLPPActivationOld::sign_normr;
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return &MLPPActivationOld::unit_step_normr;
		case ACTIVATION_FUNCTION_SINH:
			return &MLPPActivationOld::sinh_normr;
		case ACTIVATION_FUNCTION_COSH:
			return &MLPPActivationOld::cosh_normr;
		case ACTIVATION_FUNCTION_TANH:
			return &MLPPActivationOld::tanh_normr;
		case ACTIVATION_FUNCTION_CSCH:
			return &MLPPActivationOld::csch_normr;
		case ACTIVATION_FUNCTION_SECH:
			return &MLPPActivationOld::sech_normr;
		case ACTIVATION_FUNCTION_COTH:
			return &MLPPActivationOld::coth_normr;
		case ACTIVATION_FUNCTION_ARSINH:
			return &MLPPActivationOld::arsinh_normr;
		case ACTIVATION_FUNCTION_ARCOSH:
			return &MLPPActivationOld::arcosh_normr;
		case ACTIVATION_FUNCTION_ARTANH:
			return &MLPPActivationOld::artanh_normr;
		case ACTIVATION_FUNCTION_ARCSCH:
			return &MLPPActivationOld::arcsch_normr;
		case ACTIVATION_FUNCTION_ARSECH:
			return &MLPPActivationOld::arsech_normr;
		case ACTIVATION_FUNCTION_ARCOTH:
			return &MLPPActivationOld::arcoth_normr;
		default:
			return NULL;
	}
}
MLPPActivationOld::VectorActivationFunctionPointer MLPPActivationOld::get_activation_function_ptr_deriv_vector(const ActivationFunction func) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return &MLPPActivationOld::linear_derivv;
		case ACTIVATION_FUNCTION_SIGMOID:
			return &MLPPActivationOld::sigmoid_derivv;
		case ACTIVATION_FUNCTION_SWISH:
			return &MLPPActivationOld::swish_derivv;
		case ACTIVATION_FUNCTION_MISH:
			return &MLPPActivationOld::mish_derivv;
		case ACTIVATION_FUNCTION_SIN_C:
			return &MLPPActivationOld::sinc_derivv;
		case ACTIVATION_FUNCTION_SOFTMAX:
			return &MLPPActivationOld::softmax_derivv;
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return &MLPPActivationOld::softplus_derivv;
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return &MLPPActivationOld::softsign_derivv;
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return &MLPPActivationOld::adj_softmax_derivv;
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return &MLPPActivationOld::cloglog_derivv;
		case ACTIVATION_FUNCTION_LOGIT:
			return &MLPPActivationOld::logit_derivv;
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return &MLPPActivationOld::gaussian_cdf_derivv;
		case ACTIVATION_FUNCTION_RELU:
			return &MLPPActivationOld::relu_derivv;
		case ACTIVATION_FUNCTION_GELU:
			return &MLPPActivationOld::gelu_derivv;
		case ACTIVATION_FUNCTION_SIGN:
			return &MLPPActivationOld::sign_derivv;
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return &MLPPActivationOld::unit_step_derivv;
		case ACTIVATION_FUNCTION_SINH:
			return &MLPPActivationOld::sinh_derivv;
		case ACTIVATION_FUNCTION_COSH:
			return &MLPPActivationOld::cosh_derivv;
		case ACTIVATION_FUNCTION_TANH:
			return &MLPPActivationOld::tanh_derivv;
		case ACTIVATION_FUNCTION_CSCH:
			return &MLPPActivationOld::csch_derivv;
		case ACTIVATION_FUNCTION_SECH:
			return &MLPPActivationOld::sech_derivv;
		case ACTIVATION_FUNCTION_COTH:
			return &MLPPActivationOld::coth_derivv;
		case ACTIVATION_FUNCTION_ARSINH:
			return &MLPPActivationOld::arsinh_derivv;
		case ACTIVATION_FUNCTION_ARCOSH:
			return &MLPPActivationOld::arcosh_derivv;
		case ACTIVATION_FUNCTION_ARTANH:
			return &MLPPActivationOld::artanh_derivv;
		case ACTIVATION_FUNCTION_ARCSCH:
			return &MLPPActivationOld::arcsch_derivv;
		case ACTIVATION_FUNCTION_ARSECH:
			return &MLPPActivationOld::arsech_derivv;
		case ACTIVATION_FUNCTION_ARCOTH:
			return &MLPPActivationOld::arcoth_derivv;
		default:
			return NULL;
	}
}
MLPPActivationOld::MatrixActivationFunctionPointer MLPPActivationOld::get_activation_function_ptr_deriv_matrix(const ActivationFunction func) {
	switch (func) {
		case ACTIVATION_FUNCTION_LINEAR:
			return &MLPPActivationOld::linear_derivm;
		case ACTIVATION_FUNCTION_SIGMOID:
			return &MLPPActivationOld::sigmoid_derivm;
		case ACTIVATION_FUNCTION_SWISH:
			return &MLPPActivationOld::swish_derivm;
		case ACTIVATION_FUNCTION_MISH:
			return &MLPPActivationOld::mish_derivm;
		case ACTIVATION_FUNCTION_SIN_C:
			return &MLPPActivationOld::sinc_derivm;
		case ACTIVATION_FUNCTION_SOFTMAX:
			return &MLPPActivationOld::softmax_derivm;
		case ACTIVATION_FUNCTION_SOFTPLUS:
			return &MLPPActivationOld::softplus_derivm;
		case ACTIVATION_FUNCTION_SOFTSIGN:
			return &MLPPActivationOld::softsign_derivm;
		case ACTIVATION_FUNCTION_ADJ_SOFTMAX:
			return &MLPPActivationOld::adj_softmax_derivm;
		case ACTIVATION_FUNCTION_C_LOG_LOG:
			return &MLPPActivationOld::cloglog_derivm;
		case ACTIVATION_FUNCTION_LOGIT:
			return &MLPPActivationOld::logit_derivm;
		case ACTIVATION_FUNCTION_GAUSSIAN_CDF:
			return &MLPPActivationOld::gaussian_cdf_derivm;
		case ACTIVATION_FUNCTION_RELU:
			return &MLPPActivationOld::relu_derivm;
		case ACTIVATION_FUNCTION_GELU:
			return &MLPPActivationOld::gelu_derivm;
		case ACTIVATION_FUNCTION_SIGN:
			return &MLPPActivationOld::sign_derivm;
		case ACTIVATION_FUNCTION_UNIT_STEP:
			return &MLPPActivationOld::unit_step_derivm;
		case ACTIVATION_FUNCTION_SINH:
			return &MLPPActivationOld::sinh_derivm;
		case ACTIVATION_FUNCTION_COSH:
			return &MLPPActivationOld::cosh_derivm;
		case ACTIVATION_FUNCTION_TANH:
			return &MLPPActivationOld::tanh_derivm;
		case ACTIVATION_FUNCTION_CSCH:
			return &MLPPActivationOld::csch_derivm;
		case ACTIVATION_FUNCTION_SECH:
			return &MLPPActivationOld::sech_derivm;
		case ACTIVATION_FUNCTION_COTH:
			return &MLPPActivationOld::coth_derivm;
		case ACTIVATION_FUNCTION_ARSINH:
			return &MLPPActivationOld::arsinh_derivm;
		case ACTIVATION_FUNCTION_ARCOSH:
			return &MLPPActivationOld::arcosh_derivm;
		case ACTIVATION_FUNCTION_ARTANH:
			return &MLPPActivationOld::artanh_derivm;
		case ACTIVATION_FUNCTION_ARCSCH:
			return &MLPPActivationOld::arcsch_derivm;
		case ACTIVATION_FUNCTION_ARSECH:
			return &MLPPActivationOld::arsech_derivm;
		case ACTIVATION_FUNCTION_ARCOTH:
			return &MLPPActivationOld::arcoth_derivm;
		default:
			return NULL;
	}
}

real_t MLPPActivationOld::run_activation_real(const ActivationFunction func, const real_t z, const bool deriv) {
	if (deriv) {
		return run_activation_norm_real(func, z);
	} else {
		return run_activation_deriv_real(func, z);
	}
}
Ref<MLPPVector> MLPPActivationOld::run_activation_vector(const ActivationFunction func, const Ref<MLPPVector> &z, const bool deriv) {
	if (deriv) {
		return run_activation_norm_vector(func, z);
	} else {
		return run_activation_deriv_vector(func, z);
	}
}
Ref<MLPPMatrix> MLPPActivationOld::run_activation_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z, const bool deriv) {
	if (deriv) {
		return run_activation_norm_matrix(func, z);
	} else {
		return run_activation_deriv_matrix(func, z);
	}
}

real_t MLPPActivationOld::run_activation_norm_real(const ActivationFunction func, const real_t z) {
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
Ref<MLPPVector> MLPPActivationOld::run_activation_norm_vector(const ActivationFunction func, const Ref<MLPPVector> &z) {
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
Ref<MLPPMatrix> MLPPActivationOld::run_activation_norm_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z) {
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

real_t MLPPActivationOld::run_activation_deriv_real(const ActivationFunction func, const real_t z) {
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
Ref<MLPPVector> MLPPActivationOld::run_activation_deriv_vector(const ActivationFunction func, const Ref<MLPPVector> &z) {
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
Ref<MLPPMatrix> MLPPActivationOld::run_activation_deriv_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z) {
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

Ref<MLPPVector> MLPPActivationOld::activationr(const Ref<MLPPVector> &z, real_t (*function)(real_t)) {
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
real_t MLPPActivationOld::linear_normr(real_t z) {
	return z;
}
Ref<MLPPVector> MLPPActivationOld::linear_normv(const Ref<MLPPVector> &z) {
	return z->duplicate();
}
Ref<MLPPMatrix> MLPPActivationOld::linear_normm(const Ref<MLPPMatrix> &z) {
	return z->duplicate();
}

real_t MLPPActivationOld::linear_derivr(real_t z) {
	return 1;
}
Ref<MLPPVector> MLPPActivationOld::linear_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;
	return alg.onevecv(z->size());
}
Ref<MLPPMatrix> MLPPActivationOld::linear_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;
	return alg.onematm(z->size().x, z->size().y);
}

//SIGMOID
real_t MLPPActivationOld::sigmoid_normr(real_t z) {
	return 1 / (1 + exp(-z));
}
Ref<MLPPVector> MLPPActivationOld::sigmoid_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;
	return alg.element_wise_division(alg.onevecv(z->size()), alg.additionnv(alg.onevecv(z->size()), alg.expv(alg.scalar_multiplynv(-1, z))));
}
Ref<MLPPMatrix> MLPPActivationOld::sigmoid_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;
	return alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y), alg.additionm(alg.onematm(z->size().x, z->size().y), alg.expm(alg.scalar_multiplym(-1, z))));
}

real_t MLPPActivationOld::sigmoid_derivr(real_t z) {
	real_t sig_norm = sigmoid_normr(z);

	return sig_norm * (1 - sig_norm);
}

Ref<MLPPVector> MLPPActivationOld::sigmoid_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	Ref<MLPPVector> sig_norm = sigmoid_normv(z);

	return alg.subtractionnv(sig_norm, alg.hadamard_productnv(sig_norm, sig_norm));
}
Ref<MLPPMatrix> MLPPActivationOld::sigmoid_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	Ref<MLPPMatrix> sig_norm = sigmoid_normm(z);

	return alg.subtractionm(sig_norm, alg.hadamard_productm(sig_norm, sig_norm));
}

//SOFTMAX

real_t MLPPActivationOld::softmax_normr(real_t z) {
	return z;
}
Ref<MLPPVector> MLPPActivationOld::softmax_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	int z_size = z->size();

	Ref<MLPPVector> a;
	a.instance();
	a->resize(z_size);

	Ref<MLPPVector> exp_z = alg.expv(z);
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
Ref<MLPPMatrix> MLPPActivationOld::softmax_normm(const Ref<MLPPMatrix> &z) {
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

real_t MLPPActivationOld::softmax_derivr(real_t z) {
	return z;
}
Ref<MLPPVector> MLPPActivationOld::softmax_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	int z_size = z->size();

	Ref<MLPPVector> a;
	a.instance();
	a->resize(z_size);

	Ref<MLPPVector> exp_z = alg.expv(z);
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
Ref<MLPPMatrix> MLPPActivationOld::softmax_derivm(const Ref<MLPPMatrix> &z) {
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

real_t MLPPActivationOld::adj_softmax_normr(real_t z) {
	return 0;
}

Ref<MLPPVector> MLPPActivationOld::adj_softmax_normv(const Ref<MLPPVector> &z) {
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
Ref<MLPPMatrix> MLPPActivationOld::adj_softmax_normm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> n = z->duplicate();

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

real_t MLPPActivationOld::adj_softmax_derivr(real_t z) {
	return 0;
}

Ref<MLPPVector> MLPPActivationOld::adj_softmax_derivv(const Ref<MLPPVector> &z) {
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
Ref<MLPPMatrix> MLPPActivationOld::adj_softmax_derivm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> n = z->duplicate();

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

Ref<MLPPMatrix> MLPPActivationOld::softmax_deriv_normv(const Ref<MLPPVector> &z) {
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
				deriv->set_element(i, j, a_ptr[i] * (1 - a_ptr[i]));
			} else {
				deriv->set_element(i, j, -a_ptr[i] * a_ptr[j]);
			}
		}
	}

	return deriv;
}
Vector<Ref<MLPPMatrix>> MLPPActivationOld::softmax_deriv_normm(const Ref<MLPPMatrix> &z) {
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

Ref<MLPPMatrix> MLPPActivationOld::softmax_deriv_derivv(const Ref<MLPPVector> &z) {
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
				deriv->set_element(i, j, a_ptr[i] * (1 - a_ptr[i]));
			} else {
				deriv->set_element(i, j, -a_ptr[i] * a_ptr[j]);
			}
		}
	}

	return deriv;
}
Vector<Ref<MLPPMatrix>> MLPPActivationOld::softmax_deriv_derivm(const Ref<MLPPMatrix> &z) {
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

real_t MLPPActivationOld::softplus_normr(real_t z) {
	return std::log(1 + exp(z));
}
Ref<MLPPVector> MLPPActivationOld::softplus_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.logv(alg.additionnv(alg.onevecv(z->size()), alg.expv(z)));
}
Ref<MLPPMatrix> MLPPActivationOld::softplus_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.logv(alg.additionnv(alg.onematm(z->size().x, z->size().y), alg.expv(z)));
}

real_t MLPPActivationOld::softplus_derivr(real_t z) {
	return sigmoid_normr(z);
}
Ref<MLPPVector> MLPPActivationOld::softplus_derivv(const Ref<MLPPVector> &z) {
	return sigmoid_normv(z);
}
Ref<MLPPMatrix> MLPPActivationOld::softplus_derivm(const Ref<MLPPMatrix> &z) {
	return sigmoid_normm(z);
}

//SOFTSIGN

real_t MLPPActivationOld::softsign_normr(real_t z) {
	return z / (1 + abs(z));
}
Ref<MLPPVector> MLPPActivationOld::softsign_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(z, alg.additionnv(alg.onevecv(z->size()), alg.absv(z)));
}
Ref<MLPPMatrix> MLPPActivationOld::softsign_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(z, alg.additionnv(alg.onematm(z->size().x, z->size().y), alg.absm(z)));
}

real_t MLPPActivationOld::softsign_derivr(real_t z) {
	return 1 / ((1 + abs(z)) * (1 + abs(z)));
}
Ref<MLPPVector> MLPPActivationOld::softsign_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(alg.onevecv(z->size()), alg.exponentiatev(alg.additionnv(alg.onevecv(z->size()), alg.absv(z)), 2));
}
Ref<MLPPMatrix> MLPPActivationOld::softsign_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y), alg.exponentiatev(alg.additionm(alg.onematm(z->size().x, z->size().y), alg.absm(z)), 2));
}

//GAUSSIANCDF

real_t MLPPActivationOld::gaussian_cdf_normr(real_t z) {
	return 0.5 * (1 + erf(z / sqrt(2)));
}
Ref<MLPPVector> MLPPActivationOld::gaussian_cdf_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(0.5, alg.additionnv(alg.onevecv(z->size()), alg.erfv(alg.scalar_multiplynv(1 / sqrt(2), z))));
}

Ref<MLPPMatrix> MLPPActivationOld::gaussian_cdf_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplym(0.5, alg.additionm(alg.onematm(z->size().x, z->size().y), alg.erfm(alg.scalar_multiplym(1 / sqrt(2), z))));
}

real_t MLPPActivationOld::gaussian_cdf_derivr(real_t z) {
	return (1 / sqrt(2 * M_PI)) * exp(-z * z / 2);
}
Ref<MLPPVector> MLPPActivationOld::gaussian_cdf_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(1 / Math::sqrt(2 * M_PI), alg.expv(alg.scalar_multiplynv(-1 / 2.0, alg.hadamard_productnv(z, z))));
}

Ref<MLPPMatrix> MLPPActivationOld::gaussian_cdf_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplym(1 / Math::sqrt(2 * M_PI), alg.expm(alg.scalar_multiplym(-1 / 2.0, alg.hadamard_productm(z, z))));
}

//CLOGLOG

real_t MLPPActivationOld::cloglog_normr(real_t z) {
	return 1 - exp(-exp(z));
}
Ref<MLPPVector> MLPPActivationOld::cloglog_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(-1, alg.scalar_addnv(-1, alg.expv(alg.scalar_multiplynv(-1, alg.expv(z)))));
}

Ref<MLPPMatrix> MLPPActivationOld::cloglog_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplym(-1, alg.scalar_addm(-1, alg.expm(alg.scalar_multiplym(-1, alg.expm(z)))));
}

real_t MLPPActivationOld::cloglog_derivr(real_t z) {
	return exp(z - exp(z));
}
Ref<MLPPVector> MLPPActivationOld::cloglog_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.expv(alg.scalar_multiplynv(-1, alg.expv(z)));
}

Ref<MLPPMatrix> MLPPActivationOld::cloglog_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.expm(alg.scalar_multiplym(-1, alg.expm(z)));
}

//LOGIT

real_t MLPPActivationOld::logit_normr(real_t z) {
	return std::log(z / (1 - z));
}
Ref<MLPPVector> MLPPActivationOld::logit_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.logv(alg.element_wise_division(z, alg.subtractionnv(alg.onevecv(z->size()), z)));
}
Ref<MLPPMatrix> MLPPActivationOld::logit_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.logm(alg.element_wise_divisionm(z, alg.subtractionm(alg.onematm(z->size().x, z->size().y), z)));
}

real_t MLPPActivationOld::logit_derivr(real_t z) {
	return 1 / z - 1 / (z - 1);
}
Ref<MLPPVector> MLPPActivationOld::logit_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.subtractionnv(
			alg.element_wise_division(alg.onevecv(z->size()), z),
			alg.element_wise_division(alg.onevecv(z->size()), alg.subtractionnv(z, alg.onevecv(z->size()))));
}
Ref<MLPPMatrix> MLPPActivationOld::logit_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.subtractionm(
			alg.element_wise_divisionm(
					alg.onematm(z->size().x, z->size().y), z),
			alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y),
					alg.subtractionm(z, alg.onematm(z->size().x, z->size().y))));
}

//UNITSTEP

real_t MLPPActivationOld::unit_step_normr(real_t z) {
	return z < 0 ? 0 : 1;
}
Ref<MLPPVector> MLPPActivationOld::unit_step_normv(const Ref<MLPPVector> &z) {
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
Ref<MLPPMatrix> MLPPActivationOld::unit_step_normm(const Ref<MLPPMatrix> &z) {
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

real_t MLPPActivationOld::unit_step_derivr(real_t z) {
	return 0;
}
Ref<MLPPVector> MLPPActivationOld::unit_step_derivv(const Ref<MLPPVector> &z) {
	Ref<MLPPVector> a;
	a.instance();
	a->resize(z->size());
	a->fill(0);

	return a;
}
Ref<MLPPMatrix> MLPPActivationOld::unit_step_derivm(const Ref<MLPPMatrix> &z) {
	Ref<MLPPMatrix> a;
	a.instance();
	a->resize(z->size());
	a->fill(0);

	return a;
}

//SWISH

real_t MLPPActivationOld::swish_normr(real_t z) {
	return z * sigmoid_normr(z);
}
Ref<MLPPVector> MLPPActivationOld::swish_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnv(z, sigmoid_normv(z));
}
Ref<MLPPMatrix> MLPPActivationOld::swish_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnv(z, sigmoid_normm(z));
}

real_t MLPPActivationOld::swish_derivr(real_t z) {
	return swish_normr(z) + sigmoid_normr(z) * (1 - swish_normr(z));
}
Ref<MLPPVector> MLPPActivationOld::swish_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.additionnv(swish_normv(z), alg.subtractionnv(sigmoid_normv(z), alg.hadamard_productnv(sigmoid_normv(z), swish_normv(z))));
}
Ref<MLPPMatrix> MLPPActivationOld::swish_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.additionnv(swish_normm(z), alg.subtractionnv(sigmoid_normm(z), alg.hadamard_productm(sigmoid_normm(z), swish_normm(z))));
}

//MISH

real_t MLPPActivationOld::mish_normr(real_t z) {
	return z * tanh(softplus(z));
}
Ref<MLPPVector> MLPPActivationOld::mish_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnv(z, tanh_normv(softplus_normv(z)));
}
Ref<MLPPMatrix> MLPPActivationOld::mish_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productm(z, tanh_normm(softplus_normm(z)));
}

real_t MLPPActivationOld::mish_derivr(real_t z) {
	return sech(softplus_normr(z)) * sech(softplus_normr(z)) * z * sigmoid_normr(z) + mish_normr(z) / z;
}
Ref<MLPPVector> MLPPActivationOld::mish_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.additionnv(
			alg.hadamard_productnv(
					alg.hadamard_productnv(
							alg.hadamard_productnv(
									sech_normv(softplus_normv(z)), sech_normv(softplus_normv(z))),
							z),
					sigmoid_normv(z)),
			alg.element_wise_division(mish_normv(z), z));
}
Ref<MLPPMatrix> MLPPActivationOld::mish_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.additionnv(
			alg.hadamard_productm(
					alg.hadamard_productm(
							alg.hadamard_productm(
									sech_normm(softplus_normm(z)), sech_normm(softplus_normm(z))),
							z),
					sigmoid_normm(z)),
			alg.element_wise_divisionm(mish_normm(z), z));
}

//SINC

real_t MLPPActivationOld::sinc_normr(real_t z) {
	return std::sin(z) / z;
}
Ref<MLPPVector> MLPPActivationOld::sinc_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(alg.sinv(z), z);
}
Ref<MLPPMatrix> MLPPActivationOld::sinc_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(alg.sinm(z), z);
}

real_t MLPPActivationOld::sinc_derivr(real_t z) {
	return (z * std::cos(z) - std::sin(z)) / (z * z);
}
Ref<MLPPVector> MLPPActivationOld::sinc_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(alg.subtractionnv(alg.hadamard_productnv(z, alg.cosv(z)), alg.sinv(z)), alg.hadamard_productnv(z, z));
}
Ref<MLPPMatrix> MLPPActivationOld::sinc_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(alg.subtractionm(alg.hadamard_productm(z, alg.cosm(z)), alg.sinm(z)), alg.hadamard_productm(z, z));
}

//RELU

real_t MLPPActivationOld::relu_normr(real_t z) {
	return fmax(0, z);
}
Ref<MLPPVector> MLPPActivationOld::relu_normv(const Ref<MLPPVector> &z) {
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
Ref<MLPPMatrix> MLPPActivationOld::relu_normm(const Ref<MLPPMatrix> &z) {
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

real_t MLPPActivationOld::relu_derivr(real_t z) {
	if (z <= 0) {
		return 0;
	} else {
		return 1;
	}
}
Ref<MLPPVector> MLPPActivationOld::relu_derivv(const Ref<MLPPVector> &z) {
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
Ref<MLPPMatrix> MLPPActivationOld::relu_derivm(const Ref<MLPPMatrix> &z) {
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

real_t MLPPActivationOld::leaky_relu_normr(real_t z, real_t c) {
	return fmax(c * z, z);
}
Ref<MLPPVector> MLPPActivationOld::leaky_relu_normv(const Ref<MLPPVector> &z, real_t c) {
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
Ref<MLPPMatrix> MLPPActivationOld::leaky_relu_normm(const Ref<MLPPMatrix> &z, real_t c) {
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

real_t MLPPActivationOld::leaky_relu_derivr(real_t z, real_t c) {
	if (z <= 0) {
		return c;
	} else {
		return 1;
	}
}
Ref<MLPPVector> MLPPActivationOld::leaky_relu_derivv(const Ref<MLPPVector> &z, real_t c) {
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
Ref<MLPPMatrix> MLPPActivationOld::leaky_relu_derivm(const Ref<MLPPMatrix> &z, real_t c) {
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

real_t MLPPActivationOld::elu_normr(real_t z, real_t c) {
	if (z >= 0) {
		return z;
	} else {
		return c * (exp(z) - 1);
	}
}
Ref<MLPPVector> MLPPActivationOld::elu_normv(const Ref<MLPPVector> &z, real_t c) {
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
Ref<MLPPMatrix> MLPPActivationOld::elu_normm(const Ref<MLPPMatrix> &z, real_t c) {
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

real_t MLPPActivationOld::elu_derivr(real_t z, real_t c) {
	if (z <= 0) {
		return c * exp(z);
	} else {
		return 1;
	}
}
Ref<MLPPVector> MLPPActivationOld::elu_derivv(const Ref<MLPPVector> &z, real_t c) {
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
Ref<MLPPMatrix> MLPPActivationOld::elu_derivm(const Ref<MLPPMatrix> &z, real_t c) {
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

real_t MLPPActivationOld::selu_normr(real_t z, real_t lambda, real_t c) {
	return lambda * ELU(z, c);
}
Ref<MLPPVector> MLPPActivationOld::selu_normv(const Ref<MLPPVector> &z, real_t lambda, real_t c) {
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
Ref<MLPPMatrix> MLPPActivationOld::selu_normm(const Ref<MLPPMatrix> &z, real_t lambda, real_t c) {
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

real_t MLPPActivationOld::selu_derivr(real_t z, real_t lambda, real_t c) {
	return elu_derivr(z, c);
}
Ref<MLPPVector> MLPPActivationOld::selu_derivv(const Ref<MLPPVector> &z, real_t lambda, real_t c) {
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
Ref<MLPPMatrix> MLPPActivationOld::selu_derivm(const Ref<MLPPMatrix> &z, real_t lambda, real_t c) {
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

real_t MLPPActivationOld::gelu_normr(real_t z) {
	return 0.5 * z * (1 + tanh(sqrt(2 / M_PI) * (z + 0.044715 * Math::pow(z, 3))));
}
Ref<MLPPVector> MLPPActivationOld::gelu_normv(const Ref<MLPPVector> &z) {
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
Ref<MLPPMatrix> MLPPActivationOld::gelu_normm(const Ref<MLPPMatrix> &z) {
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

real_t MLPPActivationOld::gelu_derivr(real_t z) {
	return 0.5 * tanh(0.0356774 * std::pow(z, 3) + 0.797885 * z) + (0.0535161 * std::pow(z, 3) + 0.398942 * z) * std::pow(sech(0.0356774 * std::pow(z, 3) + 0.797885 * z), 2) + 0.5;
}
Ref<MLPPVector> MLPPActivationOld::gelu_derivv(const Ref<MLPPVector> &z) {
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
Ref<MLPPMatrix> MLPPActivationOld::gelu_derivm(const Ref<MLPPMatrix> &z) {
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

real_t MLPPActivationOld::sign_normr(real_t z) {
	if (z < 0) {
		return -1;
	} else if (z == 0) {
		return 0;
	} else {
		return 1;
	}
}
Ref<MLPPVector> MLPPActivationOld::sign_normv(const Ref<MLPPVector> &z) {
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
Ref<MLPPMatrix> MLPPActivationOld::sign_normm(const Ref<MLPPMatrix> &z) {
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

real_t MLPPActivationOld::sign_derivr(real_t z) {
	return 0;
}
Ref<MLPPVector> MLPPActivationOld::sign_derivv(const Ref<MLPPVector> &z) {
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
Ref<MLPPMatrix> MLPPActivationOld::sign_derivm(const Ref<MLPPMatrix> &z) {
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

real_t MLPPActivationOld::sinh_normr(real_t z) {
	return 0.5 * (Math::exp(z) - Math::exp(-z));
}
Ref<MLPPVector> MLPPActivationOld::sinh_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;
	return alg.scalar_multiplynv(0.5, alg.subtractionnv(alg.expv(z), alg.expv(alg.scalar_multiplynv(-1, z))));
}
Ref<MLPPMatrix> MLPPActivationOld::sinh_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;
	return alg.scalar_multiplym(0.5, alg.subtractionm(alg.expm(z), alg.expm(alg.scalar_multiplym(-1, z))));
}

real_t MLPPActivationOld::sinh_derivr(real_t z) {
	return cosh_normr(z);
}
Ref<MLPPVector> MLPPActivationOld::sinh_derivv(const Ref<MLPPVector> &z) {
	return cosh_normv(z);
}
Ref<MLPPMatrix> MLPPActivationOld::sinh_derivm(const Ref<MLPPMatrix> &z) {
	return cosh_normm(z);
}

//COSH

real_t MLPPActivationOld::cosh_normr(real_t z) {
	return 0.5 * (Math::exp(z) + Math::exp(-z));
}
Ref<MLPPVector> MLPPActivationOld::cosh_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;
	return alg.scalar_multiplynv(0.5, alg.additionnv(alg.expv(z), alg.expv(alg.scalar_multiplynv(-1, z))));
}
Ref<MLPPMatrix> MLPPActivationOld::cosh_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;
	return alg.scalar_multiplym(0.5, alg.additionnv(alg.expm(z), alg.expm(alg.scalar_multiplym(-1, z))));
}

real_t MLPPActivationOld::cosh_derivr(real_t z) {
	return sinh_normr(z);
}
Ref<MLPPVector> MLPPActivationOld::cosh_derivv(const Ref<MLPPVector> &z) {
	return sinh_normv(z);
}
Ref<MLPPMatrix> MLPPActivationOld::cosh_derivm(const Ref<MLPPMatrix> &z) {
	return sinh_normm(z);
}

//TANH

real_t MLPPActivationOld::tanh_normr(real_t z) {
	return (Math::exp(z) - Math::exp(-z)) / (Math::exp(z) + Math::exp(-z));
}
Ref<MLPPVector> MLPPActivationOld::tanh_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(alg.subtractionnv(alg.expv(z), alg.expv(alg.scalar_multiplynv(-1, z))), alg.additionnv(alg.expv(z), alg.expv(alg.scalar_multiplynv(-1, z))));
}
Ref<MLPPMatrix> MLPPActivationOld::tanh_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(alg.subtractionm(alg.expm(z), alg.expm(alg.scalar_multiplym(-1, z))), alg.additionm(alg.expm(z), alg.expm(alg.scalar_multiplym(-1, z))));
}

real_t MLPPActivationOld::tanh_derivr(real_t z) {
	return 1 - tanh(z) * tanh(z);
}
Ref<MLPPVector> MLPPActivationOld::tanh_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(-1, alg.scalar_addnv(-1, alg.hadamard_productnv(tanh_normv(z), tanh_normv(z))));
}
Ref<MLPPMatrix> MLPPActivationOld::tanh_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplym(-1, alg.scalar_addm(-1, alg.hadamard_productm(tanh_normm(z), tanh_normm(z))));
}

//CSCH

real_t MLPPActivationOld::csch_normr(real_t z) {
	return 1 / sinh(z);
}
Ref<MLPPVector> MLPPActivationOld::csch_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(alg.onevecv(z->size()), sinh_normv(z));
}

Ref<MLPPMatrix> MLPPActivationOld::csch_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y), sinh_normm(z));
}

real_t MLPPActivationOld::csch_derivr(real_t z) {
	return -csch(z) * coth(z);
}
Ref<MLPPVector> MLPPActivationOld::csch_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnv(alg.scalar_multiplynv(-1, csch_normv(z)), coth_normv(z));
}

Ref<MLPPMatrix> MLPPActivationOld::csch_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productm(alg.scalar_multiplym(-1, csch_normm(z)), coth_normm(z));
}

//SECH

real_t MLPPActivationOld::sech_normr(real_t z) {
	return 1 / cosh(z);
}

Ref<MLPPVector> MLPPActivationOld::sech_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(alg.onevecv(z->size()), cosh_normv(z));

	// return activation(z, deriv, static_cast<void (*)(real_t, bool)>(&sech));
}
Ref<MLPPMatrix> MLPPActivationOld::sech_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y), cosh_normm(z));

	// return activation(z, deriv, static_cast<void (*)(real_t, bool)>(&sech));
}

real_t MLPPActivationOld::sech_derivr(real_t z) {
	return -sech(z) * tanh(z);
}

Ref<MLPPVector> MLPPActivationOld::sech_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnv(alg.scalar_multiplynv(-1, sech_normv(z)), tanh_normv(z));
}
Ref<MLPPMatrix> MLPPActivationOld::sech_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productm(alg.scalar_multiplym(-1, sech_normm(z)), tanh_normm(z));
}

//COTH

real_t MLPPActivationOld::coth_normr(real_t z) {
	return 1 / tanh(z);
}
Ref<MLPPVector> MLPPActivationOld::coth_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(alg.onevecv(z->size()), tanh_normv(z));
}
Ref<MLPPMatrix> MLPPActivationOld::coth_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y), tanh_normm(z));
}

real_t MLPPActivationOld::coth_derivr(real_t z) {
	return -csch_normr(z) * csch_normr(z);
}
Ref<MLPPVector> MLPPActivationOld::coth_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productnv(alg.scalar_multiplynv(-1, csch_normv(z)), csch_normv(z));
}
Ref<MLPPMatrix> MLPPActivationOld::coth_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.hadamard_productm(alg.scalar_multiplym(-1, csch_normm(z)), csch_normm(z));
}

//ARSINH

real_t MLPPActivationOld::arsinh_normr(real_t z) {
	return std::log(z + sqrt(z * z + 1));
}

Ref<MLPPVector> MLPPActivationOld::arsinh_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.logv(alg.additionnv(z, alg.sqrtv(alg.additionnv(alg.hadamard_productnv(z, z), alg.onevecv(z->size())))));
}

Ref<MLPPMatrix> MLPPActivationOld::arsinh_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.logm(alg.additionm(z, alg.sqrtm(alg.additionm(alg.hadamard_productm(z, z), alg.onematm(z->size().x, z->size().y)))));
}

real_t MLPPActivationOld::arsinh_derivr(real_t z) {
	return 1 / sqrt(z * z + 1);
}

Ref<MLPPVector> MLPPActivationOld::arsinh_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(alg.onevecv(z->size()), alg.sqrtv(alg.additionnv(alg.hadamard_productnv(z, z), alg.onevecv(z->size()))));
}

Ref<MLPPMatrix> MLPPActivationOld::arsinh_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y), alg.sqrtm(alg.additionm(alg.hadamard_productm(z, z), alg.onematm(z->size().x, z->size().y))));
}

//ARCOSH

real_t MLPPActivationOld::arcosh_normr(real_t z) {
	return std::log(z + sqrt(z * z - 1));
}
Ref<MLPPVector> MLPPActivationOld::arcosh_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.logv(alg.additionnv(z, alg.sqrtv(alg.subtractionnv(alg.hadamard_productnv(z, z), alg.onevecv(z->size())))));
}

Ref<MLPPMatrix> MLPPActivationOld::arcosh_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.logm(alg.additionm(z, alg.sqrtm(alg.subtractionm(alg.hadamard_productm(z, z), alg.onematm(z->size().x, z->size().y)))));
}

real_t MLPPActivationOld::arcosh_derivr(real_t z) {
	return 1 / sqrt(z * z - 1);
}
Ref<MLPPVector> MLPPActivationOld::arcosh_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(alg.onevecv(z->size()), alg.sqrtv(alg.subtractionnv(alg.hadamard_productnv(z, z), alg.onevecv(z->size()))));
}

Ref<MLPPMatrix> MLPPActivationOld::arcosh_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y), alg.sqrtm(alg.subtractionm(alg.hadamard_productm(z, z), alg.onematm(z->size().x, z->size().y))));
}

//ARTANH

real_t MLPPActivationOld::artanh_normr(real_t z) {
	return 0.5 * std::log((1 + z) / (1 - z));
}
Ref<MLPPVector> MLPPActivationOld::artanh_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(0.5, alg.logv(alg.element_wise_division(alg.additionnv(alg.onevecv(z->size()), z), alg.subtractionnv(alg.onevecv(z->size()), z))));
}

Ref<MLPPMatrix> MLPPActivationOld::artanh_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplym(0.5, alg.logm(alg.element_wise_divisionm(alg.additionm(alg.onematm(z->size().x, z->size().y), z), alg.subtractionm(alg.onematm(z->size().x, z->size().y), z))));
}

real_t MLPPActivationOld::artanh_derivr(real_t z) {
	return 1 / (1 - z * z);
}
Ref<MLPPVector> MLPPActivationOld::artanh_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(alg.onevecv(z->size()), alg.subtractionnv(alg.onevecv(z->size()), alg.hadamard_productnv(z, z)));
}

Ref<MLPPMatrix> MLPPActivationOld::artanh_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y), alg.subtractionnv(alg.onematm(z->size().x, z->size().y), alg.hadamard_productm(z, z)));
}

//ARCSCH

real_t MLPPActivationOld::arcsch_normr(real_t z) {
	return std::log(sqrt(1 + (1 / (z * z))) + (1 / z));
}
Ref<MLPPVector> MLPPActivationOld::arcsch_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.logv(
			alg.additionnv(
					alg.sqrtv(
							alg.additionnv(
									alg.onevecv(z->size()),
									alg.element_wise_division(alg.onevecv(z->size()), alg.hadamard_productnv(z, z)))),
					alg.element_wise_division(alg.onevecv(z->size()), z)));
}
Ref<MLPPMatrix> MLPPActivationOld::arcsch_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.logm(
			alg.additionm(
					alg.sqrtm(
							alg.additionm(alg.onematm(z->size().x, z->size().y),
									alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y), alg.hadamard_productm(z, z)))),
					alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y), z)));
}

real_t MLPPActivationOld::arcsch_derivr(real_t z) {
	return -1 / ((z * z) * sqrt(1 + (1 / (z * z))));
}
Ref<MLPPVector> MLPPActivationOld::arcsch_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(
			alg.fullv(z->size(), -1),
			alg.hadamard_productm(
					alg.hadamard_productnv(z, z),
					alg.sqrtv(alg.additionnv(alg.onevecv(z->size()), alg.element_wise_division(alg.onevecv(z->size()), alg.hadamard_productnv(z, z))))));
}
Ref<MLPPMatrix> MLPPActivationOld::arcsch_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(
			alg.fullm(z->size().x, z->size().y, -1),
			alg.hadamard_productm(alg.hadamard_productm(z, z),
					alg.sqrtm(alg.additionm(alg.onematm(z->size().x, z->size().y),
							alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y), alg.hadamard_productm(z, z))))));
}

//ARSECH

real_t MLPPActivationOld::arsech_normr(real_t z) {
	return std::log((1 / z) + ((1 / z) + 1) * ((1 / z) - 1));
}

Ref<MLPPVector> MLPPActivationOld::arsech_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.logv(
			alg.additionnv(
					alg.element_wise_division(
							alg.onevecv(z->size()), z),
					alg.hadamard_productnv(
							alg.additionnv(alg.element_wise_division(alg.onevecv(z->size()), z), alg.onevecv(z->size())),
							alg.subtractionnv(alg.element_wise_division(alg.onevecv(z->size()), z), alg.onevecv(z->size())))));
}

Ref<MLPPMatrix> MLPPActivationOld::arsech_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.logm(
			alg.additionm(
					alg.element_wise_divisionm(
							alg.onematm(z->size().x, z->size().y), z),
					alg.hadamard_productm(
							alg.additionm(
									alg.element_wise_divisionm(
											alg.onematm(z->size().x, z->size().y), z),
									alg.onematm(z->size().x, z->size().y)),
							alg.subtractionm(
									alg.element_wise_divisionm(
											alg.onematm(z->size().x, z->size().y), z),
									alg.onematm(z->size().x, z->size().y)))));
}

real_t MLPPActivationOld::arsech_derivr(real_t z) {
	return -1 / (z * sqrt(1 - z * z));
}

Ref<MLPPVector> MLPPActivationOld::arsech_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(
			alg.fullv(z->size(), -1),
			alg.hadamard_productnv(
					z,
					alg.sqrtv(
							alg.subtractionnv(alg.onevecv(z->size()), alg.hadamard_productnv(z, z)))));
}

Ref<MLPPMatrix> MLPPActivationOld::arsech_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(
			alg.fullm(z->size().x, z->size().y, -1),
			alg.hadamard_productm(
					z,
					alg.sqrtm(alg.subtractionm(alg.onematm(z->size().x, z->size().y), alg.hadamard_productm(z, z)))));
}

//ARCOTH

real_t MLPPActivationOld::arcoth_normr(real_t z) {
	return 0.5 * std::log((1 + z) / (z - 1));
}
Ref<MLPPVector> MLPPActivationOld::arcoth_normv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplynv(
			0.5,
			alg.logv(alg.element_wise_division(alg.additionnv(alg.onevecv(z->size()), z), alg.subtractionnv(z, alg.onevecv(z->size())))));
}

Ref<MLPPMatrix> MLPPActivationOld::arcoth_normm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.scalar_multiplym(
			0.5,
			alg.logm(alg.element_wise_divisionm(alg.additionm(alg.onematm(z->size().x, z->size().y), z), alg.subtractionm(z, alg.onematm(z->size().x, z->size().y)))));
}

real_t MLPPActivationOld::arcoth_derivr(real_t z) {
	return 1 / (1 - z * z);
}
Ref<MLPPVector> MLPPActivationOld::arcoth_derivv(const Ref<MLPPVector> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_division(alg.onevecv(z->size()), alg.subtractionnv(alg.onevecv(z->size()), alg.hadamard_productnv(z, z)));
}

Ref<MLPPMatrix> MLPPActivationOld::arcoth_derivm(const Ref<MLPPMatrix> &z) {
	MLPPLinAlg alg;

	return alg.element_wise_divisionm(alg.onematm(z->size().x, z->size().y), alg.subtractionm(alg.onematm(z->size().x, z->size().y), alg.hadamard_productm(z, z)));
}

void MLPPActivationOld::_bind_methods() {
	ClassDB::bind_method(D_METHOD("run_activation_real", "func", "z", "deriv"), &MLPPActivationOld::run_activation_real, false);
	ClassDB::bind_method(D_METHOD("run_activation_vector", "func", "z", "deriv"), &MLPPActivationOld::run_activation_vector, false);
	ClassDB::bind_method(D_METHOD("run_activation_matrix", "func", "z", "deriv"), &MLPPActivationOld::run_activation_matrix, false);

	ClassDB::bind_method(D_METHOD("run_activation_norm_real", "func", "z"), &MLPPActivationOld::run_activation_norm_real);
	ClassDB::bind_method(D_METHOD("run_activation_norm_vector", "func", "z"), &MLPPActivationOld::run_activation_norm_vector);
	ClassDB::bind_method(D_METHOD("run_activation_norm_matrix", "func", "z"), &MLPPActivationOld::run_activation_norm_matrix);

	real_t run_activation_norm_real(const ActivationFunction func, const real_t z);
	Ref<MLPPVector> run_activation_norm_vector(const ActivationFunction func, const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> run_activation_norm_matrix(const ActivationFunction func, const Ref<MLPPMatrix> &z);

	ClassDB::bind_method(D_METHOD("run_activation_deriv_real", "func", "z"), &MLPPActivationOld::run_activation_deriv_real);
	ClassDB::bind_method(D_METHOD("run_activation_deriv_vector", "func", "z"), &MLPPActivationOld::run_activation_deriv_vector);
	ClassDB::bind_method(D_METHOD("run_activation_deriv_matrix", "func", "z"), &MLPPActivationOld::run_activation_deriv_matrix);

	//LINEAR

	ClassDB::bind_method(D_METHOD("linear_normr", "z"), &MLPPActivationOld::linear_normr);
	ClassDB::bind_method(D_METHOD("linear_normv", "z"), &MLPPActivationOld::linear_normv);
	ClassDB::bind_method(D_METHOD("linear_normm", "z"), &MLPPActivationOld::linear_normm);

	ClassDB::bind_method(D_METHOD("linear_derivr", "z"), &MLPPActivationOld::linear_derivr);
	ClassDB::bind_method(D_METHOD("linear_derivv", "z"), &MLPPActivationOld::linear_derivv);
	ClassDB::bind_method(D_METHOD("linear_derivm", "z"), &MLPPActivationOld::linear_derivm);

	//SIGMOID

	ClassDB::bind_method(D_METHOD("sigmoid_normr", "z"), &MLPPActivationOld::sigmoid_normr);
	ClassDB::bind_method(D_METHOD("sigmoid_normv", "z"), &MLPPActivationOld::sigmoid_normv);
	ClassDB::bind_method(D_METHOD("sigmoid_normm", "z"), &MLPPActivationOld::sigmoid_normm);

	ClassDB::bind_method(D_METHOD("sigmoid_derivr", "z"), &MLPPActivationOld::sigmoid_derivr);
	ClassDB::bind_method(D_METHOD("sigmoid_derivv", "z"), &MLPPActivationOld::sigmoid_derivv);
	ClassDB::bind_method(D_METHOD("sigmoid_derivm", "z"), &MLPPActivationOld::sigmoid_derivm);

	//SOFTMAX

	ClassDB::bind_method(D_METHOD("softmax_normr", "z"), &MLPPActivationOld::softmax_normr);
	ClassDB::bind_method(D_METHOD("softmax_normv", "z"), &MLPPActivationOld::softmax_normv);
	ClassDB::bind_method(D_METHOD("softmax_normm", "z"), &MLPPActivationOld::softmax_normm);

	ClassDB::bind_method(D_METHOD("softmax_derivr", "z"), &MLPPActivationOld::softmax_derivr);
	ClassDB::bind_method(D_METHOD("softmax_derivv", "z"), &MLPPActivationOld::softmax_derivv);
	ClassDB::bind_method(D_METHOD("softmax_derivm", "z"), &MLPPActivationOld::softmax_derivm);

	//ADJ_SOFTMAX

	real_t adj_softmax_normr(real_t z);
	Ref<MLPPVector> adj_softmax_normv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> adj_softmax_normm(const Ref<MLPPMatrix> &z);

	real_t adj_softmax_derivr(real_t z);
	Ref<MLPPVector> adj_softmax_derivv(const Ref<MLPPVector> &z);
	Ref<MLPPMatrix> adj_softmax_derivm(const Ref<MLPPMatrix> &z);

	//SOFTPLUS

	ClassDB::bind_method(D_METHOD("softplus_normr", "z"), &MLPPActivationOld::softplus_normr);
	ClassDB::bind_method(D_METHOD("softplus_normv", "z"), &MLPPActivationOld::softplus_normv);
	ClassDB::bind_method(D_METHOD("softplus_normm", "z"), &MLPPActivationOld::softplus_normm);

	ClassDB::bind_method(D_METHOD("softplus_derivr", "z"), &MLPPActivationOld::softplus_derivr);
	ClassDB::bind_method(D_METHOD("softplus_derivv", "z"), &MLPPActivationOld::softplus_derivv);
	ClassDB::bind_method(D_METHOD("softplus_derivm", "z"), &MLPPActivationOld::softplus_derivm);

	//SOFTSIGN

	ClassDB::bind_method(D_METHOD("softsign_normr", "z"), &MLPPActivationOld::softsign_normr);
	ClassDB::bind_method(D_METHOD("softsign_normv", "z"), &MLPPActivationOld::softsign_normv);
	ClassDB::bind_method(D_METHOD("softsign_normm", "z"), &MLPPActivationOld::softsign_normm);

	ClassDB::bind_method(D_METHOD("softsign_derivr", "z"), &MLPPActivationOld::softsign_derivr);
	ClassDB::bind_method(D_METHOD("softsign_derivv", "z"), &MLPPActivationOld::softsign_derivv);
	ClassDB::bind_method(D_METHOD("softsign_derivm", "z"), &MLPPActivationOld::softsign_derivm);

	//GAUSSIANCDF

	ClassDB::bind_method(D_METHOD("gaussian_cdf_normr", "z"), &MLPPActivationOld::gaussian_cdf_normr);
	ClassDB::bind_method(D_METHOD("gaussian_cdf_normv", "z"), &MLPPActivationOld::gaussian_cdf_normv);
	ClassDB::bind_method(D_METHOD("gaussian_cdf_normm", "z"), &MLPPActivationOld::gaussian_cdf_normm);

	ClassDB::bind_method(D_METHOD("gaussian_cdf_derivr", "z"), &MLPPActivationOld::gaussian_cdf_derivr);
	ClassDB::bind_method(D_METHOD("gaussian_cdf_derivv", "z"), &MLPPActivationOld::gaussian_cdf_derivv);
	ClassDB::bind_method(D_METHOD("gaussian_cdf_derivm", "z"), &MLPPActivationOld::gaussian_cdf_derivm);

	//CLOGLOG

	ClassDB::bind_method(D_METHOD("cloglog_normr", "z"), &MLPPActivationOld::cloglog_normr);
	ClassDB::bind_method(D_METHOD("cloglog_normv", "z"), &MLPPActivationOld::cloglog_normv);
	ClassDB::bind_method(D_METHOD("cloglog_normm", "z"), &MLPPActivationOld::cloglog_normm);

	ClassDB::bind_method(D_METHOD("cloglog_derivr", "z"), &MLPPActivationOld::cloglog_derivr);
	ClassDB::bind_method(D_METHOD("cloglog_derivv", "z"), &MLPPActivationOld::cloglog_derivv);
	ClassDB::bind_method(D_METHOD("cloglog_derivm", "z"), &MLPPActivationOld::cloglog_derivm);

	//LOGIT

	ClassDB::bind_method(D_METHOD("logit_normr", "z"), &MLPPActivationOld::logit_normr);
	ClassDB::bind_method(D_METHOD("logit_normv", "z"), &MLPPActivationOld::logit_normv);
	ClassDB::bind_method(D_METHOD("logit_normm", "z"), &MLPPActivationOld::logit_normm);

	ClassDB::bind_method(D_METHOD("logit_derivr", "z"), &MLPPActivationOld::logit_derivr);
	ClassDB::bind_method(D_METHOD("logit_derivv", "z"), &MLPPActivationOld::logit_derivv);
	ClassDB::bind_method(D_METHOD("logit_derivm", "z"), &MLPPActivationOld::logit_derivm);

	//UNITSTEP

	ClassDB::bind_method(D_METHOD("unit_step_normr", "z"), &MLPPActivationOld::unit_step_normr);
	ClassDB::bind_method(D_METHOD("unit_step_normv", "z"), &MLPPActivationOld::unit_step_normv);
	ClassDB::bind_method(D_METHOD("unit_step_normm", "z"), &MLPPActivationOld::unit_step_normm);

	ClassDB::bind_method(D_METHOD("unit_step_derivr", "z"), &MLPPActivationOld::unit_step_derivr);
	ClassDB::bind_method(D_METHOD("unit_step_derivv", "z"), &MLPPActivationOld::unit_step_derivv);
	ClassDB::bind_method(D_METHOD("unit_step_derivm", "z"), &MLPPActivationOld::unit_step_derivm);

	//SWISH

	ClassDB::bind_method(D_METHOD("swish_normr", "z"), &MLPPActivationOld::swish_normr);
	ClassDB::bind_method(D_METHOD("swish_normv", "z"), &MLPPActivationOld::swish_normv);
	ClassDB::bind_method(D_METHOD("swish_normm", "z"), &MLPPActivationOld::swish_normm);

	ClassDB::bind_method(D_METHOD("swish_derivr", "z"), &MLPPActivationOld::swish_derivr);
	ClassDB::bind_method(D_METHOD("swish_derivv", "z"), &MLPPActivationOld::swish_derivv);
	ClassDB::bind_method(D_METHOD("swish_derivm", "z"), &MLPPActivationOld::swish_derivm);

	//MISH

	ClassDB::bind_method(D_METHOD("mish_normr", "z"), &MLPPActivationOld::mish_normr);
	ClassDB::bind_method(D_METHOD("mish_normv", "z"), &MLPPActivationOld::mish_normv);
	ClassDB::bind_method(D_METHOD("mish_normm", "z"), &MLPPActivationOld::mish_normm);

	ClassDB::bind_method(D_METHOD("mish_derivr", "z"), &MLPPActivationOld::mish_derivr);
	ClassDB::bind_method(D_METHOD("mish_derivv", "z"), &MLPPActivationOld::mish_derivv);
	ClassDB::bind_method(D_METHOD("mish_derivm", "z"), &MLPPActivationOld::mish_derivm);

	//SINC

	ClassDB::bind_method(D_METHOD("sinc_normr", "z"), &MLPPActivationOld::sinc_normr);
	ClassDB::bind_method(D_METHOD("sinc_normv", "z"), &MLPPActivationOld::sinc_normv);
	ClassDB::bind_method(D_METHOD("sinc_normm", "z"), &MLPPActivationOld::sinc_normm);

	ClassDB::bind_method(D_METHOD("sinc_derivr", "z"), &MLPPActivationOld::sinc_derivr);
	ClassDB::bind_method(D_METHOD("sinc_derivv", "z"), &MLPPActivationOld::sinc_derivv);
	ClassDB::bind_method(D_METHOD("sinc_derivm", "z"), &MLPPActivationOld::sinc_derivm);

	//RELU

	ClassDB::bind_method(D_METHOD("relu_normr", "z"), &MLPPActivationOld::relu_normr);
	ClassDB::bind_method(D_METHOD("relu_normv", "z"), &MLPPActivationOld::relu_normv);
	ClassDB::bind_method(D_METHOD("relu_normm", "z"), &MLPPActivationOld::relu_normm);

	ClassDB::bind_method(D_METHOD("relu_derivr", "z"), &MLPPActivationOld::relu_derivr);
	ClassDB::bind_method(D_METHOD("relu_derivv", "z"), &MLPPActivationOld::relu_derivv);
	ClassDB::bind_method(D_METHOD("relu_derivm", "z"), &MLPPActivationOld::relu_derivm);

	//LEAKYRELU

	ClassDB::bind_method(D_METHOD("leaky_relu_normr", "z"), &MLPPActivationOld::leaky_relu_normr);
	ClassDB::bind_method(D_METHOD("leaky_relu_normv", "z"), &MLPPActivationOld::leaky_relu_normv);
	ClassDB::bind_method(D_METHOD("leaky_relu_normm", "z"), &MLPPActivationOld::leaky_relu_normm);

	ClassDB::bind_method(D_METHOD("leaky_relu_derivr", "z"), &MLPPActivationOld::leaky_relu_derivr);
	ClassDB::bind_method(D_METHOD("leaky_relu_derivv", "z"), &MLPPActivationOld::leaky_relu_derivv);
	ClassDB::bind_method(D_METHOD("leaky_relu_derivm", "z"), &MLPPActivationOld::leaky_relu_derivm);

	//ELU

	ClassDB::bind_method(D_METHOD("elu_normr", "z"), &MLPPActivationOld::elu_normr);
	ClassDB::bind_method(D_METHOD("elu_normv", "z"), &MLPPActivationOld::elu_normv);
	ClassDB::bind_method(D_METHOD("elu_normm", "z"), &MLPPActivationOld::elu_normm);

	ClassDB::bind_method(D_METHOD("elu_derivr", "z"), &MLPPActivationOld::elu_derivr);
	ClassDB::bind_method(D_METHOD("elu_derivv", "z"), &MLPPActivationOld::elu_derivv);
	ClassDB::bind_method(D_METHOD("elu_derivm", "z"), &MLPPActivationOld::elu_derivm);

	//SELU

	ClassDB::bind_method(D_METHOD("selu_normr", "z"), &MLPPActivationOld::selu_normr);
	ClassDB::bind_method(D_METHOD("selu_normv", "z"), &MLPPActivationOld::selu_normv);
	ClassDB::bind_method(D_METHOD("selu_normm", "z"), &MLPPActivationOld::selu_normm);

	ClassDB::bind_method(D_METHOD("selu_derivr", "z"), &MLPPActivationOld::selu_derivr);
	ClassDB::bind_method(D_METHOD("selu_derivv", "z"), &MLPPActivationOld::selu_derivv);
	ClassDB::bind_method(D_METHOD("selu_derivm", "z"), &MLPPActivationOld::selu_derivm);

	//GELU

	ClassDB::bind_method(D_METHOD("gelu_normr", "z"), &MLPPActivationOld::gelu_normr);
	ClassDB::bind_method(D_METHOD("gelu_normv", "z"), &MLPPActivationOld::gelu_normv);
	ClassDB::bind_method(D_METHOD("gelu_normm", "z"), &MLPPActivationOld::gelu_normm);

	ClassDB::bind_method(D_METHOD("gelu_derivr", "z"), &MLPPActivationOld::gelu_derivr);
	ClassDB::bind_method(D_METHOD("gelu_derivv", "z"), &MLPPActivationOld::gelu_derivv);
	ClassDB::bind_method(D_METHOD("gelu_derivm", "z"), &MLPPActivationOld::gelu_derivm);

	//SIGN

	ClassDB::bind_method(D_METHOD("sign_normr", "z"), &MLPPActivationOld::sign_normr);
	ClassDB::bind_method(D_METHOD("sign_normv", "z"), &MLPPActivationOld::sign_normv);
	ClassDB::bind_method(D_METHOD("sign_normm", "z"), &MLPPActivationOld::sign_normm);

	ClassDB::bind_method(D_METHOD("sign_derivr", "z"), &MLPPActivationOld::sign_derivr);
	ClassDB::bind_method(D_METHOD("sign_derivv", "z"), &MLPPActivationOld::sign_derivv);
	ClassDB::bind_method(D_METHOD("sign_derivm", "z"), &MLPPActivationOld::sign_derivm);

	//SINH

	ClassDB::bind_method(D_METHOD("sinh_normr", "z"), &MLPPActivationOld::sinh_normr);
	ClassDB::bind_method(D_METHOD("sinh_normv", "z"), &MLPPActivationOld::sinh_normv);
	ClassDB::bind_method(D_METHOD("sinh_normm", "z"), &MLPPActivationOld::sinh_normm);

	ClassDB::bind_method(D_METHOD("sinh_derivr", "z"), &MLPPActivationOld::sinh_derivr);
	ClassDB::bind_method(D_METHOD("sinh_derivv", "z"), &MLPPActivationOld::sinh_derivv);
	ClassDB::bind_method(D_METHOD("sinh_derivm", "z"), &MLPPActivationOld::sinh_derivm);

	//COSH

	ClassDB::bind_method(D_METHOD("cosh_normr", "z"), &MLPPActivationOld::cosh_normr);
	ClassDB::bind_method(D_METHOD("cosh_normv", "z"), &MLPPActivationOld::cosh_normv);
	ClassDB::bind_method(D_METHOD("cosh_normm", "z"), &MLPPActivationOld::cosh_normm);

	ClassDB::bind_method(D_METHOD("cosh_derivr", "z"), &MLPPActivationOld::cosh_derivr);
	ClassDB::bind_method(D_METHOD("cosh_derivv", "z"), &MLPPActivationOld::cosh_derivv);
	ClassDB::bind_method(D_METHOD("cosh_derivm", "z"), &MLPPActivationOld::cosh_derivm);

	//TANH

	ClassDB::bind_method(D_METHOD("tanh_normr", "z"), &MLPPActivationOld::tanh_normr);
	ClassDB::bind_method(D_METHOD("tanh_normv", "z"), &MLPPActivationOld::tanh_normv);
	ClassDB::bind_method(D_METHOD("tanh_normm", "z"), &MLPPActivationOld::tanh_normm);

	ClassDB::bind_method(D_METHOD("tanh_derivr", "z"), &MLPPActivationOld::tanh_derivr);
	ClassDB::bind_method(D_METHOD("tanh_derivv", "z"), &MLPPActivationOld::tanh_derivv);
	ClassDB::bind_method(D_METHOD("tanh_derivm", "z"), &MLPPActivationOld::tanh_derivm);

	//CSCH

	ClassDB::bind_method(D_METHOD("csch_normr", "z"), &MLPPActivationOld::csch_normr);
	ClassDB::bind_method(D_METHOD("csch_normv", "z"), &MLPPActivationOld::csch_normv);
	ClassDB::bind_method(D_METHOD("csch_normm", "z"), &MLPPActivationOld::csch_normm);

	ClassDB::bind_method(D_METHOD("csch_derivr", "z"), &MLPPActivationOld::csch_derivr);
	ClassDB::bind_method(D_METHOD("csch_derivv", "z"), &MLPPActivationOld::csch_derivv);
	ClassDB::bind_method(D_METHOD("csch_derivm", "z"), &MLPPActivationOld::csch_derivm);

	//SECH

	ClassDB::bind_method(D_METHOD("sech_normr", "z"), &MLPPActivationOld::sech_normr);
	ClassDB::bind_method(D_METHOD("sech_normv", "z"), &MLPPActivationOld::sech_normv);
	ClassDB::bind_method(D_METHOD("sech_normm", "z"), &MLPPActivationOld::sech_normm);

	ClassDB::bind_method(D_METHOD("sech_derivr", "z"), &MLPPActivationOld::sech_derivr);
	ClassDB::bind_method(D_METHOD("sech_derivv", "z"), &MLPPActivationOld::sech_derivv);
	ClassDB::bind_method(D_METHOD("sech_derivm", "z"), &MLPPActivationOld::sech_derivm);

	//COTH

	ClassDB::bind_method(D_METHOD("coth_normr", "z"), &MLPPActivationOld::coth_normr);
	ClassDB::bind_method(D_METHOD("coth_normv", "z"), &MLPPActivationOld::coth_normv);
	ClassDB::bind_method(D_METHOD("coth_normm", "z"), &MLPPActivationOld::coth_normm);

	ClassDB::bind_method(D_METHOD("coth_derivr", "z"), &MLPPActivationOld::coth_derivr);
	ClassDB::bind_method(D_METHOD("coth_derivv", "z"), &MLPPActivationOld::coth_derivv);
	ClassDB::bind_method(D_METHOD("coth_derivm", "z"), &MLPPActivationOld::coth_derivm);

	//ARSINH

	ClassDB::bind_method(D_METHOD("arsinh_normr", "z"), &MLPPActivationOld::arsinh_normr);
	ClassDB::bind_method(D_METHOD("arsinh_normv", "z"), &MLPPActivationOld::arsinh_normv);
	ClassDB::bind_method(D_METHOD("arsinh_normm", "z"), &MLPPActivationOld::arsinh_normm);

	ClassDB::bind_method(D_METHOD("arsinh_derivr", "z"), &MLPPActivationOld::arsinh_derivr);
	ClassDB::bind_method(D_METHOD("arsinh_derivv", "z"), &MLPPActivationOld::arsinh_derivv);
	ClassDB::bind_method(D_METHOD("arsinh_derivm", "z"), &MLPPActivationOld::arsinh_derivm);

	//ARCOSH

	ClassDB::bind_method(D_METHOD("arcosh_normr", "z"), &MLPPActivationOld::arcosh_normr);
	ClassDB::bind_method(D_METHOD("arcosh_normv", "z"), &MLPPActivationOld::arcosh_normv);
	ClassDB::bind_method(D_METHOD("arcosh_normm", "z"), &MLPPActivationOld::arcosh_normm);

	ClassDB::bind_method(D_METHOD("arcosh_derivr", "z"), &MLPPActivationOld::arcosh_derivr);
	ClassDB::bind_method(D_METHOD("arcosh_derivv", "z"), &MLPPActivationOld::arcosh_derivv);
	ClassDB::bind_method(D_METHOD("arcosh_derivm", "z"), &MLPPActivationOld::arcosh_derivm);

	//ARTANH

	ClassDB::bind_method(D_METHOD("artanh_normr", "z"), &MLPPActivationOld::artanh_normr);
	ClassDB::bind_method(D_METHOD("artanh_normv", "z"), &MLPPActivationOld::artanh_normv);
	ClassDB::bind_method(D_METHOD("artanh_normm", "z"), &MLPPActivationOld::artanh_normm);

	ClassDB::bind_method(D_METHOD("artanh_derivr", "z"), &MLPPActivationOld::artanh_derivr);
	ClassDB::bind_method(D_METHOD("artanh_derivv", "z"), &MLPPActivationOld::artanh_derivv);
	ClassDB::bind_method(D_METHOD("artanh_derivm", "z"), &MLPPActivationOld::artanh_derivm);

	//ARCSCH

	ClassDB::bind_method(D_METHOD("arcsch_normr", "z"), &MLPPActivationOld::arcsch_normr);
	ClassDB::bind_method(D_METHOD("arcsch_normv", "z"), &MLPPActivationOld::arcsch_normv);
	ClassDB::bind_method(D_METHOD("arcsch_normm", "z"), &MLPPActivationOld::arcsch_normm);

	ClassDB::bind_method(D_METHOD("arcsch_derivr", "z"), &MLPPActivationOld::arcsch_derivr);
	ClassDB::bind_method(D_METHOD("arcsch_derivv", "z"), &MLPPActivationOld::arcsch_derivv);
	ClassDB::bind_method(D_METHOD("arcsch_derivm", "z"), &MLPPActivationOld::arcsch_derivm);

	//ARSECH

	ClassDB::bind_method(D_METHOD("arsech_normr", "z"), &MLPPActivationOld::arsech_normr);
	ClassDB::bind_method(D_METHOD("arsech_normv", "z"), &MLPPActivationOld::arsech_normv);
	ClassDB::bind_method(D_METHOD("arsech_normm", "z"), &MLPPActivationOld::arsech_normm);

	ClassDB::bind_method(D_METHOD("arsech_derivr", "z"), &MLPPActivationOld::arsech_derivr);
	ClassDB::bind_method(D_METHOD("arsech_derivv", "z"), &MLPPActivationOld::arsech_derivv);
	ClassDB::bind_method(D_METHOD("arsech_derivm", "z"), &MLPPActivationOld::arsech_derivm);

	//ARCOTH

	ClassDB::bind_method(D_METHOD("arcoth_normr", "z"), &MLPPActivationOld::arcoth_normr);
	ClassDB::bind_method(D_METHOD("arcoth_normv", "z"), &MLPPActivationOld::arcoth_normv);
	ClassDB::bind_method(D_METHOD("arcoth_normm", "z"), &MLPPActivationOld::arcoth_normm);

	ClassDB::bind_method(D_METHOD("arcoth_derivr", "z"), &MLPPActivationOld::arcoth_derivr);
	ClassDB::bind_method(D_METHOD("arcoth_derivv", "z"), &MLPPActivationOld::arcoth_derivv);
	ClassDB::bind_method(D_METHOD("arcoth_derivm", "z"), &MLPPActivationOld::arcoth_derivm);

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

//======================== OLD =============================

real_t MLPPActivationOld::linear(real_t z, bool deriv) {
	if (deriv) {
		return 1;
	}
	return z;
}

std::vector<real_t> MLPPActivationOld::linear(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		MLPPLinAlg alg;
		return alg.onevec(z.size());
	}
	return z;
}

std::vector<std::vector<real_t>> MLPPActivationOld::linear(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		MLPPLinAlg alg;
		return alg.onemat(z.size(), z[0].size());
	}
	return z;
}

real_t MLPPActivationOld::sigmoid(real_t z, bool deriv) {
	if (deriv) {
		return sigmoid(z) * (1 - sigmoid(z));
	}
	return 1 / (1 + exp(-z));
}

std::vector<real_t> MLPPActivationOld::sigmoid(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.subtraction(sigmoid(z), alg.hadamard_product(sigmoid(z), sigmoid(z)));
	}
	return alg.elementWiseDivision(alg.onevec(z.size()), alg.addition(alg.onevec(z.size()), alg.exp(alg.scalarMultiply(-1, z))));
}

std::vector<std::vector<real_t>> MLPPActivationOld::sigmoid(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.subtraction(sigmoid(z), alg.hadamard_product(sigmoid(z), sigmoid(z)));
	}
	return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.addition(alg.onemat(z.size(), z[0].size()), alg.exp(alg.scalarMultiply(-1, z))));
}

std::vector<real_t> MLPPActivationOld::softmax(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	std::vector<real_t> a;
	a.resize(z.size());
	std::vector<real_t> expZ = alg.exp(z);
	real_t sum = 0;

	for (uint32_t i = 0; i < z.size(); i++) {
		sum += expZ[i];
	}
	for (uint32_t i = 0; i < z.size(); i++) {
		a[i] = expZ[i] / sum;
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivationOld::softmax(std::vector<std::vector<real_t>> z, bool deriv) {
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < z.size(); i++) {
		a[i] = softmax(z[i]);
	}
	return a;
}

std::vector<real_t> MLPPActivationOld::adjSoftmax(std::vector<real_t> z) {
	MLPPLinAlg alg;
	std::vector<real_t> a;
	real_t C = -*std::max_element(z.begin(), z.end());
	z = alg.scalarAdd(C, z);

	return softmax(z);
}

std::vector<std::vector<real_t>> MLPPActivationOld::adjSoftmax(std::vector<std::vector<real_t>> z) {
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < z.size(); i++) {
		a[i] = adjSoftmax(z[i]);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivationOld::softmaxDeriv(std::vector<real_t> z) {
	std::vector<std::vector<real_t>> deriv;
	std::vector<real_t> a = softmax(z);
	deriv.resize(a.size());
	for (uint32_t i = 0; i < deriv.size(); i++) {
		deriv[i].resize(a.size());
	}
	for (uint32_t i = 0; i < a.size(); i++) {
		for (uint32_t j = 0; j < z.size(); j++) {
			if (i == j) {
				deriv[i][j] = a[i] * (1 - a[i]);
			} else {
				deriv[i][j] = -a[i] * a[j];
			}
		}
	}
	return deriv;
}

std::vector<std::vector<std::vector<real_t>>> MLPPActivationOld::softmaxDeriv(std::vector<std::vector<real_t>> z) {
	MLPPLinAlg alg;
	std::vector<std::vector<std::vector<real_t>>> deriv;
	std::vector<std::vector<real_t>> a = softmax(z);

	deriv.resize(a.size());
	for (uint32_t i = 0; i < deriv.size(); i++) {
		deriv[i].resize(a.size());
	}
	for (uint32_t i = 0; i < a.size(); i++) {
		for (uint32_t j = 0; j < z.size(); j++) {
			if (i == j) {
				deriv[i][j] = alg.subtraction(a[i], alg.hadamard_product(a[i], a[i]));
			} else {
				deriv[i][j] = alg.scalarMultiply(-1, alg.hadamard_product(a[i], a[j]));
			}
		}
	}
	return deriv;
}

real_t MLPPActivationOld::softplus(real_t z, bool deriv) {
	if (deriv) {
		return sigmoid(z);
	}
	return std::log(1 + exp(z));
}

std::vector<real_t> MLPPActivationOld::softplus(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		return sigmoid(z);
	}
	MLPPLinAlg alg;
	return alg.log(alg.addition(alg.onevec(z.size()), alg.exp(z)));
}

std::vector<std::vector<real_t>> MLPPActivationOld::softplus(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		return sigmoid(z);
	}
	MLPPLinAlg alg;
	return alg.log(alg.addition(alg.onemat(z.size(), z[0].size()), alg.exp(z)));
}

real_t MLPPActivationOld::softsign(real_t z, bool deriv) {
	if (deriv) {
		return 1 / ((1 + abs(z)) * (1 + abs(z)));
	}
	return z / (1 + abs(z));
}

std::vector<real_t> MLPPActivationOld::softsign(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onevec(z.size()), alg.exponentiate(alg.addition(alg.onevec(z.size()), alg.abs(z)), 2));
	}
	return alg.elementWiseDivision(z, alg.addition(alg.onevec(z.size()), alg.abs(z)));
}

std::vector<std::vector<real_t>> MLPPActivationOld::softsign(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.exponentiate(alg.addition(alg.onemat(z.size(), z[0].size()), alg.abs(z)), 2));
	}
	return alg.elementWiseDivision(z, alg.addition(alg.onemat(z.size(), z[0].size()), alg.abs(z)));
}

real_t MLPPActivationOld::gaussianCDF(real_t z, bool deriv) {
	if (deriv) {
		return (1 / sqrt(2 * M_PI)) * exp(-z * z / 2);
	}
	return 0.5 * (1 + erf(z / sqrt(2)));
}

std::vector<real_t> MLPPActivationOld::gaussianCDF(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.scalarMultiply(1 / sqrt(2 * M_PI), alg.exp(alg.scalarMultiply(-1 / 2, alg.hadamard_product(z, z))));
	}
	return alg.scalarMultiply(0.5, alg.addition(alg.onevec(z.size()), alg.erf(alg.scalarMultiply(1 / sqrt(2), z))));
}

std::vector<std::vector<real_t>> MLPPActivationOld::gaussianCDF(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.scalarMultiply(1 / sqrt(2 * M_PI), alg.exp(alg.scalarMultiply(-1 / 2, alg.hadamard_product(z, z))));
	}
	return alg.scalarMultiply(0.5, alg.addition(alg.onemat(z.size(), z[0].size()), alg.erf(alg.scalarMultiply(1 / sqrt(2), z))));
}

real_t MLPPActivationOld::cloglog(real_t z, bool deriv) {
	if (deriv) {
		return exp(z - exp(z));
	}
	return 1 - exp(-exp(z));
}

std::vector<real_t> MLPPActivationOld::cloglog(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.exp(alg.scalarMultiply(-1, alg.exp(z)));
	}
	return alg.scalarMultiply(-1, alg.scalarAdd(-1, alg.exp(alg.scalarMultiply(-1, alg.exp(z)))));
}

std::vector<std::vector<real_t>> MLPPActivationOld::cloglog(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.exp(alg.scalarMultiply(-1, alg.exp(z)));
	}
	return alg.scalarMultiply(-1, alg.scalarAdd(-1, alg.exp(alg.scalarMultiply(-1, alg.exp(z)))));
}

real_t MLPPActivationOld::logit(real_t z, bool deriv) {
	if (deriv) {
		return 1 / z - 1 / (z - 1);
	}
	return std::log(z / (1 - z));
}

std::vector<real_t> MLPPActivationOld::logit(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.subtraction(alg.elementWiseDivision(alg.onevec(z.size()), z), alg.elementWiseDivision(alg.onevec(z.size()), alg.subtraction(z, alg.onevec(z.size()))));
	}
	return alg.log(alg.elementWiseDivision(z, alg.subtraction(alg.onevec(z.size()), z)));
}

std::vector<std::vector<real_t>> MLPPActivationOld::logit(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.subtraction(alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), z), alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.subtraction(z, alg.onemat(z.size(), z[0].size()))));
	}
	return alg.log(alg.elementWiseDivision(z, alg.subtraction(alg.onemat(z.size(), z[0].size()), z)));
}

real_t MLPPActivationOld::unitStep(real_t z, bool deriv) {
	if (deriv) {
		return 0;
	}
	return z < 0 ? 0 : 1;
}

std::vector<real_t> MLPPActivationOld::unitStep(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		std::vector<real_t> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = unitStep(z[i], true);
		}
		return lderiv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = unitStep(z[i]);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivationOld::unitStep(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = unitStep(z[i], true);
		}
		return lderiv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = unitStep(z[i]);
	}
	return a;
}

real_t MLPPActivationOld::swish(real_t z, bool deriv) {
	if (deriv) {
		return swish(z) + sigmoid(z) * (1 - swish(z));
	}
	return z * sigmoid(z);
}

std::vector<real_t> MLPPActivationOld::swish(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		alg.addition(swish(z), alg.subtraction(sigmoid(z), alg.hadamard_product(sigmoid(z), swish(z))));
	}
	return alg.hadamard_product(z, sigmoid(z));
}

std::vector<std::vector<real_t>> MLPPActivationOld::swish(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		alg.addition(swish(z), alg.subtraction(sigmoid(z), alg.hadamard_product(sigmoid(z), swish(z))));
	}
	return alg.hadamard_product(z, sigmoid(z));
}

real_t MLPPActivationOld::mish(real_t z, bool deriv) {
	if (deriv) {
		return sech(softplus(z)) * sech(softplus(z)) * z * sigmoid(z) + mish(z) / z;
	}
	return z * tanh(softplus(z));
}

std::vector<real_t> MLPPActivationOld::mish(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.addition(alg.hadamard_product(alg.hadamard_product(alg.hadamard_product(sech(softplus(z)), sech(softplus(z))), z), sigmoid(z)), alg.elementWiseDivision(mish(z), z));
	}
	return alg.hadamard_product(z, tanh(softplus(z)));
}

std::vector<std::vector<real_t>> MLPPActivationOld::mish(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.addition(alg.hadamard_product(alg.hadamard_product(alg.hadamard_product(sech(softplus(z)), sech(softplus(z))), z), sigmoid(z)), alg.elementWiseDivision(mish(z), z));
	}
	return alg.hadamard_product(z, tanh(softplus(z)));
}

real_t MLPPActivationOld::sinc(real_t z, bool deriv) {
	if (deriv) {
		return (z * std::cos(z) - std::sin(z)) / (z * z);
	}
	return std::sin(z) / z;
}

std::vector<real_t> MLPPActivationOld::sinc(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.subtraction(alg.hadamard_product(z, alg.cos(z)), alg.sin(z)), alg.hadamard_product(z, z));
	}
	return alg.elementWiseDivision(alg.sin(z), z);
}

std::vector<std::vector<real_t>> MLPPActivationOld::sinc(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.subtraction(alg.hadamard_product(z, alg.cos(z)), alg.sin(z)), alg.hadamard_product(z, z));
	}
	return alg.elementWiseDivision(alg.sin(z), z);
}

real_t MLPPActivationOld::RELU(real_t z, bool deriv) {
	if (deriv) {
		if (z <= 0) {
			return 0;
		} else {
			return 1;
		}
	}
	return fmax(0, z);
}

std::vector<real_t> MLPPActivationOld::RELU(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		std::vector<real_t> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = RELU(z[i], true);
		}
		return lderiv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = RELU(z[i]);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivationOld::RELU(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = RELU(z[i], true);
		}
		return lderiv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = RELU(z[i]);
	}
	return a;
}

real_t MLPPActivationOld::leakyReLU(real_t z, real_t c, bool deriv) {
	if (deriv) {
		if (z <= 0) {
			return c;
		} else {
			return 1;
		}
	}
	return fmax(c * z, z);
}

std::vector<real_t> MLPPActivationOld::leakyReLU(std::vector<real_t> z, real_t c, bool deriv) {
	if (deriv) {
		std::vector<real_t> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = leakyReLU(z[i], c, true);
		}
		return lderiv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = leakyReLU(z[i], c);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivationOld::leakyReLU(std::vector<std::vector<real_t>> z, real_t c, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = leakyReLU(z[i], c, true);
		}
		return lderiv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = leakyReLU(z[i], c);
	}
	return a;
}

real_t MLPPActivationOld::ELU(real_t z, real_t c, bool deriv) {
	if (deriv) {
		if (z <= 0) {
			return c * exp(z);
		} else {
			return 1;
		}
	}
	if (z >= 0) {
		return z;
	} else {
		return c * (exp(z) - 1);
	}
}

std::vector<real_t> MLPPActivationOld::ELU(std::vector<real_t> z, real_t c, bool deriv) {
	if (deriv) {
		std::vector<real_t> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = ELU(z[i], c, true);
		}
		return lderiv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = ELU(z[i], c);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivationOld::ELU(std::vector<std::vector<real_t>> z, real_t c, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = ELU(z[i], c, true);
		}
		return lderiv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = ELU(z[i], c);
	}
	return a;
}

real_t MLPPActivationOld::SELU(real_t z, real_t lambda, real_t c, bool deriv) {
	if (deriv) {
		return ELU(z, c, true);
	}
	return lambda * ELU(z, c);
}

std::vector<real_t> MLPPActivationOld::SELU(std::vector<real_t> z, real_t lambda, real_t c, bool deriv) {
	if (deriv) {
		std::vector<real_t> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = SELU(z[i], lambda, c, true);
		}
		return lderiv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = SELU(z[i], lambda, c);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivationOld::SELU(std::vector<std::vector<real_t>> z, real_t lambda, real_t c, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = SELU(z[i], lambda, c, true);
		}
		return lderiv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = SELU(z[i], lambda, c);
	}
	return a;
}

real_t MLPPActivationOld::GELU(real_t z, bool deriv) {
	if (deriv) {
		return 0.5 * tanh(0.0356774 * std::pow(z, 3) + 0.797885 * z) + (0.0535161 * std::pow(z, 3) + 0.398942 * z) * std::pow(sech(0.0356774 * std::pow(z, 3) + 0.797885 * z), 2) + 0.5;
	}
	return 0.5 * z * (1 + tanh(sqrt(2 / M_PI) * (z + 0.044715 * std::pow(z, 3))));
}

std::vector<real_t> MLPPActivationOld::GELU(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		std::vector<real_t> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = GELU(z[i], true);
		}
		return lderiv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = GELU(z[i]);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivationOld::GELU(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = GELU(z[i], true);
		}
		return lderiv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = GELU(z[i]);
	}
	return a;
}

real_t MLPPActivationOld::sign(real_t z, bool deriv) {
	if (deriv) {
		return 0;
	}
	if (z < 0) {
		return -1;
	} else if (z == 0) {
		return 0;
	} else {
		return 1;
	}
}

std::vector<real_t> MLPPActivationOld::sign(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		std::vector<real_t> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = sign(z[i], true);
		}
		return lderiv;
	}
	std::vector<real_t> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = sign(z[i]);
	}
	return a;
}

std::vector<std::vector<real_t>> MLPPActivationOld::sign(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		std::vector<std::vector<real_t>> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = sign(z[i], true);
		}
		return lderiv;
	}
	std::vector<std::vector<real_t>> a;
	a.resize(z.size());

	for (uint32_t i = 0; i < a.size(); i++) {
		a[i] = sign(z[i]);
	}
	return a;
}

real_t MLPPActivationOld::sinh(real_t z, bool deriv) {
	if (deriv) {
		return cosh(z);
	}
	return 0.5 * (exp(z) - exp(-z));
}

std::vector<real_t> MLPPActivationOld::sinh(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		return cosh(z);
	}
	MLPPLinAlg alg;
	return alg.scalarMultiply(0.5, alg.subtraction(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
}

std::vector<std::vector<real_t>> MLPPActivationOld::sinh(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		return cosh(z);
	}
	MLPPLinAlg alg;
	return alg.scalarMultiply(0.5, alg.subtraction(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
}

real_t MLPPActivationOld::cosh(real_t z, bool deriv) {
	if (deriv) {
		return sinh(z);
	}
	return 0.5 * (exp(z) + exp(-z));
}

std::vector<real_t> MLPPActivationOld::cosh(std::vector<real_t> z, bool deriv) {
	if (deriv) {
		return sinh(z);
	}
	MLPPLinAlg alg;
	return alg.scalarMultiply(0.5, alg.addition(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
}

std::vector<std::vector<real_t>> MLPPActivationOld::cosh(std::vector<std::vector<real_t>> z, bool deriv) {
	if (deriv) {
		return sinh(z);
	}
	MLPPLinAlg alg;
	return alg.scalarMultiply(0.5, alg.addition(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
}

real_t MLPPActivationOld::tanh(real_t z, bool deriv) {
	if (deriv) {
		return 1 - tanh(z) * tanh(z);
	}
	return (exp(z) - exp(-z)) / (exp(z) + exp(-z));
}

std::vector<real_t> MLPPActivationOld::tanh(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.scalarMultiply(-1, alg.scalarAdd(-1, alg.hadamard_product(tanh(z), tanh(z))));
	}
	return alg.elementWiseDivision(alg.subtraction(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))), alg.addition(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
}

std::vector<std::vector<real_t>> MLPPActivationOld::tanh(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.scalarMultiply(-1, alg.scalarAdd(-1, alg.hadamard_product(tanh(z), tanh(z))));
	}

	return alg.elementWiseDivision(alg.subtraction(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))), alg.addition(alg.exp(z), alg.exp(alg.scalarMultiply(-1, z))));
}

real_t MLPPActivationOld::csch(real_t z, bool deriv) {
	if (deriv) {
		return -csch(z) * coth(z);
	}
	return 1 / sinh(z);
}

std::vector<real_t> MLPPActivationOld::csch(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.hadamard_product(alg.scalarMultiply(-1, csch(z)), coth(z));
	}
	return alg.elementWiseDivision(alg.onevec(z.size()), sinh(z));
}

std::vector<std::vector<real_t>> MLPPActivationOld::csch(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.hadamard_product(alg.scalarMultiply(-1, csch(z)), coth(z));
	}
	return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), sinh(z));
}

real_t MLPPActivationOld::sech(real_t z, bool deriv) {
	if (deriv) {
		return -sech(z) * tanh(z);
	}
	return 1 / cosh(z);
}

std::vector<real_t> MLPPActivationOld::sech(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.hadamard_product(alg.scalarMultiply(-1, sech(z)), tanh(z));
	}
	return alg.elementWiseDivision(alg.onevec(z.size()), cosh(z));

	// return activation(z, deriv, static_cast<void (*)(real_t, bool)>(&sech));
}

std::vector<std::vector<real_t>> MLPPActivationOld::sech(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.hadamard_product(alg.scalarMultiply(-1, sech(z)), tanh(z));
	}
	return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), cosh(z));

	// return activation(z, deriv, static_cast<void (*)(real_t, bool)>(&sech));
}

real_t MLPPActivationOld::coth(real_t z, bool deriv) {
	if (deriv) {
		return -csch(z) * csch(z);
	}
	return 1 / tanh(z);
}

std::vector<real_t> MLPPActivationOld::coth(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.hadamard_product(alg.scalarMultiply(-1, csch(z)), csch(z));
	}
	return alg.elementWiseDivision(alg.onevec(z.size()), tanh(z));
}

std::vector<std::vector<real_t>> MLPPActivationOld::coth(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.hadamard_product(alg.scalarMultiply(-1, csch(z)), csch(z));
	}
	return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), tanh(z));
}

real_t MLPPActivationOld::arsinh(real_t z, bool deriv) {
	if (deriv) {
		return 1 / sqrt(z * z + 1);
	}
	return std::log(z + sqrt(z * z + 1));
}

std::vector<real_t> MLPPActivationOld::arsinh(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onevec(z.size()), alg.sqrt(alg.addition(alg.hadamard_product(z, z), alg.onevec(z.size()))));
	}
	return alg.log(alg.addition(z, alg.sqrt(alg.addition(alg.hadamard_product(z, z), alg.onevec(z.size())))));
}

std::vector<std::vector<real_t>> MLPPActivationOld::arsinh(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.sqrt(alg.addition(alg.hadamard_product(z, z), alg.onemat(z.size(), z[0].size()))));
	}
	return alg.log(alg.addition(z, alg.sqrt(alg.addition(alg.hadamard_product(z, z), alg.onemat(z.size(), z[0].size())))));
}

real_t MLPPActivationOld::arcosh(real_t z, bool deriv) {
	if (deriv) {
		return 1 / sqrt(z * z - 1);
	}
	return std::log(z + sqrt(z * z - 1));
}

std::vector<real_t> MLPPActivationOld::arcosh(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onevec(z.size()), alg.sqrt(alg.subtraction(alg.hadamard_product(z, z), alg.onevec(z.size()))));
	}
	return alg.log(alg.addition(z, alg.sqrt(alg.subtraction(alg.hadamard_product(z, z), alg.onevec(z.size())))));
}

std::vector<std::vector<real_t>> MLPPActivationOld::arcosh(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.sqrt(alg.subtraction(alg.hadamard_product(z, z), alg.onemat(z.size(), z[0].size()))));
	}
	return alg.log(alg.addition(z, alg.sqrt(alg.subtraction(alg.hadamard_product(z, z), alg.onemat(z.size(), z[0].size())))));
}

real_t MLPPActivationOld::artanh(real_t z, bool deriv) {
	if (deriv) {
		return 1 / (1 - z * z);
	}
	return 0.5 * std::log((1 + z) / (1 - z));
}

std::vector<real_t> MLPPActivationOld::artanh(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onevec(z.size()), alg.subtraction(alg.onevec(z.size()), alg.hadamard_product(z, z)));
	}
	return alg.scalarMultiply(0.5, alg.log(alg.elementWiseDivision(alg.addition(alg.onevec(z.size()), z), alg.subtraction(alg.onevec(z.size()), z))));
}

std::vector<std::vector<real_t>> MLPPActivationOld::artanh(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.subtraction(alg.onemat(z.size(), z[0].size()), alg.hadamard_product(z, z)));
	}
	return alg.scalarMultiply(0.5, alg.log(alg.elementWiseDivision(alg.addition(alg.onemat(z.size(), z[0].size()), z), alg.subtraction(alg.onemat(z.size(), z[0].size()), z))));
}

real_t MLPPActivationOld::arcsch(real_t z, bool deriv) {
	if (deriv) {
		return -1 / ((z * z) * sqrt(1 + (1 / (z * z))));
	}
	return std::log(sqrt(1 + (1 / (z * z))) + (1 / z));
}

std::vector<real_t> MLPPActivationOld::arcsch(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.full(z.size(), -1), alg.hadamard_product(alg.hadamard_product(z, z), alg.sqrt(alg.addition(alg.onevec(z.size()), alg.elementWiseDivision(alg.onevec(z.size()), alg.hadamard_product(z, z))))));
	}
	return alg.log(alg.addition(alg.sqrt(alg.addition(alg.onevec(z.size()), alg.elementWiseDivision(alg.onevec(z.size()), alg.hadamard_product(z, z)))), alg.elementWiseDivision(alg.onevec(z.size()), z)));
}

std::vector<std::vector<real_t>> MLPPActivationOld::arcsch(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.full(z.size(), z[0].size(), -1), alg.hadamard_product(alg.hadamard_product(z, z), alg.sqrt(alg.addition(alg.onemat(z.size(), z[0].size()), alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.hadamard_product(z, z))))));
	}
	return alg.log(alg.addition(alg.sqrt(alg.addition(alg.onemat(z.size(), z[0].size()), alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.hadamard_product(z, z)))), alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), z)));
}

real_t MLPPActivationOld::arsech(real_t z, bool deriv) {
	if (deriv) {
		return -1 / (z * sqrt(1 - z * z));
	}
	return std::log((1 / z) + ((1 / z) + 1) * ((1 / z) - 1));
}

std::vector<real_t> MLPPActivationOld::arsech(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.full(z.size(), -1), alg.hadamard_product(z, alg.sqrt(alg.subtraction(alg.onevec(z.size()), alg.hadamard_product(z, z)))));
	}
	return alg.log(alg.addition(alg.elementWiseDivision(alg.onevec(z.size()), z), alg.hadamard_product(alg.addition(alg.elementWiseDivision(alg.onevec(z.size()), z), alg.onevec(z.size())), alg.subtraction(alg.elementWiseDivision(alg.onevec(z.size()), z), alg.onevec(z.size())))));
}

std::vector<std::vector<real_t>> MLPPActivationOld::arsech(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.full(z.size(), z[0].size(), -1), alg.hadamard_product(z, alg.sqrt(alg.subtraction(alg.onemat(z.size(), z[0].size()), alg.hadamard_product(z, z)))));
	}
	return alg.log(alg.addition(alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), z), alg.hadamard_product(alg.addition(alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), z), alg.onemat(z.size(), z[0].size())), alg.subtraction(alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), z), alg.onemat(z.size(), z[0].size())))));
}

real_t MLPPActivationOld::arcoth(real_t z, bool deriv) {
	if (deriv) {
		return 1 / (1 - z * z);
	}
	return 0.5 * std::log((1 + z) / (z - 1));
}

std::vector<real_t> MLPPActivationOld::arcoth(std::vector<real_t> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onevec(z.size()), alg.subtraction(alg.onevec(z.size()), alg.hadamard_product(z, z)));
	}
	return alg.scalarMultiply(0.5, alg.log(alg.elementWiseDivision(alg.addition(alg.onevec(z.size()), z), alg.subtraction(z, alg.onevec(z.size())))));
}

std::vector<std::vector<real_t>> MLPPActivationOld::arcoth(std::vector<std::vector<real_t>> z, bool deriv) {
	MLPPLinAlg alg;
	if (deriv) {
		return alg.elementWiseDivision(alg.onemat(z.size(), z[0].size()), alg.subtraction(alg.onemat(z.size(), z[0].size()), alg.hadamard_product(z, z)));
	}
	return alg.scalarMultiply(0.5, alg.log(alg.elementWiseDivision(alg.addition(alg.onemat(z.size(), z[0].size()), z), alg.subtraction(z, alg.onemat(z.size(), z[0].size())))));
}

// TO DO: Implement this template activation
std::vector<real_t> MLPPActivationOld::activation(std::vector<real_t> z, bool deriv, real_t (*function)(real_t, bool)) {
	if (deriv) {
		std::vector<real_t> lderiv;
		lderiv.resize(z.size());
		for (uint32_t i = 0; i < z.size(); i++) {
			lderiv[i] = function(z[i], true);
		}
		return lderiv;
	}
	std::vector<real_t> a;
	a.resize(z.size());
	for (uint32_t i = 0; i < z.size(); i++) {
		a[i] = function(z[i], deriv);
	}
	return a;
}
