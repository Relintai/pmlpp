
#ifndef MLPP_ACTIVATION_H
#define MLPP_ACTIVATION_H

//
//  Activation.hpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "core/math/math_defs.h"

#include <vector>

class MLPPActivation {
public:
	real_t linear(real_t z, bool deriv = 0);
	std::vector<real_t> linear(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> linear(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t sigmoid(real_t z, bool deriv = 0);
	std::vector<real_t> sigmoid(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> sigmoid(std::vector<std::vector<real_t>> z, bool deriv = 0);

	std::vector<real_t> softmax(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> softmax(std::vector<std::vector<real_t>> z, bool deriv = 0);

	std::vector<real_t> adjSoftmax(std::vector<real_t> z);
	std::vector<std::vector<real_t>> adjSoftmax(std::vector<std::vector<real_t>> z);

	std::vector<std::vector<real_t>> softmaxDeriv(std::vector<real_t> z);
	std::vector<std::vector<std::vector<real_t>>> softmaxDeriv(std::vector<std::vector<real_t>> z);

	real_t softplus(real_t z, bool deriv = 0);
	std::vector<real_t> softplus(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> softplus(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t softsign(real_t z, bool deriv = 0);
	std::vector<real_t> softsign(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> softsign(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t gaussianCDF(real_t z, bool deriv = 0);
	std::vector<real_t> gaussianCDF(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> gaussianCDF(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t cloglog(real_t z, bool deriv = 0);
	std::vector<real_t> cloglog(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> cloglog(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t logit(real_t z, bool deriv = 0);
	std::vector<real_t> logit(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> logit(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t unitStep(real_t z, bool deriv = 0);
	std::vector<real_t> unitStep(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> unitStep(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t swish(real_t z, bool deriv = 0);
	std::vector<real_t> swish(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> swish(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t mish(real_t z, bool deriv = 0);
	std::vector<real_t> mish(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> mish(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t sinc(real_t z, bool deriv = 0);
	std::vector<real_t> sinc(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> sinc(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t RELU(real_t z, bool deriv = 0);
	std::vector<real_t> RELU(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> RELU(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t leakyReLU(real_t z, real_t c, bool deriv = 0);
	std::vector<real_t> leakyReLU(std::vector<real_t> z, real_t c, bool deriv = 0);
	std::vector<std::vector<real_t>> leakyReLU(std::vector<std::vector<real_t>> z, real_t c, bool deriv = 0);

	real_t ELU(real_t z, real_t c, bool deriv = 0);
	std::vector<real_t> ELU(std::vector<real_t> z, real_t c, bool deriv = 0);
	std::vector<std::vector<real_t>> ELU(std::vector<std::vector<real_t>> z, real_t c, bool deriv = 0);

	real_t SELU(real_t z, real_t lambda, real_t c, bool deriv = 0);
	std::vector<real_t> SELU(std::vector<real_t> z, real_t lambda, real_t c, bool deriv = 0);
	std::vector<std::vector<real_t>> SELU(std::vector<std::vector<real_t>>, real_t lambda, real_t c, bool deriv = 0);

	real_t GELU(real_t z, bool deriv = 0);
	std::vector<real_t> GELU(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> GELU(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t sign(real_t z, bool deriv = 0);
	std::vector<real_t> sign(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> sign(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t sinh(real_t z, bool deriv = 0);
	std::vector<real_t> sinh(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> sinh(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t cosh(real_t z, bool deriv = 0);
	std::vector<real_t> cosh(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> cosh(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t tanh(real_t z, bool deriv = 0);
	std::vector<real_t> tanh(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> tanh(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t csch(real_t z, bool deriv = 0);
	std::vector<real_t> csch(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> csch(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t sech(real_t z, bool deriv = 0);
	std::vector<real_t> sech(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> sech(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t coth(real_t z, bool deriv = 0);
	std::vector<real_t> coth(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> coth(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t arsinh(real_t z, bool deriv = 0);
	std::vector<real_t> arsinh(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> arsinh(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t arcosh(real_t z, bool deriv = 0);
	std::vector<real_t> arcosh(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> arcosh(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t artanh(real_t z, bool deriv = 0);
	std::vector<real_t> artanh(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> artanh(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t arcsch(real_t z, bool deriv = 0);
	std::vector<real_t> arcsch(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> arcsch(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t arsech(real_t z, bool deriv = 0);
	std::vector<real_t> arsech(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> arsech(std::vector<std::vector<real_t>> z, bool deriv = 0);

	real_t arcoth(real_t z, bool deriv = 0);
	std::vector<real_t> arcoth(std::vector<real_t> z, bool deriv = 0);
	std::vector<std::vector<real_t>> arcoth(std::vector<std::vector<real_t>> z, bool deriv = 0);

	std::vector<real_t> activation(std::vector<real_t> z, bool deriv, real_t (*function)(real_t, bool));

private:
};

#endif /* Activation_hpp */
