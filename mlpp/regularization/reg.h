

#ifndef MLPP_REG_H
#define MLPP_REG_H

//
//  Reg.hpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <string>
#include <vector>

class MLPPReg : public Reference {
	GDCLASS(MLPPReg, Reference);

public:
	enum RegularizationType {
		REGULARIZATION_TYPE_NONE = 0,
		REGULARIZATION_TYPE_RIDGE,
		REGULARIZATION_TYPE_LASSO,
		REGULARIZATION_TYPE_ELASTIC_NET,
		REGULARIZATION_TYPE_WEIGHT_CLIPPING,
	};

	real_t reg_termv(const Ref<MLPPVector> &weights, real_t lambda, real_t alpha, RegularizationType reg);
	real_t reg_termm(const Ref<MLPPMatrix> &weights, real_t lambda, real_t alpha, RegularizationType reg);

	Ref<MLPPVector> reg_weightsv(const Ref<MLPPVector> &weights, real_t lambda, real_t alpha, RegularizationType reg);
	Ref<MLPPMatrix> reg_weightsm(const Ref<MLPPMatrix> &weights, real_t lambda, real_t alpha, RegularizationType reg);

	Ref<MLPPVector> reg_deriv_termv(const Ref<MLPPVector> &weights, real_t lambda, real_t alpha, RegularizationType reg);
	Ref<MLPPMatrix> reg_deriv_termm(const Ref<MLPPMatrix> &weights, real_t lambda, real_t alpha, RegularizationType reg);

	MLPPReg();
	~MLPPReg();

protected:
	static void _bind_methods();

private:
	real_t reg_deriv_termvr(const Ref<MLPPVector> &weights, real_t lambda, real_t alpha, RegularizationType reg, int j);
	real_t reg_deriv_termmr(const Ref<MLPPMatrix> &weights, real_t lambda, real_t alpha, RegularizationType reg, int i, int j);

public:
	// ======== OLD =========

	real_t regTerm(std::vector<real_t> weights, real_t lambda, real_t alpha, std::string reg);
	real_t regTerm(std::vector<std::vector<real_t>> weights, real_t lambda, real_t alpha, std::string reg);

	std::vector<real_t> regWeights(std::vector<real_t> weights, real_t lambda, real_t alpha, std::string reg);
	std::vector<std::vector<real_t>> regWeights(std::vector<std::vector<real_t>> weights, real_t lambda, real_t alpha, std::string reg);

	std::vector<real_t> regDerivTerm(std::vector<real_t> weights, real_t lambda, real_t alpha, std::string reg);
	std::vector<std::vector<real_t>> regDerivTerm(std::vector<std::vector<real_t>>, real_t lambda, real_t alpha, std::string reg);

private:
	real_t regDerivTerm(std::vector<real_t> weights, real_t lambda, real_t alpha, std::string reg, int j);
	real_t regDerivTerm(std::vector<std::vector<real_t>> weights, real_t lambda, real_t alpha, std::string reg, int i, int j);
};

VARIANT_ENUM_CAST(MLPPReg::RegularizationType);

#endif /* Reg_hpp */
