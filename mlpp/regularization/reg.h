

#ifndef MLPP_REG_H
#define MLPP_REG_H



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
};

VARIANT_ENUM_CAST(MLPPReg::RegularizationType);

#endif /* Reg_hpp */
