
#ifndef MLPP_UNI_LIN_REG_H
#define MLPP_UNI_LIN_REG_H

//
//  UniLinReg.hpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPUniLinReg : public Reference {
	GDCLASS(MLPPUniLinReg, Reference);

public:
	Ref<MLPPVector> get_input_set() const;
	void set_input_set(const Ref<MLPPVector> &val);

	Ref<MLPPVector> get_output_set() const;
	void set_output_set(const Ref<MLPPVector> &val);

	real_t get_b0() const;
	void set_b0(const real_t val);

	real_t get_b1() const;
	void set_b1(const real_t val);

	void fit();

	Ref<MLPPVector> model_set_test(const Ref<MLPPVector> &x);
	real_t model_test(real_t x);

	MLPPUniLinReg(const Ref<MLPPVector> &p_input_set, const Ref<MLPPVector> &p_output_set);

	MLPPUniLinReg();
	~MLPPUniLinReg();

protected:
	static void _bind_methods();

	Ref<MLPPVector> _input_set;
	Ref<MLPPVector> _output_set;

	real_t _b0;
	real_t _b1;
};

#endif /* UniLinReg_hpp */
