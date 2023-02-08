
#ifndef MLPP_PCA_H
#define MLPP_PCA_H

//
//  PCA.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPPCA : public Reference {
	GDCLASS(MLPPPCA, Reference);

public:
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	int get_k();
	void set_k(const int val);

	Ref<MLPPMatrix> principal_components();
	real_t score();

	MLPPPCA(const Ref<MLPPMatrix> &p_input_set, int p_k);

	MLPPPCA();
	~MLPPPCA();

protected:
	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	int _k;

	Ref<MLPPMatrix> _x_normalized;
	Ref<MLPPMatrix> _u_reduce;
	Ref<MLPPMatrix> _z;
};

#endif /* PCA_hpp */
