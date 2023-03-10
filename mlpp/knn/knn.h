
#ifndef MLPP_KNN_H
#define MLPP_KNN_H

//
//  kNN.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPKNN : public Reference {
	GDCLASS(MLPPKNN, Reference);

public:
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output_set();
	void set_output_set(const Ref<MLPPVector> &val);

	int get_k();
	void set_k(const int val);

	PoolIntArray model_set_test(const Ref<MLPPMatrix> &X);
	int model_test(const Ref<MLPPVector> &x);
	real_t score();

	MLPPKNN();
	~MLPPKNN();

protected:
	// Private Model Functions
	PoolIntArray nearest_neighbors(const Ref<MLPPVector> &x);
	int determine_class(const PoolIntArray &knn);

	static void _bind_methods();

	// Model Inputs and Parameters
	Ref<MLPPMatrix> _input_set;
	Ref<MLPPVector> _output_set;
	int _k;
};

#endif /* kNN_hpp */
