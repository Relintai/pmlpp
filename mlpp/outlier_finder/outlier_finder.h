
#ifndef MLPP_OUTLIER_FINDER_H
#define MLPP_OUTLIER_FINDER_H

//
//  OutlierFinder.hpp
//
//  Created by Marc Melikyan on 11/13/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPOutlierFinder : public Reference {
	GDCLASS(MLPPOutlierFinder, Reference);

public:
	real_t get_threshold();
	void set_threshold(real_t val);

	Vector<Vector<real_t>> model_set_test(const Ref<MLPPMatrix> &input_set);
	Array model_set_test_bind(const Ref<MLPPMatrix> &input_set);

	PoolVector2iArray model_set_test_indices(const Ref<MLPPMatrix> &input_set);

	PoolRealArray model_test(const Ref<MLPPVector> &input_set);

	MLPPOutlierFinder(real_t threshold);

	MLPPOutlierFinder();
	~MLPPOutlierFinder();

protected:
	static void _bind_methods();

	real_t _threshold;
};

#endif /* OutlierFinder_hpp */
