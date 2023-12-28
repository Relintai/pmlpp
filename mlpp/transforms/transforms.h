
#ifndef MLPP_TRANSFORMS_H
#define MLPP_TRANSFORMS_H

//
//  Transforms.hpp
//
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"

class MLPPTransforms : public Reference {
	GDCLASS(MLPPTransforms, Reference);

public:
	Ref<MLPPMatrix> discrete_cosine_transform(const Ref<MLPPMatrix> &p_A);

protected:
	static void _bind_methods();
};

#endif /* Transforms_hpp */
