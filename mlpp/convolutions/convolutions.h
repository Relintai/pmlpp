
#ifndef MLPP_CONVOLUTIONS_H
#define MLPP_CONVOLUTIONS_H

#include "core/containers/vector.h"
#include "core/string/ustring.h"

#include "core/math/math_defs.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_tensor3.h"
#include "../lin_alg/mlpp_vector.h"

#include "core/object/reference.h"

class MLPPConvolutions : public Reference {
	GDCLASS(MLPPConvolutions, Reference);

public:
	enum PoolType {
		POOL_TYPE_AVERAGE = 0,
		POOL_TYPE_MIN,
		POOL_TYPE_MAX,
	};

	Ref<MLPPMatrix> convolve_2d(const Ref<MLPPMatrix> &input, const Ref<MLPPMatrix> &filter, const int S, const int P = 0);
	Ref<MLPPTensor3> convolve_3d(const Ref<MLPPTensor3> &input, const Ref<MLPPTensor3> &filter, const int S, const int P = 0);

	Ref<MLPPMatrix> pool_2d(const Ref<MLPPMatrix> &input, const int F, const int S, const PoolType type);
	Ref<MLPPTensor3> pool_3d(const Ref<MLPPTensor3> &input, const int F, const int S, const PoolType type);

	real_t global_pool_2d(const Ref<MLPPMatrix> &input, const PoolType type);
	Ref<MLPPVector> global_pool_3d(const Ref<MLPPTensor3> &input, const PoolType type);

	real_t gaussian_2d(const real_t x, const real_t y, const real_t std);
	Ref<MLPPMatrix> gaussian_filter_2d(const int size, const real_t std);

	Ref<MLPPMatrix> dx(const Ref<MLPPMatrix> &input);
	Ref<MLPPMatrix> dy(const Ref<MLPPMatrix> &input);

	Ref<MLPPMatrix> grad_magnitude(const Ref<MLPPMatrix> &input);
	Ref<MLPPMatrix> grad_orientation(const Ref<MLPPMatrix> &input);

	Ref<MLPPTensor3> compute_m(const Ref<MLPPMatrix> &input);
	Vector<Ref<MLPPMatrix>> compute_mv(const Ref<MLPPMatrix> &input);

	//TODO better data srtucture for this. Maybe IntMatrix?
	Vector<Vector<CharType>> harris_corner_detection(const Ref<MLPPMatrix> &input);

	Ref<MLPPMatrix> get_prewitt_horizontal() const;
	Ref<MLPPMatrix> get_prewitt_vertical() const;
	Ref<MLPPMatrix> get_sobel_horizontal() const;
	Ref<MLPPMatrix> get_sobel_vertical() const;
	Ref<MLPPMatrix> get_scharr_horizontal() const;
	Ref<MLPPMatrix> get_scharr_vertical() const;
	Ref<MLPPMatrix> get_roberts_horizontal() const;
	Ref<MLPPMatrix> get_roberts_vertical() const;

	MLPPConvolutions();

protected:
	static void _bind_methods();

	Ref<MLPPMatrix> _prewitt_horizontal;
	Ref<MLPPMatrix> _prewitt_vertical;
	Ref<MLPPMatrix> _sobel_horizontal;
	Ref<MLPPMatrix> _sobel_vertical;
	Ref<MLPPMatrix> _scharr_horizontal;
	Ref<MLPPMatrix> _scharr_vertical;
	Ref<MLPPMatrix> _roberts_horizontal;
	Ref<MLPPMatrix> _roberts_vertical;
};

#endif // Convolutions_hpp