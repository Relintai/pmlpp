
#ifndef MLPP_CONVOLUTIONS_OLD_H
#define MLPP_CONVOLUTIONS_OLD_H

#include <string>
#include <vector>

#include "core/math/math_defs.h"

#include "core/object/reference.h"

class MLPPConvolutionsOld : public Reference {
	GDCLASS(MLPPConvolutionsOld, Reference);

public:
	std::vector<std::vector<real_t>> convolve_2d(std::vector<std::vector<real_t>> input, std::vector<std::vector<real_t>> filter, int S, int P = 0);
	std::vector<std::vector<std::vector<real_t>>> convolve_3d(std::vector<std::vector<std::vector<real_t>>> input, std::vector<std::vector<std::vector<real_t>>> filter, int S, int P = 0);

	std::vector<std::vector<real_t>> pool_2d(std::vector<std::vector<real_t>> input, int F, int S, std::string type);
	std::vector<std::vector<std::vector<real_t>>> pool_3d(std::vector<std::vector<std::vector<real_t>>> input, int F, int S, std::string type);

	real_t global_pool_2d(std::vector<std::vector<real_t>> input, std::string type);
	std::vector<real_t> global_pool_3d(std::vector<std::vector<std::vector<real_t>>> input, std::string type);

	real_t gaussian_2d(real_t x, real_t y, real_t std);
	std::vector<std::vector<real_t>> gaussian_filter_2d(int size, real_t std);

	std::vector<std::vector<real_t>> dx(std::vector<std::vector<real_t>> input);
	std::vector<std::vector<real_t>> dy(std::vector<std::vector<real_t>> input);

	std::vector<std::vector<real_t>> grad_magnitude(std::vector<std::vector<real_t>> input);
	std::vector<std::vector<real_t>> grad_orientation(std::vector<std::vector<real_t>> input);

	std::vector<std::vector<std::vector<real_t>>> compute_m(std::vector<std::vector<real_t>> input);
	std::vector<std::vector<std::string>> harris_corner_detection(std::vector<std::vector<real_t>> input);

	std::vector<std::vector<real_t>> get_prewitt_horizontal();
	std::vector<std::vector<real_t>> get_prewitt_vertical();
	std::vector<std::vector<real_t>> get_sobel_horizontal();
	std::vector<std::vector<real_t>> get_sobel_vertical();
	std::vector<std::vector<real_t>> get_scharr_horizontal();
	std::vector<std::vector<real_t>> get_scharr_vertical();
	std::vector<std::vector<real_t>> get_roberts_horizontal();
	std::vector<std::vector<real_t>> get_roberts_vertical();

	MLPPConvolutionsOld();

protected:
	static void _bind_methods();

	std::vector<std::vector<real_t>> _prewitt_horizontal;
	std::vector<std::vector<real_t>> _prewitt_vertical;
	std::vector<std::vector<real_t>> _sobel_horizontal;
	std::vector<std::vector<real_t>> _sobel_vertical;
	std::vector<std::vector<real_t>> _scharr_horizontal;
	std::vector<std::vector<real_t>> _scharr_vertical;
	std::vector<std::vector<real_t>> _roberts_horizontal;
	std::vector<std::vector<real_t>> _roberts_vertical;
};

#endif // Convolutions_hpp