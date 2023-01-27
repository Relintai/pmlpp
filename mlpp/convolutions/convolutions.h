
#ifndef MLPP_CONVOLUTIONS_H
#define MLPP_CONVOLUTIONS_H

#include <vector>
#include <string>

#include "core/math/math_defs.h"

class MLPPConvolutions {
public:
	MLPPConvolutions();
	std::vector<std::vector<real_t>> convolve(std::vector<std::vector<real_t>> input, std::vector<std::vector<real_t>> filter, int S, int P = 0);
	std::vector<std::vector<std::vector<real_t>>> convolve(std::vector<std::vector<std::vector<real_t>>> input, std::vector<std::vector<std::vector<real_t>>> filter, int S, int P = 0);
	std::vector<std::vector<real_t>> pool(std::vector<std::vector<real_t>> input, int F, int S, std::string type);
	std::vector<std::vector<std::vector<real_t>>> pool(std::vector<std::vector<std::vector<real_t>>> input, int F, int S, std::string type);
	real_t globalPool(std::vector<std::vector<real_t>> input, std::string type);
	std::vector<real_t> globalPool(std::vector<std::vector<std::vector<real_t>>> input, std::string type);

	real_t gaussian2D(real_t x, real_t y, real_t std);
	std::vector<std::vector<real_t>> gaussianFilter2D(int size, real_t std);

	std::vector<std::vector<real_t>> dx(std::vector<std::vector<real_t>> input);
	std::vector<std::vector<real_t>> dy(std::vector<std::vector<real_t>> input);

	std::vector<std::vector<real_t>> gradMagnitude(std::vector<std::vector<real_t>> input);
	std::vector<std::vector<real_t>> gradOrientation(std::vector<std::vector<real_t>> input);

	std::vector<std::vector<std::vector<real_t>>> computeM(std::vector<std::vector<real_t>> input);
	std::vector<std::vector<std::string>> harrisCornerDetection(std::vector<std::vector<real_t>> input);

	std::vector<std::vector<real_t>> getPrewittHorizontal();
	std::vector<std::vector<real_t>> getPrewittVertical();
	std::vector<std::vector<real_t>> getSobelHorizontal();
	std::vector<std::vector<real_t>> getSobelVertical();
	std::vector<std::vector<real_t>> getScharrHorizontal();
	std::vector<std::vector<real_t>> getScharrVertical();
	std::vector<std::vector<real_t>> getRobertsHorizontal();
	std::vector<std::vector<real_t>> getRobertsVertical();

private:
	std::vector<std::vector<real_t>> prewittHorizontal;
	std::vector<std::vector<real_t>> prewittVertical;
	std::vector<std::vector<real_t>> sobelHorizontal;
	std::vector<std::vector<real_t>> sobelVertical;
	std::vector<std::vector<real_t>> scharrHorizontal;
	std::vector<std::vector<real_t>> scharrVertical;
	std::vector<std::vector<real_t>> robertsHorizontal;
	std::vector<std::vector<real_t>> robertsVertical;
};

#endif // Convolutions_hpp