//
//  Convolutions.cpp
//
//  Created by Marc Melikyan on 4/6/21.
//

#include "../convolutions/convolutions_old.h"

#include "../lin_alg/lin_alg_old.h"
#include "../stat/stat_old.h"
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.141592653
#endif

std::vector<std::vector<real_t>> MLPPConvolutionsOld::convolve_2d(std::vector<std::vector<real_t>> input, std::vector<std::vector<real_t>> filter, int S, int P) {
	MLPPLinAlgOld alg;
	std::vector<std::vector<real_t>> feature_map;
	uint32_t N = input.size();
	uint32_t F = filter.size();
	uint32_t map_size = (N - F + 2 * P) / S + 1; // This is computed as ⌊map_size⌋ by def- thanks C++!

	if (P != 0) {
		std::vector<std::vector<real_t>> padded_input;
		padded_input.resize(N + 2 * P);
		for (uint32_t i = 0; i < padded_input.size(); i++) {
			padded_input[i].resize(N + 2 * P);
		}
		for (uint32_t i = 0; i < padded_input.size(); i++) {
			for (uint32_t j = 0; j < padded_input[i].size(); j++) {
				if (i - P < 0 || j - P < 0 || i - P > input.size() - 1 || j - P > input[0].size() - 1) {
					padded_input[i][j] = 0;
				} else {
					padded_input[i][j] = input[i - P][j - P];
				}
			}
		}
		input.resize(padded_input.size());
		for (uint32_t i = 0; i < padded_input.size(); i++) {
			input[i].resize(padded_input[i].size());
		}
		input = padded_input;
	}

	feature_map.resize(map_size);
	for (uint32_t i = 0; i < map_size; i++) {
		feature_map[i].resize(map_size);
	}

	for (uint32_t i = 0; i < map_size; i++) {
		for (uint32_t j = 0; j < map_size; j++) {
			std::vector<real_t> convolving_input;
			for (uint32_t k = 0; k < F; k++) {
				for (uint32_t p = 0; p < F; p++) {
					if (i == 0 && j == 0) {
						convolving_input.push_back(input[i + k][j + p]);
					} else if (i == 0) {
						convolving_input.push_back(input[i + k][j + (S - 1) + p]);
					} else if (j == 0) {
						convolving_input.push_back(input[i + (S - 1) + k][j + p]);
					} else {
						convolving_input.push_back(input[i + (S - 1) + k][j + (S - 1) + p]);
					}
				}
			}
			feature_map[i][j] = alg.dot(convolving_input, alg.flatten(filter));
		}
	}
	return feature_map;
}

std::vector<std::vector<std::vector<real_t>>> MLPPConvolutionsOld::convolve_3d(std::vector<std::vector<std::vector<real_t>>> input, std::vector<std::vector<std::vector<real_t>>> filter, int S, int P) {
	MLPPLinAlgOld alg;
	std::vector<std::vector<std::vector<real_t>>> feature_map;
	uint32_t N = input[0].size();
	uint32_t F = filter[0].size();
	uint32_t C = filter.size() / input.size();
	uint32_t map_size = (N - F + 2 * P) / S + 1; // This is computed as ⌊map_size⌋ by def.

	if (P != 0) {
		for (uint32_t c = 0; c < input.size(); c++) {
			std::vector<std::vector<real_t>> padded_input;
			padded_input.resize(N + 2 * P);
			for (uint32_t i = 0; i < padded_input.size(); i++) {
				padded_input[i].resize(N + 2 * P);
			}
			for (uint32_t i = 0; i < padded_input.size(); i++) {
				for (uint32_t j = 0; j < padded_input[i].size(); j++) {
					if (i - P < 0 || j - P < 0 || i - P > input[c].size() - 1 || j - P > input[c][0].size() - 1) {
						padded_input[i][j] = 0;
					} else {
						padded_input[i][j] = input[c][i - P][j - P];
					}
				}
			}
			input[c].resize(padded_input.size());
			for (uint32_t i = 0; i < padded_input.size(); i++) {
				input[c][i].resize(padded_input[i].size());
			}
			input[c] = padded_input;
		}
	}

	feature_map.resize(C);
	for (uint32_t i = 0; i < feature_map.size(); i++) {
		feature_map[i].resize(map_size);
		for (uint32_t j = 0; j < feature_map[i].size(); j++) {
			feature_map[i][j].resize(map_size);
		}
	}

	for (uint32_t c = 0; c < C; c++) {
		for (uint32_t i = 0; i < map_size; i++) {
			for (uint32_t j = 0; j < map_size; j++) {
				std::vector<real_t> convolving_input;
				for (uint32_t t = 0; t < input.size(); t++) {
					for (uint32_t k = 0; k < F; k++) {
						for (uint32_t p = 0; p < F; p++) {
							if (i == 0 && j == 0) {
								convolving_input.push_back(input[t][i + k][j + p]);
							} else if (i == 0) {
								convolving_input.push_back(input[t][i + k][j + (S - 1) + p]);
							} else if (j == 0) {
								convolving_input.push_back(input[t][i + (S - 1) + k][j + p]);
							} else {
								convolving_input.push_back(input[t][i + (S - 1) + k][j + (S - 1) + p]);
							}
						}
					}
				}
				feature_map[c][i][j] = alg.dot(convolving_input, alg.flatten(filter));
			}
		}
	}
	return feature_map;
}

std::vector<std::vector<real_t>> MLPPConvolutionsOld::pool_2d(std::vector<std::vector<real_t>> input, int F, int S, std::string type) {
	MLPPLinAlgOld alg;
	std::vector<std::vector<real_t>> pooled_map;
	uint32_t N = input.size();
	uint32_t map_size = floor((N - F) / S + 1);

	pooled_map.resize(map_size);
	for (uint32_t i = 0; i < map_size; i++) {
		pooled_map[i].resize(map_size);
	}

	for (uint32_t i = 0; i < map_size; i++) {
		for (uint32_t j = 0; j < map_size; j++) {
			std::vector<real_t> pooling_input;
			for (int k = 0; k < F; k++) {
				for (int p = 0; p < F; p++) {
					if (i == 0 && j == 0) {
						pooling_input.push_back(input[i + k][j + p]);
					} else if (i == 0) {
						pooling_input.push_back(input[i + k][j + (S - 1) + p]);
					} else if (j == 0) {
						pooling_input.push_back(input[i + (S - 1) + k][j + p]);
					} else {
						pooling_input.push_back(input[i + (S - 1) + k][j + (S - 1) + p]);
					}
				}
			}
			if (type == "Average") {
				MLPPStatOld stat;
				pooled_map[i][j] = stat.mean(pooling_input);
			} else if (type == "Min") {
				pooled_map[i][j] = alg.min(pooling_input);
			} else {
				pooled_map[i][j] = alg.max(pooling_input);
			}
		}
	}
	return pooled_map;
}

std::vector<std::vector<std::vector<real_t>>> MLPPConvolutionsOld::pool_3d(std::vector<std::vector<std::vector<real_t>>> input, int F, int S, std::string type) {
	std::vector<std::vector<std::vector<real_t>>> pooled_map;
	for (uint32_t i = 0; i < input.size(); i++) {
		pooled_map.push_back(pool_2d(input[i], F, S, type));
	}
	return pooled_map;
}

real_t MLPPConvolutionsOld::global_pool_2d(std::vector<std::vector<real_t>> input, std::string type) {
	MLPPLinAlgOld alg;
	if (type == "Average") {
		MLPPStatOld stat;
		return stat.mean(alg.flatten(input));
	} else if (type == "Min") {
		return alg.min(alg.flatten(input));
	} else {
		return alg.max(alg.flatten(input));
	}
}

std::vector<real_t> MLPPConvolutionsOld::global_pool_3d(std::vector<std::vector<std::vector<real_t>>> input, std::string type) {
	std::vector<real_t> pooled_map;
	for (uint32_t i = 0; i < input.size(); i++) {
		pooled_map.push_back(global_pool_2d(input[i], type));
	}
	return pooled_map;
}

real_t MLPPConvolutionsOld::gaussian_2d(real_t x, real_t y, real_t std) {
	real_t std_sq = std * std;
	return 1 / (2 * M_PI * std_sq) * std::exp(-(x * x + y * y) / 2 * std_sq);
}

std::vector<std::vector<real_t>> MLPPConvolutionsOld::gaussian_filter_2d(int size, real_t std) {
	std::vector<std::vector<real_t>> filter;
	filter.resize(size);
	for (uint32_t i = 0; i < filter.size(); i++) {
		filter[i].resize(size);
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			filter[i][j] = gaussian_2d(i - (size - 1) / 2, (size - 1) / 2 - j, std);
		}
	}
	return filter;
}

/*
Indeed a filter could have been used for this purpose, but I decided that it would've just
been easier to carry out the calculation explicitly, mainly because it is more informative,
and also because my convolution algorithm is only built for filters with equally sized
heights and widths.
*/
std::vector<std::vector<real_t>> MLPPConvolutionsOld::dx(std::vector<std::vector<real_t>> input) {
	std::vector<std::vector<real_t>> deriv; // We assume a gray scale image.
	deriv.resize(input.size());
	for (uint32_t i = 0; i < deriv.size(); i++) {
		deriv[i].resize(input[i].size());
	}

	for (uint32_t i = 0; i < input.size(); i++) {
		for (uint32_t j = 0; j < input[i].size(); j++) {
			if (j != 0 && j != input.size() - 1) {
				deriv[i][j] = input[i][j + 1] - input[i][j - 1];
			} else if (j == 0) {
				deriv[i][j] = input[i][j + 1] - 0; // Implicit zero-padding
			} else {
				deriv[i][j] = 0 - input[i][j - 1]; // Implicit zero-padding
			}
		}
	}
	return deriv;
}

std::vector<std::vector<real_t>> MLPPConvolutionsOld::dy(std::vector<std::vector<real_t>> input) {
	std::vector<std::vector<real_t>> deriv;
	deriv.resize(input.size());
	for (uint32_t i = 0; i < deriv.size(); i++) {
		deriv[i].resize(input[i].size());
	}

	for (uint32_t i = 0; i < input.size(); i++) {
		for (uint32_t j = 0; j < input[i].size(); j++) {
			if (i != 0 && i != input.size() - 1) {
				deriv[i][j] = input[i - 1][j] - input[i + 1][j];
			} else if (i == 0) {
				deriv[i][j] = 0 - input[i + 1][j]; // Implicit zero-padding
			} else {
				deriv[i][j] = input[i - 1][j] - 0; // Implicit zero-padding
			}
		}
	}
	return deriv;
}

std::vector<std::vector<real_t>> MLPPConvolutionsOld::grad_magnitude(std::vector<std::vector<real_t>> input) {
	MLPPLinAlgOld alg;
	std::vector<std::vector<real_t>> x_deriv_2 = alg.hadamard_product(dx(input), dx(input));
	std::vector<std::vector<real_t>> y_deriv_2 = alg.hadamard_product(dy(input), dy(input));
	return alg.sqrt(alg.addition(x_deriv_2, y_deriv_2));
}

std::vector<std::vector<real_t>> MLPPConvolutionsOld::grad_orientation(std::vector<std::vector<real_t>> input) {
	std::vector<std::vector<real_t>> deriv;
	deriv.resize(input.size());
	for (uint32_t i = 0; i < deriv.size(); i++) {
		deriv[i].resize(input[i].size());
	}

	std::vector<std::vector<real_t>> x_deriv = dx(input);
	std::vector<std::vector<real_t>> y_deriv = dy(input);
	for (uint32_t i = 0; i < deriv.size(); i++) {
		for (uint32_t j = 0; j < deriv[i].size(); j++) {
			deriv[i][j] = std::atan2(y_deriv[i][j], x_deriv[i][j]);
		}
	}
	return deriv;
}

std::vector<std::vector<std::vector<real_t>>> MLPPConvolutionsOld::compute_m(std::vector<std::vector<real_t>> input) {
	real_t const SIGMA = 1;
	real_t const GAUSSIAN_SIZE = 3;

	real_t const GAUSSIAN_PADDING = ((input.size() - 1) + GAUSSIAN_SIZE - input.size()) / 2; // Convs must be same.
	std::cout << GAUSSIAN_PADDING << std::endl;
	MLPPLinAlgOld alg;
	std::vector<std::vector<real_t>> x_deriv = dx(input);
	std::vector<std::vector<real_t>> y_deriv = dy(input);

	std::vector<std::vector<real_t>> gaussian_filter = gaussian_filter_2d(GAUSSIAN_SIZE, SIGMA); // Sigma of 1, size of 3.
	std::vector<std::vector<real_t>> xx_deriv = convolve_2d(alg.hadamard_product(x_deriv, x_deriv), gaussian_filter, 1, GAUSSIAN_PADDING);
	std::vector<std::vector<real_t>> yy_deriv = convolve_2d(alg.hadamard_product(y_deriv, y_deriv), gaussian_filter, 1, GAUSSIAN_PADDING);
	std::vector<std::vector<real_t>> xy_deriv = convolve_2d(alg.hadamard_product(x_deriv, y_deriv), gaussian_filter, 1, GAUSSIAN_PADDING);

	std::vector<std::vector<std::vector<real_t>>> M = { xx_deriv, yy_deriv, xy_deriv };
	return M;
}
std::vector<std::vector<std::string>> MLPPConvolutionsOld::harris_corner_detection(std::vector<std::vector<real_t>> input) {
	real_t const k = 0.05; // Empirically determined wherein k -> [0.04, 0.06], though conventionally 0.05 is typically used as well.
	MLPPLinAlgOld alg;
	std::vector<std::vector<std::vector<real_t>>> M = compute_m(input);
	std::vector<std::vector<real_t>> det = alg.subtraction(alg.hadamard_product(M[0], M[1]), alg.hadamard_product(M[2], M[2]));
	std::vector<std::vector<real_t>> trace = alg.addition(M[0], M[1]);

	// The reason this is not a scalar is because xx_deriv, xy_deriv, yx_deriv, and yy_deriv are not scalars.
	std::vector<std::vector<real_t>> r = alg.subtraction(det, alg.scalarMultiply(k, alg.hadamard_product(trace, trace)));
	std::vector<std::vector<std::string>> imageTypes;
	imageTypes.resize(r.size());
	alg.printMatrix(r);
	for (uint32_t i = 0; i < r.size(); i++) {
		imageTypes[i].resize(r[i].size());
		for (uint32_t j = 0; j < r[i].size(); j++) {
			if (r[i][j] > 0) {
				imageTypes[i][j] = "C";
			} else if (r[i][j] < 0) {
				imageTypes[i][j] = "E";
			} else {
				imageTypes[i][j] = "N";
			}
		}
	}
	return imageTypes;
}

std::vector<std::vector<real_t>> MLPPConvolutionsOld::get_prewitt_horizontal() {
	return _prewitt_horizontal;
}
std::vector<std::vector<real_t>> MLPPConvolutionsOld::get_prewitt_vertical() {
	return _prewitt_vertical;
}
std::vector<std::vector<real_t>> MLPPConvolutionsOld::get_sobel_horizontal() {
	return _sobel_horizontal;
}
std::vector<std::vector<real_t>> MLPPConvolutionsOld::get_sobel_vertical() {
	return _sobel_vertical;
}
std::vector<std::vector<real_t>> MLPPConvolutionsOld::get_scharr_horizontal() {
	return _scharr_horizontal;
}
std::vector<std::vector<real_t>> MLPPConvolutionsOld::get_scharr_vertical() {
	return _scharr_vertical;
}
std::vector<std::vector<real_t>> MLPPConvolutionsOld::get_roberts_horizontal() {
	return _roberts_horizontal;
}
std::vector<std::vector<real_t>> MLPPConvolutionsOld::get_roberts_vertical() {
	return _roberts_vertical;
}

MLPPConvolutionsOld::MLPPConvolutionsOld() {
	_prewitt_horizontal = { { 1, 1, 1 }, { 0, 0, 0 }, { -1, -1, -1 } };
	_prewitt_vertical = { { 1, 0, -1 }, { 1, 0, -1 }, { 1, 0, -1 } };
	_sobel_horizontal = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
	_sobel_vertical = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	_scharr_horizontal = { { 3, 10, 3 }, { 0, 0, 0 }, { -3, -10, -3 } };
	_scharr_vertical = { { 3, 0, -3 }, { 10, 0, -10 }, { 3, 0, -3 } };
	_roberts_horizontal = { { 0, 1 }, { -1, 0 } };
	_roberts_vertical = { { 1, 0 }, { 0, -1 } };
}
