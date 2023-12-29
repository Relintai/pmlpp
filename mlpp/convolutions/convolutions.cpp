/*************************************************************************/
/*  convolutions.cpp                                                     */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2022-present Péter Magyar.                              */
/* Copyright (c) 2022-2023 Marc Melikyan                                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "convolutions.h"
#include "../lin_alg/lin_alg.h"
#include "../stat/stat.h"
#include "core/math/math_funcs.h"

#include <cmath>

Ref<MLPPMatrix> MLPPConvolutions::convolve_2d(const Ref<MLPPMatrix> &p_input, const Ref<MLPPMatrix> &filter, const int S, const int P) {
	MLPPLinAlg alg;

	Ref<MLPPMatrix> input = p_input;

	Size2i input_size = input->size();
	int N = input_size.y;
	int F = filter->size().y;
	int map_size = (N - F + 2 * P) / S + 1; // This is computed as ⌊map_size⌋ by def- thanks C++!

	if (P != 0) {
		Ref<MLPPMatrix> padded_input;
		padded_input.instance();

		Size2i pis = Size2i(N + 2 * P, N + 2 * P);

		padded_input->resize(pis);

		for (int i = 0; i < pis.y; i++) {
			for (int j = 0; j < pis.x; j++) {
				if (i - P < 0 || j - P < 0 || i - P > input_size.y - 1 || j - P > input_size.x - 1) {
					padded_input->element_set(i, j, 0);
				} else {
					padded_input->element_set(i, j, input->element_get(i - P, j - P));
				}
			}
		}

		input = padded_input;
	}

	Ref<MLPPMatrix> feature_map;
	feature_map.instance();

	feature_map->resize(Size2i(map_size, map_size));

	Ref<MLPPVector> filter_flattened = filter->flatten();

	Ref<MLPPVector> convolving_input;
	convolving_input.instance();
	convolving_input->resize(F * F);

	for (int i = 0; i < map_size; i++) {
		for (int j = 0; j < map_size; j++) {
			int current_index = 0;

			for (int k = 0; k < F; k++) {
				for (int p = 0; p < F; p++) {
					real_t val;

					if (i == 0 && j == 0) {
						val = input->element_get(i + k, j + p);
					} else if (i == 0) {
						val = input->element_get(i + k, j + (S - 1) + p);
					} else if (j == 0) {
						val = input->element_get(i + (S - 1) + k, j + p);
					} else {
						val = input->element_get(i + (S - 1) + k, j + (S - 1) + p);
					}

					convolving_input->element_set(current_index, val);
					++current_index;
				}
			}

			feature_map->element_set(i, j, convolving_input->dot(filter_flattened));
		}
	}

	return feature_map;
}

Ref<MLPPTensor3> MLPPConvolutions::convolve_3d(const Ref<MLPPTensor3> &p_input, const Ref<MLPPTensor3> &filter, const int S, const int P) {
	MLPPLinAlg alg;

	Ref<MLPPTensor3> input = p_input;

	Size3i input_size = input->size();
	Size3i filter_size = filter->size();

	int N = input_size.y;
	int F = filter_size.y;
	int C = filter_size.z / input_size.z;
	int map_size = (N - F + 2 * P) / S + 1; // This is computed as ⌊map_size⌋ by def.

	if (P != 0) {
		Ref<MLPPTensor3> padded_input;
		padded_input.instance();

		Ref<MLPPMatrix> padded_input_slice;
		padded_input_slice.instance();

		Size2i padded_input_slice_size = Size2i(N + 2 * P, N + 2 * P);
		padded_input_slice->resize(padded_input_slice_size);

		padded_input->resize(Size3i(padded_input_slice_size.x, padded_input_slice_size.y, input_size.z));

		for (int c = 0; c < input_size.z; c++) {
			for (int i = 0; i < padded_input_slice_size.y; i++) {
				for (int j = 0; j < padded_input_slice_size.x; j++) {
					if (i - P < 0 || j - P < 0 || i - P > input_size.y - 1 || j - P > input_size.x - 1) {
						padded_input_slice->element_set(i, j, 0);
					} else {
						padded_input_slice->element_set(i, j, input->element_get(i - P, j - P, c));
					}
				}
			}

			padded_input->z_slice_set_mlpp_matrix(c, padded_input_slice);
		}

		input = padded_input;
	}

	Ref<MLPPTensor3> feature_map;
	feature_map.instance();
	feature_map->resize(Size3i(map_size, map_size, C));

	Ref<MLPPVector> filter_flattened = filter->flatten();

	Ref<MLPPVector> convolving_input;
	convolving_input.instance();
	convolving_input->resize(input_size.z * F * F);

	for (int c = 0; c < C; c++) {
		for (int i = 0; i < map_size; i++) {
			for (int j = 0; j < map_size; j++) {
				int current_index = 0;

				for (int t = 0; t < input_size.z; t++) {
					for (int k = 0; k < F; k++) {
						for (int p = 0; p < F; p++) {
							real_t val;

							if (i == 0 && j == 0) {
								val = input->element_get(i + k, j + p, t);
							} else if (i == 0) {
								val = input->element_get(i + k, j + (S - 1) + p, t);
							} else if (j == 0) {
								val = input->element_get(i + (S - 1) + k, j + p, t);
							} else {
								val = input->element_get(i + (S - 1) + k, j + (S - 1) + p, t);
							}

							convolving_input->element_set(current_index, val);
							++current_index;
						}
					}
				}

				feature_map->element_set(c, i, j, convolving_input->dot(filter_flattened));
			}
		}
	}
	return feature_map;
}

Ref<MLPPMatrix> MLPPConvolutions::pool_2d(const Ref<MLPPMatrix> &input, const int F, const int S, const PoolType type) {
	MLPPLinAlg alg;

	Size2i input_size = input->size();

	int N = input_size.y;
	int map_size = (N - F) / S + 1;

	Ref<MLPPMatrix> pooled_map;
	pooled_map.instance();
	pooled_map->resize(Size2i(map_size, map_size));

	Ref<MLPPVector> pooling_input;
	pooling_input.instance();
	pooling_input->resize(F * F);

	for (int i = 0; i < map_size; i++) {
		for (int j = 0; j < map_size; j++) {
			int current_index = 0;

			for (int k = 0; k < F; k++) {
				for (int p = 0; p < F; p++) {
					real_t val;

					if (i == 0 && j == 0) {
						val = input->element_get(i + k, j + p);
					} else if (i == 0) {
						val = input->element_get(i + k, j + (S - 1) + p);
					} else if (j == 0) {
						val = input->element_get(i + (S - 1) + k, j + p);
					} else {
						val = input->element_get(i + (S - 1) + k, j + (S - 1) + p);
					}

					pooling_input->element_set(current_index, val);
					++current_index;
				}
			}

			if (type == POOL_TYPE_AVERAGE) {
				MLPPStat stat;
				pooled_map->element_set(i, j, stat.meanv(pooling_input));
			} else if (type == POOL_TYPE_MIN) {
				pooled_map->element_set(i, j, alg.minvr(pooling_input));
			} else {
				pooled_map->element_set(i, j, alg.maxvr(pooling_input));
			}
		}
	}

	return pooled_map;
}

Ref<MLPPTensor3> MLPPConvolutions::pool_3d(const Ref<MLPPTensor3> &input, const int F, const int S, const PoolType type) {
	Size3i input_size = input->size();

	Ref<MLPPMatrix> z_slice;
	z_slice.instance();
	z_slice->resize(Size2i(input_size.x, input_size.y));

	int N = input_size.y;
	int map_size = (N - F) / S + 1;

	Ref<MLPPTensor3> pooled_map;
	pooled_map.instance();
	pooled_map->resize(Size3i(map_size, map_size, input_size.z));

	for (int i = 0; i < input_size.z; i++) {
		input->z_slice_get_into_mlpp_matrix(i, z_slice);

		Ref<MLPPMatrix> p = pool_2d(z_slice, F, S, type);

		pooled_map->z_slice_set_mlpp_matrix(i, p);
	}

	return pooled_map;
}

real_t MLPPConvolutions::global_pool_2d(const Ref<MLPPMatrix> &input, const PoolType type) {
	MLPPLinAlg alg;

	Ref<MLPPVector> f = input->flatten();

	if (type == POOL_TYPE_AVERAGE) {
		MLPPStat stat;
		return stat.meanv(f);
	} else if (type == POOL_TYPE_MIN) {
		return alg.minvr(f);
	} else {
		return alg.maxvr(f);
	}
}

Ref<MLPPVector> MLPPConvolutions::global_pool_3d(const Ref<MLPPTensor3> &input, const PoolType type) {
	Size3i input_size = input->size();

	Ref<MLPPVector> pooled_map;
	pooled_map.instance();
	pooled_map->resize(input_size.z);

	Ref<MLPPMatrix> z_slice;
	z_slice.instance();
	z_slice->resize(Size2i(input_size.x, input_size.y));

	for (int i = 0; i < input_size.z; i++) {
		input->z_slice_get_into_mlpp_matrix(i, z_slice);

		pooled_map->element_set(i, global_pool_2d(z_slice, type));
	}

	return pooled_map;
}

real_t MLPPConvolutions::gaussian_2d(const real_t x, const real_t y, const real_t std) {
	real_t std_sq = std * std;
	return 1 / (2 * Math_PI * std_sq) * Math::exp(-(x * x + y * y) / 2 * std_sq);
}

Ref<MLPPMatrix> MLPPConvolutions::gaussian_filter_2d(const int size, const real_t std) {
	Ref<MLPPMatrix> filter;
	filter.instance();
	filter->resize(Size2i(size, size));

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			real_t val = gaussian_2d(i - (size - 1) / 2, (size - 1) / 2 - j, std);

			filter->element_set(i, j, val);
		}
	}

	return filter;
}

// Indeed a filter could have been used for this purpose, but I decided that it would've just
// been easier to carry out the calculation explicitly, mainly because it is more informative,
// and also because my convolution algorithm is only built for filters with equally sized
// heights and widths.

Ref<MLPPMatrix> MLPPConvolutions::dx(const Ref<MLPPMatrix> &input) {
	Size2i input_size = input->size();

	Ref<MLPPMatrix> deriv; // We assume a gray scale image.
	deriv.instance();
	deriv->resize(input_size);

	for (int i = 0; i < input_size.y; i++) {
		for (int j = 0; j < input_size.x; j++) {
			if (j != 0 && j != input_size.y - 1) {
				deriv->element_set(i, j, input->element_get(i, j + 1) - input->element_get(i, j - 1));
			} else if (j == 0) {
				deriv->element_set(i, j, input->element_get(i, j + 1)); // E0 - 0 = Implicit zero-padding
			} else {
				deriv->element_set(i, j, -input->element_get(i, j - 1)); // 0 - E1 = Implicit zero-padding
			}
		}
	}

	return deriv;
}

Ref<MLPPMatrix> MLPPConvolutions::dy(const Ref<MLPPMatrix> &input) {
	Size2i input_size = input->size();

	Ref<MLPPMatrix> deriv; // We assume a gray scale image.
	deriv.instance();
	deriv->resize(input_size);

	for (int i = 0; i < input_size.y; i++) {
		for (int j = 0; j < input_size.x; j++) {
			if (i != 0 && i != input_size.y - 1) {
				deriv->element_set(i, j, input->element_get(i - 1, j) - input->element_get(i + 1, j));
			} else if (i == 0) {
				deriv->element_set(i, j, -input->element_get(i + 1, j)); // 0 - E1 = Implicit zero-padding
			} else {
				deriv->element_set(i, j, input->element_get(i - 1, j)); // E0 - 0 =Implicit zero-padding
			}
		}
	}

	return deriv;
}

Ref<MLPPMatrix> MLPPConvolutions::grad_magnitude(const Ref<MLPPMatrix> &input) {
	MLPPLinAlg alg;

	Ref<MLPPMatrix> x_deriv_2 = dx(input)->hadamard_productn(dx(input));
	Ref<MLPPMatrix> y_deriv_2 = dy(input)->hadamard_productn(dy(input));

	return x_deriv_2->addn(y_deriv_2)->sqrtn();
}

Ref<MLPPMatrix> MLPPConvolutions::grad_orientation(const Ref<MLPPMatrix> &input) {
	Ref<MLPPMatrix> deriv; // We assume a gray scale image.
	deriv.instance();
	deriv->resize(input->size());

	Size2i deriv_size = deriv->size();

	Ref<MLPPMatrix> x_deriv = dx(input);
	Ref<MLPPMatrix> y_deriv = dy(input);

	for (int i = 0; i < deriv_size.y; i++) {
		for (int j = 0; j < deriv_size.x; j++) {
			deriv->element_set(i, j, Math::atan2(y_deriv->element_get(i, j), x_deriv->element_get(i, j)));
		}
	}

	return deriv;
}

Ref<MLPPTensor3> MLPPConvolutions::compute_m(const Ref<MLPPMatrix> &input) {
	Size2i input_size = input->size();

	real_t const SIGMA = 1;
	real_t const GAUSSIAN_SIZE = 3;

	real_t const GAUSSIAN_PADDING = ((input_size.y - 1) + GAUSSIAN_SIZE - input_size.y) / 2; // Convs must be same.

	Ref<MLPPMatrix> x_deriv = dx(input);
	Ref<MLPPMatrix> y_deriv = dy(input);

	Ref<MLPPMatrix> gaussian_filter = gaussian_filter_2d(GAUSSIAN_SIZE, SIGMA); // Sigma of 1, size of 3.
	Ref<MLPPMatrix> xx_deriv = convolve_2d(x_deriv->hadamard_productn(x_deriv), gaussian_filter, 1, GAUSSIAN_PADDING);
	Ref<MLPPMatrix> yy_deriv = convolve_2d(y_deriv->hadamard_productn(y_deriv), gaussian_filter, 1, GAUSSIAN_PADDING);
	Ref<MLPPMatrix> xy_deriv = convolve_2d(x_deriv->hadamard_productn(y_deriv), gaussian_filter, 1, GAUSSIAN_PADDING);

	Size2i ds = xx_deriv->size();

	Ref<MLPPTensor3> M;
	M.instance();
	M->resize(Size3i(ds.x, ds.y, 3));

	M->z_slice_set_mlpp_matrix(0, xx_deriv);
	M->z_slice_set_mlpp_matrix(1, yy_deriv);
	M->z_slice_set_mlpp_matrix(2, xy_deriv);

	return M;
}

Vector<Ref<MLPPMatrix>> MLPPConvolutions::compute_mv(const Ref<MLPPMatrix> &input) {
	Size2i input_size = input->size();

	real_t const SIGMA = 1;
	real_t const GAUSSIAN_SIZE = 3;

	real_t const GAUSSIAN_PADDING = ((input_size.y - 1) + GAUSSIAN_SIZE - input_size.y) / 2; // Convs must be same.

	Ref<MLPPMatrix> x_deriv = dx(input);
	Ref<MLPPMatrix> y_deriv = dy(input);

	Ref<MLPPMatrix> gaussian_filter = gaussian_filter_2d(GAUSSIAN_SIZE, SIGMA); // Sigma of 1, size of 3.
	Ref<MLPPMatrix> xx_deriv = convolve_2d(x_deriv->hadamard_productn(x_deriv), gaussian_filter, 1, GAUSSIAN_PADDING);
	Ref<MLPPMatrix> yy_deriv = convolve_2d(y_deriv->hadamard_productn(y_deriv), gaussian_filter, 1, GAUSSIAN_PADDING);
	Ref<MLPPMatrix> xy_deriv = convolve_2d(x_deriv->hadamard_productn(y_deriv), gaussian_filter, 1, GAUSSIAN_PADDING);

	Vector<Ref<MLPPMatrix>> M;
	M.resize(3);

	M.set(0, xx_deriv);
	M.set(1, yy_deriv);
	M.set(2, xy_deriv);

	return M;
}

Vector<Vector<CharType>> MLPPConvolutions::harris_corner_detection(const Ref<MLPPMatrix> &input) {
	real_t const k = 0.05; // Empirically determined wherein k -> [0.04, 0.06], though conventionally 0.05 is typically used as well.

	Vector<Ref<MLPPMatrix>> M = compute_mv(input);

	Ref<MLPPMatrix> M0 = M[0];
	Ref<MLPPMatrix> M1 = M[1];
	Ref<MLPPMatrix> M2 = M[2];

	Ref<MLPPMatrix> det = M0->hadamard_productn(M1)->subn(M2->hadamard_productn(M2));
	Ref<MLPPMatrix> trace = M0->addn(M1);

	// The reason this is not a scalar is because xx_deriv, xy_deriv, yx_deriv, and yy_deriv are not scalars.
	Ref<MLPPMatrix> r = det->subn(trace->hadamard_productn(trace)->scalar_multiplyn(k));
	Size2i r_size = r->size();

	Vector<Vector<CharType>> image_types;
	image_types.resize(r_size.y);
	//alg.printMatrix(r);

	for (int i = 0; i < r_size.y; i++) {
		image_types.write[i].resize(r_size.x);

		for (int j = 0; j < r_size.x; j++) {
			real_t e = r->element_get(i, j);

			if (e > 0) {
				image_types.write[i].write[j] = 'C';
			} else if (e < 0) {
				image_types.write[i].write[j] = 'E';
			} else {
				image_types.write[i].write[j] = 'N';
			}
		}
	}

	return image_types;
}

Ref<MLPPMatrix> MLPPConvolutions::get_prewitt_horizontal() const {
	return _prewitt_horizontal;
}
Ref<MLPPMatrix> MLPPConvolutions::get_prewitt_vertical() const {
	return _prewitt_vertical;
}
Ref<MLPPMatrix> MLPPConvolutions::get_sobel_horizontal() const {
	return _sobel_horizontal;
}
Ref<MLPPMatrix> MLPPConvolutions::get_sobel_vertical() const {
	return _sobel_vertical;
}
Ref<MLPPMatrix> MLPPConvolutions::get_scharr_horizontal() const {
	return _scharr_horizontal;
}
Ref<MLPPMatrix> MLPPConvolutions::get_scharr_vertical() const {
	return _scharr_vertical;
}
Ref<MLPPMatrix> MLPPConvolutions::get_roberts_horizontal() const {
	return _roberts_horizontal;
}
Ref<MLPPMatrix> MLPPConvolutions::get_roberts_vertical() const {
	return _roberts_vertical;
}

MLPPConvolutions::MLPPConvolutions() {
	const real_t prewitt_horizontal_arr[]{
		1, 1, 1, //
		0, 0, 0, //
		-1, -1, -1, //
	};
	const real_t prewitt_vertical_arr[] = {
		1, 0, -1, //
		1, 0, -1, //
		1, 0, -1 //
	};
	const real_t sobel_horizontal_arr[] = {
		1, 2, 1, //
		0, 0, 0, //
		-1, -2, -1 //
	};
	const real_t sobel_vertical_arr[] = {
		-1, 0, 1, //
		-2, 0, 2, //
		-1, 0, 1 //
	};
	const real_t scharr_horizontal_arr[] = {
		3, 10, 3, //
		0, 0, 0, //
		-3, -10, -3 //
	};
	const real_t scharr_vertical_arr[] = {
		3, 0, -3, //
		10, 0, -10, //
		3, 0, -3 //
	};
	const real_t roberts_horizontal_arr[] = {
		0, 1, //
		-1, 0 //
	};
	const real_t roberts_vertical_arr[] = {
		1, 0, //
		0, -1 //
	};

	_prewitt_horizontal = Ref<MLPPMatrix>(memnew(MLPPMatrix(prewitt_horizontal_arr, 3, 3)));
	_prewitt_vertical = Ref<MLPPMatrix>(memnew(MLPPMatrix(prewitt_vertical_arr, 3, 3)));
	_sobel_horizontal = Ref<MLPPMatrix>(memnew(MLPPMatrix(sobel_horizontal_arr, 3, 3)));
	_sobel_vertical = Ref<MLPPMatrix>(memnew(MLPPMatrix(sobel_vertical_arr, 3, 3)));
	_scharr_horizontal = Ref<MLPPMatrix>(memnew(MLPPMatrix(scharr_horizontal_arr, 3, 3)));
	_scharr_vertical = Ref<MLPPMatrix>(memnew(MLPPMatrix(scharr_vertical_arr, 3, 3)));
	_roberts_horizontal = Ref<MLPPMatrix>(memnew(MLPPMatrix(roberts_horizontal_arr, 2, 2)));
	_roberts_vertical = Ref<MLPPMatrix>(memnew(MLPPMatrix(roberts_vertical_arr, 2, 2)));
}

void MLPPConvolutions::_bind_methods() {
}
