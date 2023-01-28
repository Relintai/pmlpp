//
//  kNN.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "knn.h"
#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

#include "core/containers/hash_map.h"
#include "core/containers/vector.h"

Ref<MLPPMatrix> MLPPKNN::get_input_set() {
	return _input_set;
}
void MLPPKNN::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;
}

Ref<MLPPVector> MLPPKNN::get_output_set() {
	return _output_set;
}
void MLPPKNN::set_output_set(const Ref<MLPPVector> &val) {
	_output_set = val;
}

int MLPPKNN::get_k() {
	return _k;
}
void MLPPKNN::set_k(const int val) {
	_k = val;
}

PoolIntArray MLPPKNN::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(!X.is_valid(), PoolIntArray());

	Ref<MLPPVector> v;
	v.instance();

	int y_size = X->size().y;

	PoolIntArray y_hat;
	y_hat.resize(y_size);

	for (int i = 0; i < y_size; i++) {
		X->get_row_into_mlpp_vector(i, v);

		y_hat.set(i, model_test(v));
	}

	return y_hat;
}

int MLPPKNN::model_test(const Ref<MLPPVector> &x) {
	return determine_class(nearest_neighbors(x));
}

real_t MLPPKNN::score() {
	MLPPUtilities util;
	return util.performance_pool_int_array_vec(model_set_test(_input_set), _output_set);
}

MLPPKNN::MLPPKNN(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, int k) {
	_k = k;
}

MLPPKNN::MLPPKNN() {
	_k = 0;
}

MLPPKNN::~MLPPKNN() {
}

// Private Model Functions
PoolIntArray MLPPKNN::nearest_neighbors(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!_input_set.is_valid(), PoolIntArray());

	MLPPLinAlg alg;
	// The nearest neighbors
	PoolIntArray knn;

	HashMap<int, bool> skip_map;

	Ref<MLPPVector> tmpv1;
	tmpv1.instance();
	Ref<MLPPVector> tmpv2;
	tmpv2.instance();

	int iuss = _input_set->size().y;

	//Perfom this loop unless and until all k nearest neighbors are found, appended, and returned
	for (int i = 0; i < _k; ++i) {
		int neighbor = 0;

		for (int j = 0; j < iuss; j++) {
			if (skip_map.has(j)) {
				continue;
			}

			_input_set->get_row_into_mlpp_vector(j, tmpv1);
			_input_set->get_row_into_mlpp_vector(neighbor, tmpv2);

			bool is_neighbor_nearer = alg.euclidean_distance(x, tmpv1) < alg.euclidean_distance(x, tmpv2);

			if (is_neighbor_nearer) {
				neighbor = j;
			}
		}

		if (!skip_map.has(neighbor)) {
			knn.push_back(neighbor);
			skip_map.set(neighbor, true);
		}
	}

	return knn;
}

int MLPPKNN::determine_class(const PoolIntArray &knn) {
	ERR_FAIL_COND_V(!_output_set.is_valid(), 0);

	int output_set_size = _output_set->size();

	ERR_FAIL_COND_V(output_set_size == 0, 0);

	const real_t *os_ptr = _output_set->ptr();

	HashMap<int, int> class_nums;

	for (int i = 0; i < output_set_size; ++i) {
		class_nums[static_cast<int>(os_ptr[i])] = 0;
	}

	PoolIntArray::Read knn_r = knn.read();
	const int *knn_ptr = knn_r.ptr();
	int knn_size = knn.size();

	for (int i = 0; i < knn_size; ++i) {
		for (int j = 0; j < output_set_size; j++) {
			int opj = static_cast<int>(os_ptr[j]);
			if (knn_ptr[i] == opj) {
				class_nums[opj]++;
			}
		}
	}

	int final_class = static_cast<int>(os_ptr[0]);
	int max = class_nums[final_class];

	for (int i = 0; i < output_set_size; ++i) {
		int opi = static_cast<int>(os_ptr[i]);

		if (class_nums[opi] > max) {
			max = class_nums[opi];
			final_class = opi;
		}
	}

	return final_class;
}

void MLPPKNN::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPKNN::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "value"), &MLPPKNN::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_output_set"), &MLPPKNN::get_output_set);
	ClassDB::bind_method(D_METHOD("set_output_set", "value"), &MLPPKNN::set_output_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output_set", "get_output_set");

	ClassDB::bind_method(D_METHOD("get_k"), &MLPPKNN::get_k);
	ClassDB::bind_method(D_METHOD("set_k", "value"), &MLPPKNN::set_k);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "k"), "set_k", "get_k");

	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPKNN::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPKNN::model_test);
	ClassDB::bind_method(D_METHOD("score"), &MLPPKNN::score);
}
