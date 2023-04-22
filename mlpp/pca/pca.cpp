//
//  PCA.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "pca.h"
#include "../data/data.h"
#include "../lin_alg/lin_alg.h"

Ref<MLPPMatrix> MLPPPCA::get_input_set() {
	return _input_set;
}
void MLPPPCA::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;
}

int MLPPPCA::get_k() {
	return _k;
}
void MLPPPCA::set_k(const int val) {
	_k = val;
}

Ref<MLPPMatrix> MLPPPCA::principal_components() {
	ERR_FAIL_COND_V(!_input_set.is_valid() || _k == 0, Ref<MLPPMatrix>());

	MLPPLinAlg alg;
	MLPPData data;

	MLPPLinAlg::SVDResult svr_res = alg.svd(alg.covm(_input_set));
	_x_normalized = data.mean_centering(_input_set);

	Size2i svr_res_u_size = svr_res.U->size();

	_u_reduce->resize(Size2i(_k, svr_res_u_size.y));

	for (int i = 0; i < _k; ++i) {
		for (int j = 0; j < svr_res_u_size.y; ++j) {
			_u_reduce->set_element(j, i, svr_res.U->get_element(j, i));
		}
	}

	_z = alg.matmultnm(alg.transposenm(_u_reduce), _x_normalized);

	return _z;
}

// Simply tells us the percentage of variance maintained.
real_t MLPPPCA::score() {
	ERR_FAIL_COND_V(!_input_set.is_valid() || _k == 0, 0);

	MLPPLinAlg alg;

	Ref<MLPPMatrix> x_approx = alg.matmultnm(_u_reduce, _z);
	real_t num = 0;
	real_t den = 0;

	Size2i x_normalized_size = _x_normalized->size();

	int x_normalized_size_y = x_normalized_size.y;

	Ref<MLPPVector> x_approx_row_tmp;
	x_approx_row_tmp.instance();
	x_approx_row_tmp->resize(x_approx->size().x);

	Ref<MLPPVector> x_normalized_row_tmp;
	x_normalized_row_tmp.instance();
	x_normalized_row_tmp->resize(x_normalized_size.x);

	for (int i = 0; i < x_normalized_size_y; ++i) {
		_x_normalized->get_row_into_mlpp_vector(i, x_normalized_row_tmp);
		x_approx->get_row_into_mlpp_vector(i, x_approx_row_tmp);

		num += alg.norm_sqv(alg.subtractionnv(x_normalized_row_tmp, x_approx_row_tmp));
	}

	num /= x_normalized_size_y;

	for (int i = 0; i < x_normalized_size_y; ++i) {
		_x_normalized->get_row_into_mlpp_vector(i, x_normalized_row_tmp);

		den += alg.norm_sqv(x_normalized_row_tmp);
	}

	den /= x_normalized_size_y;

	if (den == 0) {
		den += 1e-10; // For numerical sanity as to not recieve a domain error
	}

	return 1 - num / den;
}

MLPPPCA::MLPPPCA(const Ref<MLPPMatrix> &p_input_set, int p_k) {
	_k = p_k;
	_input_set = p_input_set;

	_x_normalized.instance();
	_u_reduce.instance();
	_z.instance();
}

MLPPPCA::MLPPPCA() {
	_k = 0;

	_x_normalized.instance();
	_u_reduce.instance();
	_z.instance();
}
MLPPPCA::~MLPPPCA() {
}

void MLPPPCA::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPPCA::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "val"), &MLPPPCA::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "get_input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_k"), &MLPPPCA::get_k);
	ClassDB::bind_method(D_METHOD("set_k", "val"), &MLPPPCA::set_k);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "k"), "set_k", "get_k");

	ClassDB::bind_method(D_METHOD("principal_components"), &MLPPPCA::principal_components);
	ClassDB::bind_method(D_METHOD("score"), &MLPPPCA::score);
}
