

#include "pca.h"
#include "../data/data.h"

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

	MLPPData data;

	MLPPMatrix::SVDResult svr_res = _input_set->cov()->svd();
	_x_normalized = data.mean_centering(_input_set);

	Size2i svr_res_u_size = svr_res.U->size();

	_u_reduce->resize(Size2i(_k, svr_res_u_size.y));

	for (int i = 0; i < _k; ++i) {
		for (int j = 0; j < svr_res_u_size.y; ++j) {
			_u_reduce->element_set(j, i, svr_res.U->element_get(j, i));
		}
	}

	_z = _u_reduce->transposen()->multn(_x_normalized);

	return _z;
}

// Simply tells us the percentage of variance maintained.
real_t MLPPPCA::score() {
	ERR_FAIL_COND_V(!_input_set.is_valid() || _k == 0, 0);

	Ref<MLPPMatrix> x_approx = _u_reduce->multn(_z);
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
		_x_normalized->row_get_into_mlpp_vector(i, x_normalized_row_tmp);
		x_approx->row_get_into_mlpp_vector(i, x_approx_row_tmp);

		num += x_normalized_row_tmp->subn(x_approx_row_tmp)->norm_sq();
	}

	num /= x_normalized_size_y;

	for (int i = 0; i < x_normalized_size_y; ++i) {
		_x_normalized->row_get_into_mlpp_vector(i, x_normalized_row_tmp);

		den += x_normalized_row_tmp->norm_sq();
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
