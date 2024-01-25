/*************************************************************************/
/*  kmeans.cpp                                                           */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2023-present Péter Magyar.                              */
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

#include "kmeans.h"
#include "../utilities/utilities.h"

#include "core/math/random_pcg.h"

#include <climits>
#include <iostream>
#include <random>

Ref<MLPPMatrix> MLPPKMeans::get_input_set() {
	return _input_set;
}
void MLPPKMeans::set_input_set(const Ref<MLPPMatrix> &val) {
	_input_set = val;
	_initialized = false;
}

int MLPPKMeans::get_k() {
	return _k;
}
void MLPPKMeans::set_k(const int val) {
	_k = val;
	_initialized = false;
}

MLPPKMeans::MeanType MLPPKMeans::get_mean_type() {
	return _mean_type;
}
void MLPPKMeans::set_mean_type(const MLPPKMeans::MeanType val) {
	_mean_type = val;
	_initialized = false;
}

void MLPPKMeans::initialize() {
	ERR_FAIL_COND(!_input_set.is_valid());

	if (_mean_type == MEAN_TYPE_KMEANSPP) {
		_kmeanspp_initialization();
	} else {
		_centroid_initialization();
	}

	_initialized = true;
}

Ref<MLPPMatrix> MLPPKMeans::model_set_test(const Ref<MLPPMatrix> &X) {
	ERR_FAIL_COND_V(!X.is_valid(), Ref<MLPPMatrix>());
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPMatrix>());

	int input_set_size_y = _input_set->size().y;

	Ref<MLPPMatrix> closest_centroids;
	closest_centroids.instance();
	closest_centroids->resize(Size2i(_mu->size().x, input_set_size_y));

	Ref<MLPPVector> closest_centroid;
	closest_centroid.instance();
	closest_centroid->resize(_mu->size().x);

	Ref<MLPPVector> tmp_xiv;
	tmp_xiv.instance();
	tmp_xiv->resize(X->size().x);

	Ref<MLPPVector> tmp_mujv;
	tmp_mujv.instance();
	tmp_mujv->resize(_mu->size().x);

	int r0_size = _r->size().x;

	for (int i = 0; i < input_set_size_y; ++i) {
		_mu->row_get_into_mlpp_vector(0, closest_centroid);
		X->row_get_into_mlpp_vector(i, tmp_xiv);

		for (int j = 0; j < r0_size; ++j) {
			_mu->row_get_into_mlpp_vector(j, tmp_mujv);

			bool is_centroid_closer = tmp_xiv->euclidean_distance(tmp_mujv) < tmp_xiv->euclidean_distance(closest_centroid);

			if (is_centroid_closer) {
				closest_centroid->set_from_mlpp_vector(tmp_mujv);
			}
		}

		closest_centroids->row_set_mlpp_vector(i, closest_centroid);
	}

	return closest_centroids;
}
Ref<MLPPVector> MLPPKMeans::model_test(const Ref<MLPPVector> &x) {
	ERR_FAIL_COND_V(!x.is_valid(), Ref<MLPPVector>());
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPVector>());

	Ref<MLPPVector> closest_centroid;
	closest_centroid.instance();
	closest_centroid->resize(_mu->size().x);

	_mu->row_get_into_mlpp_vector(0, closest_centroid);

	int mu_size_y = _mu->size().y;

	Ref<MLPPVector> tmp_mujv;
	tmp_mujv.instance();
	tmp_mujv->resize(_mu->size().x);

	for (int j = 0; j < mu_size_y; ++j) {
		_mu->row_get_into_mlpp_vector(j, tmp_mujv);

		if (x->euclidean_distance(tmp_mujv) < x->euclidean_distance(closest_centroid)) {
			closest_centroid->set_from_mlpp_vector(tmp_mujv);
		}
	}

	return closest_centroid;
}
void MLPPKMeans::train(int epoch_num, bool UI) {
	ERR_FAIL_COND(!_input_set.is_valid());

	if (!_initialized) {
		initialize();
	}

	real_t cost_prev = 0;
	int epoch = 1;

	_evaluate();

	while (true) {
		// STEPS OF THE ALGORITHM
		// 1. DETERMINE r_nk
		// 2. DETERMINE J
		// 3. DETERMINE mu_k

		// STOP IF CONVERGED, ELSE REPEAT

		cost_prev = _cost();

		_compute_mu();
		_evaluate();

		// UI PORTION
		if (UI) {
			MLPPUtilities::cost_info(epoch, cost_prev, _cost());
		}

		epoch++;

		if (epoch > epoch_num) {
			break;
		}
	}
}

real_t MLPPKMeans::score() {
	return _cost();
}

Ref<MLPPVector> MLPPKMeans::silhouette_scores() {
	ERR_FAIL_COND_V(!_initialized, Ref<MLPPVector>());

	Ref<MLPPMatrix> closest_centroids = model_set_test(_input_set);

	ERR_FAIL_COND_V(!closest_centroids.is_valid(), Ref<MLPPVector>());

	int input_set_size_y = _input_set->size().y;
	int input_set_size_x = _input_set->size().x;

	int mu_size_y = _mu->size().y;

	int closest_centroids_size_y = closest_centroids->size().y;

	Ref<MLPPVector> silhouette_scores;
	silhouette_scores.instance();
	silhouette_scores->resize(input_set_size_y);

	Ref<MLPPVector> input_set_i_tempv;
	input_set_i_tempv.instance();
	input_set_i_tempv->resize(input_set_size_x);

	Ref<MLPPVector> input_set_j_tempv;
	input_set_j_tempv.instance();
	input_set_j_tempv->resize(input_set_size_x);

	Ref<MLPPVector> input_set_k_tempv;
	input_set_k_tempv.instance();
	input_set_k_tempv->resize(input_set_size_x);

	Ref<MLPPVector> r_i_tempv;
	r_i_tempv.instance();
	r_i_tempv->resize(_r->size().x);

	Ref<MLPPVector> r_j_tempv;
	r_j_tempv.instance();
	r_j_tempv->resize(_r->size().x);

	Ref<MLPPVector> closest_centroids_i_tempv;
	closest_centroids_i_tempv.instance();
	closest_centroids_i_tempv->resize(closest_centroids->size().x);

	Ref<MLPPVector> closest_centroids_k_tempv;
	closest_centroids_k_tempv.instance();
	closest_centroids_k_tempv->resize(closest_centroids->size().x);

	Ref<MLPPVector> mu_j_tempv;
	mu_j_tempv.instance();
	mu_j_tempv->resize(_mu->size().x);

	for (int i = 0; i < input_set_size_y; ++i) {
		_r->row_get_into_mlpp_vector(i, r_i_tempv);
		_input_set->row_get_into_mlpp_vector(i, input_set_i_tempv);

		// COMPUTING a[i]
		real_t a = 0;
		for (int j = 0; j < input_set_size_y; ++j) {
			if (i == j) {
				continue;
			}

			_r->row_get_into_mlpp_vector(j, r_j_tempv);

			if (r_i_tempv->is_equal_approx(r_j_tempv)) {
				_input_set->row_get_into_mlpp_vector(j, input_set_j_tempv);

				a += input_set_i_tempv->euclidean_distance(input_set_j_tempv);
			}
		}

		// NORMALIZE a[i]
		a /= closest_centroids->size().x - 1;

		closest_centroids->row_get_into_mlpp_vector(i, closest_centroids_i_tempv);

		// COMPUTING b[i]
		real_t b = Math_INF;
		for (int j = 0; j < mu_size_y; ++j) {
			_mu->row_get_into_mlpp_vector(j, mu_j_tempv);

			if (!closest_centroids_i_tempv->is_equal_approx(mu_j_tempv)) {
				real_t sum = 0;
				for (int k = 0; k < input_set_size_y; ++k) {
					_input_set->row_get_into_mlpp_vector(k, input_set_k_tempv);

					sum += input_set_i_tempv->euclidean_distance(input_set_k_tempv);
				}

				// NORMALIZE b[i]
				real_t k_cluster_size = 0;
				for (int k = 0; k < closest_centroids_size_y; ++k) {
					_input_set->row_get_into_mlpp_vector(k, closest_centroids_k_tempv);

					if (closest_centroids_k_tempv->is_equal_approx(mu_j_tempv)) {
						++k_cluster_size;
					}
				}

				if (sum / k_cluster_size < b) {
					b = sum / k_cluster_size;
				}
			}
		}

		silhouette_scores->element_set(i, (b - a) / fmax(a, b));

		// Or the expanded version:
		// if(a < b) {
		//     silhouette_scores->element_set(i, 1 - a/b);
		// }
		// else if(a == b){
		//     silhouette_scores->element_set(i, 0);
		// }
		// else{
		//     silhouette_scores->element_set(i, b/a - 1);
		// }
	}

	return silhouette_scores;
}

MLPPKMeans::MLPPKMeans() {
	_mu.instance();
	_r.instance();

	_accuracy_threshold = 0;
	_k = 0;
	_initialized = false;

	_mean_type = MEAN_TYPE_CENTROID;
}
MLPPKMeans::~MLPPKMeans() {
}

// This simply computes r_nk
void MLPPKMeans::_evaluate() {
	ERR_FAIL_COND(!_initialized);

	if (_r->size() != Size2i(_k, _input_set->size().y)) {
		_r->resize(Size2i(_k, _input_set->size().y));
	}

	int r_size_y = _r->size().y;
	int r_size_x = _r->size().x;

	Ref<MLPPVector> closest_centroid;
	closest_centroid.instance();
	closest_centroid->resize(_mu->size().x);

	Ref<MLPPVector> input_set_i_tempv;
	input_set_i_tempv.instance();
	input_set_i_tempv->resize(_input_set->size().x);

	Ref<MLPPVector> mu_j_tempv;
	mu_j_tempv.instance();
	mu_j_tempv->resize(_mu->size().x);

	real_t closest_centroid_current_dist = 0;
	int closest_centroid_index = 0;

	_r->fill(0);

	for (int i = 0; i < r_size_y; ++i) {
		_mu->row_get_into_mlpp_vector(0, closest_centroid);
		_input_set->row_get_into_mlpp_vector(i, input_set_i_tempv);

		closest_centroid_current_dist = input_set_i_tempv->euclidean_distance(closest_centroid);

		for (int j = 0; j < r_size_x; ++j) {
			_mu->row_get_into_mlpp_vector(j, mu_j_tempv);

			bool is_centroid_closer = input_set_i_tempv->euclidean_distance(mu_j_tempv) < closest_centroid_current_dist;

			if (is_centroid_closer) {
				_mu->row_get_into_mlpp_vector(j, closest_centroid);
				closest_centroid_current_dist = input_set_i_tempv->euclidean_distance(closest_centroid);
				closest_centroid_index = j;
			}
		}

		_r->element_set(i, closest_centroid_index, 1);
	}
}

// This simply computes or re-computes mu_k
void MLPPKMeans::_compute_mu() {
	int mu_size_y = _mu->size().y;
	int r_size_y = _r->size().y;

	Ref<MLPPVector> num;
	num.instance();
	num->resize(_r->size().x);

	Ref<MLPPVector> input_set_j_tempv;
	input_set_j_tempv.instance();
	input_set_j_tempv->resize(_input_set->size().x);

	Ref<MLPPVector> mat_tempv;
	mat_tempv.instance();
	mat_tempv->resize(_input_set->size().x);

	Ref<MLPPVector> mu_tempv;
	mu_tempv.instance();
	mu_tempv->resize(_mu->size().x);

	for (int i = 0; i < mu_size_y; ++i) {
		num->fill(0);

		real_t den = 0;
		for (int j = 0; j < r_size_y; ++j) {
			_input_set->row_get_into_mlpp_vector(j, input_set_j_tempv);

			real_t r_j_i = _r->element_get(j, i);

			mat_tempv->scalar_multiplyb(_r->element_get(j, i), input_set_j_tempv);
			num->add(mat_tempv);

			den += r_j_i;
		}

		mu_tempv->scalar_multiplyb(real_t(1) / real_t(den), num);

		_mu->row_set_mlpp_vector(i, mu_tempv);
	}
}

void MLPPKMeans::_centroid_initialization() {
	RandomPCG rand;
	rand.randomize();

	Size2i mu_size = Size2i(_input_set->size().x, _k);

	if (_mu->size() != mu_size) {
		_mu->resize(mu_size);
	}

	Ref<MLPPVector> mu_tempv;
	mu_tempv.instance();
	mu_tempv->resize(_mu->size().x);

	int input_set_size_y_rand = _input_set->size().y - 1;

	for (int i = 0; i < _k; ++i) {
		int indx = rand.random(0, input_set_size_y_rand);

		_input_set->row_get_into_mlpp_vector(indx, mu_tempv);
		_mu->row_set_mlpp_vector(i, mu_tempv);
	}
}

void MLPPKMeans::_kmeanspp_initialization() {
	RandomPCG rand;
	rand.randomize();

	Size2i mu_size = Size2i(_input_set->size().x, _k);

	if (_mu->size() != mu_size) {
		_mu->resize(mu_size);
	}

	int input_set_size_y = _input_set->size().y;

	Ref<MLPPVector> mu_tempv;
	mu_tempv.instance();
	mu_tempv->resize(_mu->size().x);

	_input_set->row_get_into_mlpp_vector(rand.random(0, input_set_size_y - 1), mu_tempv);
	_mu->row_set_mlpp_vector(0, mu_tempv);

	Ref<MLPPVector> input_set_j_tempv;
	input_set_j_tempv.instance();
	input_set_j_tempv->resize(_input_set->size().x);

	Ref<MLPPVector> farthest_centroid;
	farthest_centroid.instance();
	farthest_centroid->resize(_input_set->size().x);

	for (int i = 1; i < _k - 1; ++i) {
		for (int j = 0; j < input_set_size_y; ++j) {
			_input_set->row_get_into_mlpp_vector(j, input_set_j_tempv);

			real_t max_dist = 0;
			// SUM ALL THE SQUARED DISTANCES, CHOOSE THE ONE THAT'S FARTHEST
			// AS TO SPREAD OUT THE CLUSTER CENTROIDS.
			real_t sum = 0;
			for (int k = 0; k < i; k++) {
				_mu->row_get_into_mlpp_vector(k, mu_tempv);

				sum += input_set_j_tempv->euclidean_distance(mu_tempv);
			}

			if (sum * sum > max_dist) {
				farthest_centroid->set_from_mlpp_vector(input_set_j_tempv);
				max_dist = sum * sum;
			}
		}

		_mu->row_set_mlpp_vector(i, farthest_centroid);
	}
}
real_t MLPPKMeans::_cost() {
	ERR_FAIL_COND_V(!_initialized, 0);

	Ref<MLPPVector> input_set_i_tempv;
	input_set_i_tempv.instance();
	input_set_i_tempv->resize(_input_set->size().x);

	Ref<MLPPVector> mu_j_tempv;
	mu_j_tempv.instance();
	mu_j_tempv->resize(_mu->size().x);

	Ref<MLPPVector> sub_tempv;
	sub_tempv.instance();
	sub_tempv->resize(_input_set->size().x);

	int r_size_y = _r->size().y;
	int r_size_x = _r->size().x;

	real_t sum = 0;
	for (int i = 0; i < r_size_y; i++) {
		_input_set->row_get_into_mlpp_vector(i, input_set_i_tempv);

		for (int j = 0; j < r_size_x; j++) {
			_mu->row_get_into_mlpp_vector(j, mu_j_tempv);

			sub_tempv->subb(input_set_i_tempv, mu_j_tempv);
			sum += _r->element_get(i, j) * sub_tempv->norm_sq();
		}
	}

	return sum;
}

void MLPPKMeans::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input_set"), &MLPPKMeans::get_input_set);
	ClassDB::bind_method(D_METHOD("set_input_set", "value"), &MLPPKMeans::set_input_set);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input_set", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input_set", "get_input_set");

	ClassDB::bind_method(D_METHOD("get_k"), &MLPPKMeans::get_k);
	ClassDB::bind_method(D_METHOD("set_k", "value"), &MLPPKMeans::set_k);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "k"), "set_k", "get_k");

	ClassDB::bind_method(D_METHOD("get_mean_type"), &MLPPKMeans::get_mean_type);
	ClassDB::bind_method(D_METHOD("set_mean_type", "value"), &MLPPKMeans::set_mean_type);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mean_type", PROPERTY_HINT_ENUM, "Centroid,KMeansPP"), "set_mean_type", "get_mean_type");

	ClassDB::bind_method(D_METHOD("initialize"), &MLPPKMeans::initialize);
	ClassDB::bind_method(D_METHOD("model_set_test", "X"), &MLPPKMeans::model_set_test);
	ClassDB::bind_method(D_METHOD("model_test", "x"), &MLPPKMeans::model_test);
	ClassDB::bind_method(D_METHOD("train", "epoch_num", "UI"), &MLPPKMeans::train, false);
	ClassDB::bind_method(D_METHOD("score"), &MLPPKMeans::score);
	ClassDB::bind_method(D_METHOD("silhouette_scores"), &MLPPKMeans::silhouette_scores);

	BIND_ENUM_CONSTANT(MEAN_TYPE_CENTROID);
	BIND_ENUM_CONSTANT(MEAN_TYPE_KMEANSPP);
}

/*
std::vector<std::vector<real_t>> MLPPKMeans::modelSetTest(std::vector<std::vector<real_t>> X) {
	MLPPLinAlg alg;
	std::vector<std::vector<real_t>> closestCentroids;
	for (int i = 0; i < inputSet.size(); i++) {
		std::vector<real_t> closestCentroid = mu[0];
		for (int j = 0; j < r[0].size(); j++) {
			bool isCentroidCloser = alg.euclideanDistance(X[i], mu[j]) < alg.euclideanDistance(X[i], closestCentroid);
			if (isCentroidCloser) {
				closestCentroid = mu[j];
			}
		}
		closestCentroids.push_back(closestCentroid);
	}
	return closestCentroids;
}

std::vector<real_t> MLPPKMeans::modelTest(std::vector<real_t> x) {
	MLPPLinAlg alg;
	std::vector<real_t> closestCentroid = mu[0];
	for (int j = 0; j < mu.size(); j++) {
		if (alg.euclideanDistance(x, mu[j]) < alg.euclideanDistance(x, closestCentroid)) {
			closestCentroid = mu[j];
		}
	}
	return closestCentroid;
}

void MLPPKMeans::train(int epoch_num, bool UI) {
	real_t cost_prev = 0;
	int epoch = 1;

	Evaluate();

	while (true) {
		// STEPS OF THE ALGORITHM
		// 1. DETERMINE r_nk
		// 2. DETERMINE J
		// 3. DETERMINE mu_k

		// STOP IF CONVERGED, ELSE REPEAT

		cost_prev = Cost();

		computeMu();
		Evaluate();

		// UI PORTION
		if (UI) {
			MLPPUtilities::CostInfo(epoch, cost_prev, Cost());
		}
		epoch++;

		if (epoch > epoch_num) {
			break;
		}
	}
}

real_t MLPPKMeans::score() {
	return Cost();
}

std::vector<real_t> MLPPKMeans::silhouette_scores() {
	MLPPLinAlg alg;
	std::vector<std::vector<real_t>> closestCentroids = modelSetTest(inputSet);
	std::vector<real_t> silhouette_scores;
	for (int i = 0; i < inputSet.size(); i++) {
		// COMPUTING a[i]
		real_t a = 0;
		for (int j = 0; j < inputSet.size(); j++) {
			if (i != j && r[i] == r[j]) {
				a += alg.euclideanDistance(inputSet[i], inputSet[j]);
			}
		}
		// NORMALIZE a[i]
		a /= closestCentroids[i].size() - 1;

		// COMPUTING b[i]
		real_t b = INT_MAX;
		for (int j = 0; j < mu.size(); j++) {
			if (closestCentroids[i] != mu[j]) {
				real_t sum = 0;
				for (int k = 0; k < inputSet.size(); k++) {
					sum += alg.euclideanDistance(inputSet[i], inputSet[k]);
				}
				// NORMALIZE b[i]
				real_t k_clusterSize = 0;
				for (int k = 0; k < closestCentroids.size(); k++) {
					if (closestCentroids[k] == mu[j]) {
						k_clusterSize++;
					}
				}
				if (sum / k_clusterSize < b) {
					b = sum / k_clusterSize;
				}
			}
		}
		silhouette_scores.push_back((b - a) / fmax(a, b));
		// Or the expanded version:
		// if(a < b) {
		//     silhouette_scores.push_back(1 - a/b);
		// }
		// else if(a == b){
		//     silhouette_scores.push_back(0);
		// }
		// else{
		//     silhouette_scores.push_back(b/a - 1);
		// }
	}
	return silhouette_scores;
}

// This simply computes r_nk
void MLPPKMeans::Evaluate() {
	MLPPLinAlg alg;
	r.resize(inputSet.size());

	for (int i = 0; i < r.size(); i++) {
		r[i].resize(k);
	}

	for (int i = 0; i < r.size(); i++) {
		std::vector<real_t> closestCentroid = mu[0];
		for (int j = 0; j < r[0].size(); j++) {
			bool isCentroidCloser = alg.euclideanDistance(inputSet[i], mu[j]) < alg.euclideanDistance(inputSet[i], closestCentroid);
			if (isCentroidCloser) {
				closestCentroid = mu[j];
			}
		}
		for (int j = 0; j < r[0].size(); j++) {
			if (mu[j] == closestCentroid) {
				r[i][j] = 1;
			} else {
				r[i][j] = 0;
			}
		}
	}
}

// This simply computes or re-computes mu_k
void MLPPKMeans::computeMu() {
	MLPPLinAlg alg;
	for (int i = 0; i < mu.size(); i++) {
		std::vector<real_t> num;
		num.resize(r.size());

		for (int i = 0; i < num.size(); i++) {
			num[i] = 0;
		}

		real_t den = 0;
		for (int j = 0; j < r.size(); j++) {
			num = alg.addition(num, alg.scalarMultiply(r[j][i], inputSet[j]));
		}
		for (int j = 0; j < r.size(); j++) {
			den += r[j][i];
		}
		mu[i] = alg.scalarMultiply(real_t(1) / real_t(den), num);
	}
}

void MLPPKMeans::centroidInitialization(int k) {
	mu.resize(k);

	for (int i = 0; i < k; i++) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::uniform_int_distribution<int> distribution(0, int(inputSet.size() - 1));

		mu[i].resize(inputSet.size());
		mu[i] = inputSet[distribution(generator)];
	}
}

void MLPPKMeans::kmeansppInitialization(int k) {
	MLPPLinAlg alg;
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, int(inputSet.size() - 1));
	mu.push_back(inputSet[distribution(generator)]);

	for (int i = 0; i < k - 1; i++) {
		std::vector<real_t> farthestCentroid;
		for (int j = 0; j < inputSet.size(); j++) {
			real_t max_dist = 0;
			// SUM ALL THE SQUARED DISTANCES, CHOOSE THE ONE THAT'S FARTHEST
			// AS TO SPREAD OUT THE CLUSTER CENTROIDS.
			real_t sum = 0;
			for (int k = 0; k < mu.size(); k++) {
				sum += alg.euclideanDistance(inputSet[j], mu[k]);
			}
			if (sum * sum > max_dist) {
				farthestCentroid = inputSet[j];
				max_dist = sum * sum;
			}
		}
		mu.push_back(farthestCentroid);
	}
}

real_t MLPPKMeans::Cost() {
	MLPPLinAlg alg;
	real_t sum = 0;
	for (int i = 0; i < r.size(); i++) {
		for (int j = 0; j < r[0].size(); j++) {
			sum += r[i][j] * alg.norm_sq(alg.subtraction(inputSet[i], mu[j]));
		}
	}
	return sum;
}

*/