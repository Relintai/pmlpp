//
//  KMeans.cpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "kmeans.h"
#include "../lin_alg/lin_alg.h"
#include "../utilities/utilities.h"

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
	if (_mean_type == MEAN_TYPE_KMEANSPP) {
		_kmeanspp_initialization(_k);
	} else {
		_centroid_initialization(_k);
	}
}

Ref<MLPPMatrix> MLPPKMeans::model_set_test(const Ref<MLPPMatrix> &X) {
	return Ref<MLPPMatrix>();
}
Ref<MLPPVector> MLPPKMeans::model_test(const Ref<MLPPVector> &x) {
	return Ref<MLPPVector>();
}
void MLPPKMeans::train(int epoch_num, bool UI) {
}
real_t MLPPKMeans::score() {
	return 0;
}
Ref<MLPPVector> MLPPKMeans::silhouette_scores() {
	return Ref<MLPPVector>();
}

MLPPKMeans::MLPPKMeans() {
	_accuracy_threshold = 0;
	_k = 0;
	_initialized = false;

	_mean_type = MEAN_TYPE_CENTROID;
}
MLPPKMeans::~MLPPKMeans() {
}

void MLPPKMeans::_evaluate() {
}
void MLPPKMeans::_compute_mu() {
}

void MLPPKMeans::_centroid_initialization(int k) {
}
void MLPPKMeans::_kmeanspp_initialization(int k) {
}
real_t MLPPKMeans::_cost() {
	return 0;
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