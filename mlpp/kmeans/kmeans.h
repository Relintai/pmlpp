
#ifndef MLPP_K_MEANS_H
#define MLPP_K_MEANS_H

//
//  KMeans.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

class MLPPKMeans : public Reference {
	GDCLASS(MLPPKMeans, Reference);

public:
	enum MeanType {
		MEAN_TYPE_CENTROID = 0,
		MEAN_TYPE_KMEANSPP,
	};

public:
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	int get_k();
	void set_k(const int val);

	MeanType get_mean_type();
	void set_mean_type(const MeanType val);

	void initialize();

	Ref<MLPPMatrix> model_set_test(const Ref<MLPPMatrix> &X);
	Ref<MLPPVector> model_test(const Ref<MLPPVector> &x);
	void train(int epoch_num, bool UI = false);
	real_t score();
	Ref<MLPPVector> silhouette_scores();

	MLPPKMeans();
	~MLPPKMeans();

protected:
	void _evaluate();
	void _compute_mu();

	void _centroid_initialization();
	void _kmeanspp_initialization();
	real_t _cost();

	static void _bind_methods();

	Ref<MLPPMatrix> _input_set;
	Ref<MLPPMatrix> _mu;
	Ref<MLPPMatrix> _r;

	real_t _accuracy_threshold;
	int _k;
	bool _initialized;

	MeanType _mean_type;
};

VARIANT_ENUM_CAST(MLPPKMeans::MeanType);

#endif /* KMeans_hpp */
