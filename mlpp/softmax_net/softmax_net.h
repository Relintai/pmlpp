#ifndef MLPP_SOFTMAX_NET_H
#define MLPP_SOFTMAX_NET_H

//
//  SoftmaxNet.hpp
//
//  Created by Marc Melikyan on 10/2/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../regularization/reg.h"

#include <string>
#include <vector>

class MLPPSoftmaxNet : public Reference {
	GDCLASS(MLPPSoftmaxNet, Reference);

public:
	/*
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> get_output_set();
	void set_output_set(const Ref<MLPPMatrix> &val);

	MLPPReg::RegularizationType get_reg();
	void set_reg(const MLPPReg::RegularizationType val);

	real_t get_lambda();
	void set_lambda(const real_t val);

	real_t get_alpha();
	void set_alpha(const real_t val);
	*/

	std::vector<real_t> model_test(std::vector<real_t> x);
	std::vector<std::vector<real_t>> model_set_test(std::vector<std::vector<real_t>> X);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	void save(std::string fileName);

	std::vector<std::vector<real_t>> get_embeddings(); // This class is used (mostly) for word2Vec. This function returns our embeddings.

	bool is_initialized();
	void initialize();

	MLPPSoftmaxNet(std::vector<std::vector<real_t>> p_input_set, std::vector<std::vector<real_t>> p_output_set, int p_n_hidden, MLPPReg::RegularizationType p_reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t p_lambda = 0.5, real_t p_alpha = 0.5);
	//MLPPSoftmaxNet(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPMatrix> &p_output_set, MLPPReg::RegularizationType p_reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t p_lambda = 0.5, real_t p_alpha = 0.5);

	MLPPSoftmaxNet();
	~MLPPSoftmaxNet();

protected:
	real_t cost(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	std::vector<real_t> evaluatev(std::vector<real_t> x);
	std::tuple<std::vector<real_t>, std::vector<real_t>> propagatev(std::vector<real_t> x);

	std::vector<std::vector<real_t>> evaluatem(std::vector<std::vector<real_t>> X);
	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> propagatem(std::vector<std::vector<real_t>> X);

	void forward_pass();

	static void _bind_methods();

	std::vector<std::vector<real_t>> _input_set;
	std::vector<std::vector<real_t>> _output_set;
	std::vector<std::vector<real_t>> _y_hat;

	std::vector<std::vector<real_t>> _weights1;
	std::vector<std::vector<real_t>> _weights2;

	std::vector<real_t> _bias1;
	std::vector<real_t> _bias2;

	std::vector<std::vector<real_t>> _z2;
	std::vector<std::vector<real_t>> _a2;

	int _n;
	int _k;
	int _n_class;
	int _n_hidden;

	// Regularization Params
	MLPPReg::RegularizationType _reg;
	real_t _lambda;
	real_t _alpha; /* This is the controlling param for Elastic Net*/

	bool _initialized;
};

#endif /* SoftmaxNet_hpp */
