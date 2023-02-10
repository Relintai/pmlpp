
#ifndef MLPP_AUTO_ENCODER_H
#define MLPP_AUTO_ENCODER_H

//
//  AutoEncoder.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../regularization/reg.h"

//REMOVE
#include <iostream>
#include <string>
#include <vector>

class MLPPAutoEncoder : public Reference {
	GDCLASS(MLPPAutoEncoder, Reference);

public:
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	int get_n_hidden();
	void set_n_hidden(const int val);

	std::vector<std::vector<real_t>> model_set_test(std::vector<std::vector<real_t>> X);
	std::vector<real_t> model_test(std::vector<real_t> x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	void sgd(real_t learning_rate, int max_epoch, bool ui = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool ui = false);

	real_t score();

	void save(std::string fileName);

	MLPPAutoEncoder(std::vector<std::vector<real_t>> inputSet, int n_hidden);

	MLPPAutoEncoder();
	~MLPPAutoEncoder();

protected:
	real_t cost(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	std::vector<real_t> evaluatev(std::vector<real_t> x);
	std::tuple<std::vector<real_t>, std::vector<real_t>> propagatev(std::vector<real_t> x);

	std::vector<std::vector<real_t>> evaluatem(std::vector<std::vector<real_t>> X);
	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> propagatem(std::vector<std::vector<real_t>> X);

	void forward_pass();

	static void _bind_methods();

	std::vector<std::vector<real_t>> _input_set;
	std::vector<std::vector<real_t>> _y_hat;

	std::vector<std::vector<real_t>> _weights1;
	std::vector<std::vector<real_t>> _weights2;

	std::vector<real_t> _bias1;
	std::vector<real_t> _bias2;

	std::vector<std::vector<real_t>> _z2;
	std::vector<std::vector<real_t>> _a2;

	int _n;
	int _k;
	int _n_hidden;

	bool _initialized;
};

#endif /* AutoEncoder_hpp */
