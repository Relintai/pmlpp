
#ifndef MLPP_MANN_H
#define MLPP_MANN_H

//
//  MANN.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/math/math_defs.h"

#include "core/object/reference.h"

#include "../regularization/reg.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include "../hidden_layer/hidden_layer.h"
#include "../multi_output_layer/multi_output_layer.h"

#include "../hidden_layer/hidden_layer_old.h"
#include "../multi_output_layer/multi_output_layer_old.h"

#include <string>
#include <vector>

class MLPPMANN : public Reference {
	GDCLASS(MLPPMANN, Reference);

public:
	/*
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> get_output_set();
	void set_output_set(const Ref<MLPPMatrix> &val);
	*/

	std::vector<std::vector<real_t>> model_set_test(std::vector<std::vector<real_t>> X);
	std::vector<real_t> model_test(std::vector<real_t> x);

	void gradient_descent(real_t learning_rate, int max_epoch, bool ui = false);
	real_t score();

	void save(std::string file_name);

	void add_layer(int n_hidden, std::string activation, std::string weight_init = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);
	void add_output_layer(std::string activation, std::string loss, std::string weight_init = "Default", std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);

	bool is_initialized();
	void initialize();

	MLPPMANN(std::vector<std::vector<real_t>> p_input_set, std::vector<std::vector<real_t>> p_output_set);

	MLPPMANN();
	~MLPPMANN();

private:
	real_t cost(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	void forward_pass();

	static void _bind_methods();

	std::vector<std::vector<real_t>> _input_set;
	std::vector<std::vector<real_t>> _output_set;
	std::vector<std::vector<real_t>> _y_hat;

	std::vector<MLPPOldHiddenLayer> _network;
	MLPPOldMultiOutputLayer *_output_layer;

	int _n;
	int _k;
	int _n_output;

	bool _initialized;
};

#endif /* MANN_hpp */