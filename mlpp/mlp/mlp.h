
#ifndef MLPP_MLP_H
#define MLPP_MLP_H

//
//  MLP.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/containers/vector.h"
#include "core/math/math_defs.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"

#include "core/object/reference.h"

#include "../regularization/reg.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <map>
#include <string>
#include <vector>

class MLPPMLP : public Reference {
	GDCLASS(MLPPMLP, Reference);

public:
	Ref<MLPPMatrix> get_input_set();
	void set_input_set(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output_set();
	void set_output_set(const Ref<MLPPVector> &val);

	int get_n_hidden();
	void set_n_hidden(const int val);

	real_t get_lambda();
	void set_lambda(const real_t val);

	real_t get_alpha();
	void set_alpha(const real_t val);

	MLPPReg::RegularizationType get_reg();
	void set_reg(const MLPPReg::RegularizationType val);

	Ref<MLPPVector> model_set_test(const Ref<MLPPMatrix> &X);
	real_t model_test(const Ref<MLPPVector> &x);

	bool is_initialized();
	void initialize();

	void gradient_descent(real_t learning_rate, int max_epoch, bool UI = false);
	void sgd(real_t learning_rate, int max_epoch, bool UI = false);
	void mbgd(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI = false);

	real_t score();
	void save(const String &file_name);

	MLPPMLP(const Ref<MLPPMatrix> &p_input_set, const Ref<MLPPVector> &p_output_set, int p_n_hidden, MLPPReg::RegularizationType p_reg = MLPPReg::REGULARIZATION_TYPE_NONE, real_t p_lambda = 0.5, real_t p_alpha = 0.5);

	MLPPMLP();
	~MLPPMLP();

protected:
	real_t cost(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &y);

	Ref<MLPPVector> evaluatem(const Ref<MLPPMatrix> &X);
	void propagatem(const Ref<MLPPMatrix> &X, Ref<MLPPMatrix> z2_out, Ref<MLPPMatrix> a2_out);

	real_t evaluatev(const Ref<MLPPVector> &x);
	void propagatev(const Ref<MLPPVector> &x, Ref<MLPPVector> z2_out, Ref<MLPPVector> a2_out);

	void forward_pass();

	static void _bind_methods();

	Ref<MLPPMatrix> input_set;
	Ref<MLPPVector> output_set;
	Ref<MLPPVector> y_hat;

	Ref<MLPPMatrix> weights1;
	Ref<MLPPVector> weights2;

	Ref<MLPPVector> bias1;
	real_t bias2;

	Ref<MLPPMatrix> z2;
	Ref<MLPPMatrix> a2;

	int n;
	int k;
	int n_hidden;

	// Regularization Params
	MLPPReg::RegularizationType reg;
	real_t lambda; /* Regularization Parameter */
	real_t alpha; /* This is the controlling param for Elastic Net*/

	int _initialized;
};

class MLPPMLPOld {
public:
	MLPPMLPOld(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, int n_hidden, std::string reg = "None", real_t lambda = 0.5, real_t alpha = 0.5);
	std::vector<real_t> modelSetTest(std::vector<std::vector<real_t>> X);
	real_t modelTest(std::vector<real_t> x);
	void gradientDescent(real_t learning_rate, int max_epoch, bool UI = false);
	void SGD(real_t learning_rate, int max_epoch, bool UI = false);
	void MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI = false);
	real_t score();
	void save(std::string fileName);

private:
	real_t Cost(std::vector<real_t> y_hat, std::vector<real_t> y);

	std::vector<real_t> Evaluate(std::vector<std::vector<real_t>> X);
	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> propagate(std::vector<std::vector<real_t>> X);
	real_t Evaluate(std::vector<real_t> x);
	std::tuple<std::vector<real_t>, std::vector<real_t>> propagate(std::vector<real_t> x);
	void forwardPass();

	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;
	std::vector<real_t> y_hat;

	std::vector<std::vector<real_t>> weights1;
	std::vector<real_t> weights2;

	std::vector<real_t> bias1;
	real_t bias2;

	std::vector<std::vector<real_t>> z2;
	std::vector<std::vector<real_t>> a2;

	int n;
	int k;
	int n_hidden;

	// Regularization Params
	std::string reg;
	real_t lambda; /* Regularization Parameter */
	real_t alpha; /* This is the controlling param for Elastic Net*/
};

#endif /* MLP_hpp */
