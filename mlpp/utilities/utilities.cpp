//
//  Reg.cpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "utilities.h"

#include "core/log/logger.h"
#include "core/math/math_funcs.h"
#include "core/math/random_pcg.h"

#include <fstream>
#include <iostream>
#include <random>
#include <string>

std::vector<real_t> MLPPUtilities::weightInitialization(int n, std::string type) {
	std::random_device rd;
	std::default_random_engine generator(rd());

	std::vector<real_t> weights;
	for (int i = 0; i < n; i++) {
		if (type == "XavierNormal") {
			std::normal_distribution<real_t> distribution(0, sqrt(2 / (n + 1)));
			weights.push_back(distribution(generator));
		} else if (type == "XavierUniform") {
			std::uniform_real_distribution<real_t> distribution(-sqrt(6 / (n + 1)), sqrt(6 / (n + 1)));
			weights.push_back(distribution(generator));
		} else if (type == "HeNormal") {
			std::normal_distribution<real_t> distribution(0, sqrt(2 / n));
			weights.push_back(distribution(generator));
		} else if (type == "HeUniform") {
			std::uniform_real_distribution<real_t> distribution(-sqrt(6 / n), sqrt(6 / n));
			weights.push_back(distribution(generator));
		} else if (type == "LeCunNormal") {
			std::normal_distribution<real_t> distribution(0, sqrt(1 / n));
			weights.push_back(distribution(generator));
		} else if (type == "LeCunUniform") {
			std::uniform_real_distribution<real_t> distribution(-sqrt(3 / n), sqrt(3 / n));
			weights.push_back(distribution(generator));
		} else if (type == "Uniform") {
			std::uniform_real_distribution<real_t> distribution(-1 / sqrt(n), 1 / sqrt(n));
			weights.push_back(distribution(generator));
		} else {
			std::uniform_real_distribution<real_t> distribution(0, 1);
			weights.push_back(distribution(generator));
		}
	}
	return weights;
}

real_t MLPPUtilities::biasInitialization() {
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_real_distribution<real_t> distribution(0, 1);

	return distribution(generator);
}

std::vector<std::vector<real_t>> MLPPUtilities::weightInitialization(int n, int m, std::string type) {
	std::random_device rd;
	std::default_random_engine generator(rd());

	std::vector<std::vector<real_t>> weights;
	weights.resize(n);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (type == "XavierNormal") {
				std::normal_distribution<real_t> distribution(0, sqrt(2 / (n + m)));
				weights[i].push_back(distribution(generator));
			} else if (type == "XavierUniform") {
				std::uniform_real_distribution<real_t> distribution(-sqrt(6 / (n + m)), sqrt(6 / (n + m)));
				weights[i].push_back(distribution(generator));
			} else if (type == "HeNormal") {
				std::normal_distribution<real_t> distribution(0, sqrt(2 / n));
				weights[i].push_back(distribution(generator));
			} else if (type == "HeUniform") {
				std::uniform_real_distribution<real_t> distribution(-sqrt(6 / n), sqrt(6 / n));
				weights[i].push_back(distribution(generator));
			} else if (type == "LeCunNormal") {
				std::normal_distribution<real_t> distribution(0, sqrt(1 / n));
				weights[i].push_back(distribution(generator));
			} else if (type == "LeCunUniform") {
				std::uniform_real_distribution<real_t> distribution(-sqrt(3 / n), sqrt(3 / n));
				weights[i].push_back(distribution(generator));
			} else if (type == "Uniform") {
				std::uniform_real_distribution<real_t> distribution(-1 / sqrt(n), 1 / sqrt(n));
				weights[i].push_back(distribution(generator));
			} else {
				std::uniform_real_distribution<real_t> distribution(0, 1);
				weights[i].push_back(distribution(generator));
			}
		}
	}
	return weights;
}

std::vector<real_t> MLPPUtilities::biasInitialization(int n) {
	std::vector<real_t> bias;
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_real_distribution<real_t> distribution(0, 1);

	for (int i = 0; i < n; i++) {
		bias.push_back(distribution(generator));
	}
	return bias;
}

void MLPPUtilities::weight_initializationv(Ref<MLPPVector> weights, WeightDistributionType type) {
	ERR_FAIL_COND(!weights.is_valid());

	int n = weights->size();
	real_t *weights_ptr = weights->ptrw();

	RandomPCG rnd;
	rnd.randomize();

	std::random_device rd;
	std::default_random_engine generator(rd());

	switch (type) {
		case WEIGHT_DISTRIBUTION_TYPE_DEFAULT: {
			std::uniform_real_distribution<real_t> distribution(0, 1);

			for (int i = 0; i < n; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_XAVIER_NORMAL: {
			std::normal_distribution<real_t> distribution(0, Math::sqrt(2.0 / (n + 1.0)));

			for (int i = 0; i < n; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_XAVIER_UNIFORM: {
			std::uniform_real_distribution<real_t> distribution(-Math::sqrt(6.0 / (n + 1.0)), Math::sqrt(6.0 / (n + 1.0)));

			for (int i = 0; i < n; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_HE_NORMAL: {
			std::normal_distribution<real_t> distribution(0, Math::sqrt(2.0 / n));

			for (int i = 0; i < n; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_HE_UNIFORM: {
			std::uniform_real_distribution<real_t> distribution(-Math::sqrt(6.0 / n), Math::sqrt(6.0 / n));

			for (int i = 0; i < n; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_LE_CUN_NORMAL: {
			std::normal_distribution<real_t> distribution(0, Math::sqrt(1.0 / n));

			for (int i = 0; i < n; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_LE_CUN_UNIFORM: {
			std::uniform_real_distribution<real_t> distribution(-Math::sqrt(3.0 / n), Math::sqrt(3.0 / n));

			for (int i = 0; i < n; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_UNIFORM: {
			std::uniform_real_distribution<real_t> distribution(-1.0 / Math::sqrt(static_cast<real_t>(n)), 1.0 / Math::sqrt(static_cast<real_t>(n)));

			for (int i = 0; i < n; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		default:
			break;
	}
}
void MLPPUtilities::weight_initializationm(Ref<MLPPMatrix> weights, WeightDistributionType type) {
	ERR_FAIL_COND(!weights.is_valid());

	int n = weights->size().x;
	int m = weights->size().y;
	int data_size = weights->data_size();
	real_t *weights_ptr = weights->ptrw();

	RandomPCG rnd;
	rnd.randomize();

	std::random_device rd;
	std::default_random_engine generator(rd());

	switch (type) {
		case WEIGHT_DISTRIBUTION_TYPE_DEFAULT: {
			std::uniform_real_distribution<real_t> distribution(0, 1);

			for (int i = 0; i < data_size; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_XAVIER_NORMAL: {
			std::normal_distribution<real_t> distribution(0, sqrt(2 / (n + m)));

			for (int i = 0; i < data_size; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_XAVIER_UNIFORM: {
			std::uniform_real_distribution<real_t> distribution(-sqrt(6 / (n + m)), sqrt(6 / (n + m)));

			for (int i = 0; i < data_size; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_HE_NORMAL: {
			std::normal_distribution<real_t> distribution(0, sqrt(2 / n));

			for (int i = 0; i < data_size; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_HE_UNIFORM: {
			std::uniform_real_distribution<real_t> distribution(-sqrt(6 / n), sqrt(6 / n));

			for (int i = 0; i < data_size; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_LE_CUN_NORMAL: {
			std::normal_distribution<real_t> distribution(0, sqrt(1 / n));

			for (int i = 0; i < data_size; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_LE_CUN_UNIFORM: {
			std::uniform_real_distribution<real_t> distribution(-sqrt(3 / n), sqrt(3 / n));

			for (int i = 0; i < data_size; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		case WEIGHT_DISTRIBUTION_TYPE_UNIFORM: {
			std::uniform_real_distribution<real_t> distribution(-1 / sqrt(n), 1 / sqrt(n));

			for (int i = 0; i < data_size; ++i) {
				weights_ptr[i] = distribution(generator);
			}
		} break;
		default:
			break;
	}
}
real_t MLPPUtilities::bias_initializationr() {
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_real_distribution<real_t> distribution(0, 1);

	return distribution(generator);
}
void MLPPUtilities::bias_initializationv(Ref<MLPPVector> z) {
	ERR_FAIL_COND(!z.is_valid());

	std::vector<real_t> bias;
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_real_distribution<real_t> distribution(0, 1);

	int n = z->size();

	for (int i = 0; i < n; i++) {
		bias.push_back(distribution(generator));
	}
}

real_t MLPPUtilities::performance(std::vector<real_t> y_hat, std::vector<real_t> outputSet) {
	real_t correct = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		if (std::round(y_hat[i]) == outputSet[i]) {
			correct++;
		}
	}
	return correct / y_hat.size();
}

real_t MLPPUtilities::performance(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y) {
	real_t correct = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		uint32_t sub_correct = 0;
		for (uint32_t j = 0; j < y_hat[i].size(); j++) {
			if (std::round(y_hat[i][j]) == y[i][j]) {
				sub_correct++;
			}
			if (sub_correct == y_hat[0].size()) {
				correct++;
			}
		}
	}
	return correct / y_hat.size();
}

real_t MLPPUtilities::performance_vec(const Ref<MLPPVector> &y_hat, const Ref<MLPPVector> &output_set) {
	ERR_FAIL_COND_V(!y_hat.is_valid(), 0);
	ERR_FAIL_COND_V(!output_set.is_valid(), 0);

	int correct = 0;
	for (int i = 0; i < y_hat->size(); i++) {
		if (Math::is_equal_approx(Math::round(y_hat->element_get(i)), output_set->element_get(i))) {
			correct++;
		}
	}
	return correct / (real_t)y_hat->size();
}
real_t MLPPUtilities::performance_mat(const Ref<MLPPMatrix> &y_hat, const Ref<MLPPMatrix> &y) {
	ERR_FAIL_COND_V(!y_hat.is_valid(), 0);
	ERR_FAIL_COND_V(!y.is_valid(), 0);

	real_t correct = 0;
	for (int i = 0; i < y_hat->size().y; i++) {
		int sub_correct = 0;

		for (int j = 0; j < y_hat->size().x; j++) {
			if (Math::is_equal_approx(Math::round(y_hat->element_get(i, j)), y->element_get(i, j))) {
				sub_correct++;
			}

			if (sub_correct == y_hat->size().x) {
				correct++;
			}
		}
	}
	return correct / (real_t)y_hat->size().y;
}
real_t MLPPUtilities::performance_pool_int_array_vec(PoolIntArray y_hat, const Ref<MLPPVector> &output_set) {
	ERR_FAIL_COND_V(!output_set.is_valid(), 0);

	real_t correct = 0;
	for (int i = 0; i < y_hat.size(); i++) {
		if (y_hat[i] == Math::round(output_set->element_get(i))) {
			correct++;
		}
	}
	return correct / (real_t)y_hat.size();
}

void MLPPUtilities::saveParameters(std::string fileName, std::vector<real_t> weights, real_t bias, bool app, int layer) {
	std::string layer_info = "";
	std::ofstream saveFile;

	if (layer > -1) {
		layer_info = " for layer " + std::to_string(layer);
	}

	if (app) {
		saveFile.open(fileName.c_str(), std::ios_base::app);
	} else {
		saveFile.open(fileName.c_str());
	}

	if (!saveFile.is_open()) {
		std::cout << fileName << " failed to open." << std::endl;
	}

	saveFile << "Weight(s)" << layer_info << std::endl;
	for (uint32_t i = 0; i < weights.size(); i++) {
		saveFile << weights[i] << std::endl;
	}
	saveFile << "Bias" << layer_info << std::endl;
	saveFile << bias << std::endl;

	saveFile.close();
}

void MLPPUtilities::saveParameters(std::string fileName, std::vector<real_t> weights, std::vector<real_t> initial, real_t bias, bool app, int layer) {
	std::string layer_info = "";
	std::ofstream saveFile;

	if (layer > -1) {
		layer_info = " for layer " + std::to_string(layer);
	}

	if (app) {
		saveFile.open(fileName.c_str(), std::ios_base::app);
	} else {
		saveFile.open(fileName.c_str());
	}

	if (!saveFile.is_open()) {
		std::cout << fileName << " failed to open." << std::endl;
	}

	saveFile << "Weight(s)" << layer_info << std::endl;
	for (uint32_t i = 0; i < weights.size(); i++) {
		saveFile << weights[i] << std::endl;
	}

	saveFile << "Initial(s)" << layer_info << std::endl;
	for (uint32_t i = 0; i < initial.size(); i++) {
		saveFile << initial[i] << std::endl;
	}

	saveFile << "Bias" << layer_info << std::endl;
	saveFile << bias << std::endl;

	saveFile.close();
}

void MLPPUtilities::saveParameters(std::string fileName, std::vector<std::vector<real_t>> weights, std::vector<real_t> bias, bool app, int layer) {
	std::string layer_info = "";
	std::ofstream saveFile;

	if (layer > -1) {
		layer_info = " for layer " + std::to_string(layer);
	}

	if (app) {
		saveFile.open(fileName.c_str(), std::ios_base::app);
	} else {
		saveFile.open(fileName.c_str());
	}

	if (!saveFile.is_open()) {
		std::cout << fileName << " failed to open." << std::endl;
	}

	saveFile << "Weight(s)" << layer_info << std::endl;
	for (uint32_t i = 0; i < weights.size(); i++) {
		for (uint32_t j = 0; j < weights[i].size(); j++) {
			saveFile << weights[i][j] << std::endl;
		}
	}
	saveFile << "Bias(es)" << layer_info << std::endl;
	for (uint32_t i = 0; i < bias.size(); i++) {
		saveFile << bias[i] << std::endl;
	}

	saveFile.close();
}

void MLPPUtilities::UI(std::vector<real_t> weights, real_t bias) {
	std::cout << "Values of the weight(s):" << std::endl;
	for (uint32_t i = 0; i < weights.size(); i++) {
		std::cout << weights[i] << std::endl;
	}
	std::cout << "Value of the bias:" << std::endl;
	std::cout << bias << std::endl;
}

void MLPPUtilities::UI(std::vector<std::vector<real_t>> weights, std::vector<real_t> bias) {
	std::cout << "Values of the weight(s):" << std::endl;
	for (uint32_t i = 0; i < weights.size(); i++) {
		for (uint32_t j = 0; j < weights[i].size(); j++) {
			std::cout << weights[i][j] << std::endl;
		}
	}
	std::cout << "Value of the biases:" << std::endl;
	for (uint32_t i = 0; i < bias.size(); i++) {
		std::cout << bias[i] << std::endl;
	}
}

void MLPPUtilities::UI(std::vector<real_t> weights, std::vector<real_t> initial, real_t bias) {
	std::cout << "Values of the weight(s):" << std::endl;
	for (uint32_t i = 0; i < weights.size(); i++) {
		std::cout << weights[i] << std::endl;
	}
	std::cout << "Values of the initial(s):" << std::endl;
	for (uint32_t i = 0; i < initial.size(); i++) {
		std::cout << initial[i] << std::endl;
	}
	std::cout << "Value of the bias:" << std::endl;
	std::cout << bias << std::endl;
}

void MLPPUtilities::print_ui_vb(Ref<MLPPVector> weights, real_t bias) {
	String str = "Values of the weight(s):\n";
	str += weights->to_string();
	str += "\nValue of the bias:\n";
	str += String::num(bias);

	PLOG_MSG(str);
}
void MLPPUtilities::print_ui_vib(Ref<MLPPVector> weights, Ref<MLPPVector> initial, real_t bias) {
	String str = "Values of the weight(s):\n";
	str += weights->to_string();

	str += "\nValues of the initial(s):\n";
	str += initial->to_string();

	str += "\nValue of the bias:\n";
	str += String::num(bias);

	PLOG_MSG(str);
}
void MLPPUtilities::print_ui_mb(Ref<MLPPMatrix> weights, Ref<MLPPVector> bias) {
	String str = "Values of the weight(s):\n";
	str += weights->to_string();

	str += "\nValue of the biased:\n";
	str += bias->to_string();

	PLOG_MSG(str);
}

void MLPPUtilities::CostInfo(int epoch, real_t cost_prev, real_t Cost) {
	std::cout << "-----------------------------------" << std::endl;
	std::cout << "This is epoch: " << epoch << std::endl;
	std::cout << "The cost function has been minimized by " << cost_prev - Cost << std::endl;
	std::cout << "Current Cost:" << std::endl;
	std::cout << Cost << std::endl;
}

void MLPPUtilities::cost_info(int epoch, real_t cost_prev, real_t cost) {
	String str = "This is epoch: " + itos(epoch) + ",";
	str += "The cost function has been minimized by " + String::num(cost_prev - cost);
	str += ", Current Cost:" + String::num(cost);

	PLOG_MSG(str);
}

std::vector<std::vector<std::vector<real_t>>> MLPPUtilities::createMiniBatches(std::vector<std::vector<real_t>> inputSet, int n_mini_batch) {
	int n = inputSet.size();

	std::vector<std::vector<std::vector<real_t>>> inputMiniBatches;

	// Creating the mini-batches
	for (int i = 0; i < n_mini_batch; i++) {
		std::vector<std::vector<real_t>> currentInputSet;
		for (int j = 0; j < n / n_mini_batch; j++) {
			currentInputSet.push_back(inputSet[n / n_mini_batch * i + j]);
		}
		inputMiniBatches.push_back(currentInputSet);
	}

	if (real_t(n) / real_t(n_mini_batch) - int(n / n_mini_batch) != 0) {
		for (int i = 0; i < n - n / n_mini_batch * n_mini_batch; i++) {
			inputMiniBatches[n_mini_batch - 1].push_back(inputSet[n / n_mini_batch * n_mini_batch + i]);
		}
	}
	return inputMiniBatches;
}

std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<std::vector<real_t>>> MLPPUtilities::createMiniBatches(std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet, int n_mini_batch) {
	int n = inputSet.size();

	std::vector<std::vector<std::vector<real_t>>> inputMiniBatches;
	std::vector<std::vector<real_t>> outputMiniBatches;

	for (int i = 0; i < n_mini_batch; i++) {
		std::vector<std::vector<real_t>> currentInputSet;
		std::vector<real_t> currentOutputSet;
		for (int j = 0; j < n / n_mini_batch; j++) {
			currentInputSet.push_back(inputSet[n / n_mini_batch * i + j]);
			currentOutputSet.push_back(outputSet[n / n_mini_batch * i + j]);
		}
		inputMiniBatches.push_back(currentInputSet);
		outputMiniBatches.push_back(currentOutputSet);
	}

	if (real_t(n) / real_t(n_mini_batch) - int(n / n_mini_batch) != 0) {
		for (int i = 0; i < n - n / n_mini_batch * n_mini_batch; i++) {
			inputMiniBatches[n_mini_batch - 1].push_back(inputSet[n / n_mini_batch * n_mini_batch + i]);
			outputMiniBatches[n_mini_batch - 1].push_back(outputSet[n / n_mini_batch * n_mini_batch + i]);
		}
	}
	return { inputMiniBatches, outputMiniBatches };
}

std::tuple<std::vector<std::vector<std::vector<real_t>>>, std::vector<std::vector<std::vector<real_t>>>> MLPPUtilities::createMiniBatches(std::vector<std::vector<real_t>> inputSet, std::vector<std::vector<real_t>> outputSet, int n_mini_batch) {
	int n = inputSet.size();

	std::vector<std::vector<std::vector<real_t>>> inputMiniBatches;
	std::vector<std::vector<std::vector<real_t>>> outputMiniBatches;

	for (int i = 0; i < n_mini_batch; i++) {
		std::vector<std::vector<real_t>> currentInputSet;
		std::vector<std::vector<real_t>> currentOutputSet;
		for (int j = 0; j < n / n_mini_batch; j++) {
			currentInputSet.push_back(inputSet[n / n_mini_batch * i + j]);
			currentOutputSet.push_back(outputSet[n / n_mini_batch * i + j]);
		}
		inputMiniBatches.push_back(currentInputSet);
		outputMiniBatches.push_back(currentOutputSet);
	}

	if (real_t(n) / real_t(n_mini_batch) - int(n / n_mini_batch) != 0) {
		for (int i = 0; i < n - n / n_mini_batch * n_mini_batch; i++) {
			inputMiniBatches[n_mini_batch - 1].push_back(inputSet[n / n_mini_batch * n_mini_batch + i]);
			outputMiniBatches[n_mini_batch - 1].push_back(outputSet[n / n_mini_batch * n_mini_batch + i]);
		}
	}
	return { inputMiniBatches, outputMiniBatches };
}

Vector<Ref<MLPPMatrix>> MLPPUtilities::create_mini_batchesm(const Ref<MLPPMatrix> &input_set, int n_mini_batch) {
	Size2i size = input_set->size();

	int n = size.y;
	int mini_batch_element_count = n / n_mini_batch;

	Ref<MLPPVector> row_tmp;
	row_tmp.instance();
	row_tmp->resize(size.x);

	Vector<Ref<MLPPMatrix>> input_mini_batches;

	// Creating the mini-batches
	for (int i = 0; i < n_mini_batch; i++) {
		int mini_batch_start_offset = n_mini_batch * i;
		Ref<MLPPMatrix> current_input_set;
		current_input_set.instance();
		current_input_set->resize(Size2i(size.x, mini_batch_element_count));

		for (int j = 0; j < mini_batch_element_count; j++) {
			input_set->row_get_into_mlpp_vector(mini_batch_start_offset + j, row_tmp);
			current_input_set->row_set_mlpp_vector(j, row_tmp);
		}

		input_mini_batches.push_back(current_input_set);
	}

	/* Don't think this can ever happen, todo double check
	if (real_t(n) / real_t(n_mini_batch) - int(n / n_mini_batch) != 0) {
		for (int i = 0; i < n - n / n_mini_batch * n_mini_batch; i++) {
			inputMiniBatches[n_mini_batch - 1].push_back(inputSet[n_mini_batch * n_mini_batch + i]);
		}
	}
	*/

	return input_mini_batches;
}
MLPPUtilities::CreateMiniBatchMVBatch MLPPUtilities::create_mini_batchesmv(const Ref<MLPPMatrix> &input_set, const Ref<MLPPVector> &output_set, int n_mini_batch) {
	Size2i size = input_set->size();

	int n = size.y;
	int mini_batch_element_count = n / n_mini_batch;

	Ref<MLPPVector> row_tmp;
	row_tmp.instance();
	row_tmp->resize(size.x);

	CreateMiniBatchMVBatch ret;

	for (int i = 0; i < n_mini_batch; i++) {
		int mini_batch_start_offset = mini_batch_element_count * i;
		Ref<MLPPMatrix> current_input_set;
		current_input_set.instance();
		current_input_set->resize(Size2i(size.x, mini_batch_element_count));

		Ref<MLPPVector> current_output_set;
		current_output_set.instance();
		current_output_set->resize(mini_batch_element_count);

		for (int j = 0; j < mini_batch_element_count; j++) {
			int main_indx = mini_batch_start_offset + j;

			input_set->row_get_into_mlpp_vector(main_indx, row_tmp);
			current_input_set->row_set_mlpp_vector(j, row_tmp);

			current_output_set->element_set(j, output_set->element_get(j));
		}

		ret.input_sets.push_back(current_input_set);
		ret.output_sets.push_back(current_output_set);
	}

	/* Don't think this can ever happen, todo double check
	if (real_t(n) / real_t(n_mini_batch) - int(n / n_mini_batch) != 0) {
		for (int i = 0; i < n - n / n_mini_batch * n_mini_batch; i++) {
			inputMiniBatches[n_mini_batch - 1].push_back(inputSet[n / n_mini_batch * n_mini_batch + i]);
			outputMiniBatches[n_mini_batch - 1].push_back(outputSet[n / n_mini_batch * n_mini_batch + i]);
		}
	}
	*/

	return ret;
}
MLPPUtilities::CreateMiniBatchMMBatch MLPPUtilities::create_mini_batchesmm(const Ref<MLPPMatrix> &input_set, const Ref<MLPPMatrix> &output_set, int n_mini_batch) {
	Size2i input_set_size = input_set->size();
	Size2i output_set_size = output_set->size();

	int n = input_set_size.y;
	int mini_batch_element_count = n / n_mini_batch;

	Ref<MLPPVector> input_row_tmp;
	input_row_tmp.instance();
	input_row_tmp->resize(input_set_size.x);

	Ref<MLPPVector> output_row_tmp;
	output_row_tmp.instance();
	output_row_tmp->resize(output_set_size.x);

	CreateMiniBatchMMBatch ret;

	for (int i = 0; i < n_mini_batch; i++) {
		int mini_batch_start_offset = n_mini_batch * i;
		Ref<MLPPMatrix> current_input_set;
		current_input_set.instance();
		current_input_set->resize(Size2i(input_set_size.x, mini_batch_element_count));

		Ref<MLPPMatrix> current_output_set;
		current_output_set.instance();
		current_output_set->resize(Size2i(output_set_size.x, mini_batch_element_count));

		for (int j = 0; j < mini_batch_element_count; j++) {
			int main_indx = mini_batch_start_offset + j;

			input_set->row_get_into_mlpp_vector(main_indx, input_row_tmp);
			current_input_set->row_set_mlpp_vector(j, input_row_tmp);

			output_set->row_get_into_mlpp_vector(main_indx, output_row_tmp);
			current_output_set->row_set_mlpp_vector(j, output_row_tmp);
		}

		ret.input_sets.push_back(current_input_set);
		ret.output_sets.push_back(current_output_set);
	}

	/* Don't think this can ever happen, todo double check
	if (real_t(n) / real_t(n_mini_batch) - int(n / n_mini_batch) != 0) {
		for (int i = 0; i < n - n / n_mini_batch * n_mini_batch; i++) {
			inputMiniBatches[n_mini_batch - 1].push_back(inputSet[n / n_mini_batch * n_mini_batch + i]);
		}
	}
	*/

	return ret;
}

Array MLPPUtilities::create_mini_batchesm_bind(const Ref<MLPPMatrix> &input_set, int n_mini_batch) {
	Vector<Ref<MLPPMatrix>> batches = create_mini_batchesm(input_set, n_mini_batch);

	Array ret;

	for (int i = 0; i < batches.size(); ++i) {
		ret.push_back(batches[i].get_ref_ptr());
	}

	return ret;
}
Array MLPPUtilities::create_mini_batchesmv_bind(const Ref<MLPPMatrix> &input_set, const Ref<MLPPVector> &output_set, int n_mini_batch) {
	CreateMiniBatchMVBatch batches = create_mini_batchesmv(input_set, output_set, n_mini_batch);

	Array inputs;
	Array outputs;

	for (int i = 0; i < batches.input_sets.size(); ++i) {
		inputs.push_back(batches.input_sets[i].get_ref_ptr());
		outputs.push_back(batches.output_sets[i].get_ref_ptr());
	}

	Array ret;

	ret.push_back(inputs);
	ret.push_back(outputs);

	return ret;
}
Array MLPPUtilities::create_mini_batchesmm_bind(const Ref<MLPPMatrix> &input_set, const Ref<MLPPMatrix> &output_set, int n_mini_batch) {
	CreateMiniBatchMMBatch batches = create_mini_batchesmm(input_set, output_set, n_mini_batch);

	Array inputs;
	Array outputs;

	for (int i = 0; i < batches.input_sets.size(); ++i) {
		inputs.push_back(batches.input_sets[i].get_ref_ptr());
		outputs.push_back(batches.output_sets[i].get_ref_ptr());
	}

	Array ret;

	ret.push_back(inputs);
	ret.push_back(outputs);

	return ret;
}

std::tuple<real_t, real_t, real_t, real_t> MLPPUtilities::TF_PN(std::vector<real_t> y_hat, std::vector<real_t> y) {
	real_t TP = 0;
	real_t FP = 0;
	real_t TN = 0;
	real_t FN = 0;
	for (uint32_t i = 0; i < y_hat.size(); i++) {
		if (y_hat[i] == y[i]) {
			if (y_hat[i] == 1) {
				TP++;
			} else {
				TN++;
			}
		} else {
			if (y_hat[i] == 1) {
				FP++;
			} else {
				FN++;
			}
		}
	}
	return { TP, FP, TN, FN };
}

real_t MLPPUtilities::recall(std::vector<real_t> y_hat, std::vector<real_t> y) {
	auto res = TF_PN(y_hat, y);
	auto TP = std::get<0>(res);
	//auto FP = std::get<1>(res);
	//auto TN = std::get<2>(res);
	auto FN = std::get<3>(res);

	return TP / (TP + FN);
}

real_t MLPPUtilities::precision(std::vector<real_t> y_hat, std::vector<real_t> y) {
	auto res = TF_PN(y_hat, y);
	auto TP = std::get<0>(res);
	auto FP = std::get<1>(res);
	//auto TN = std::get<2>(res);
	//auto FN = std::get<3>(res);

	return TP / (TP + FP);
}

real_t MLPPUtilities::accuracy(std::vector<real_t> y_hat, std::vector<real_t> y) {
	auto res = TF_PN(y_hat, y);
	auto TP = std::get<0>(res);
	auto FP = std::get<1>(res);
	auto TN = std::get<2>(res);
	auto FN = std::get<3>(res);

	return (TP + TN) / (TP + FP + FN + TN);
}
real_t MLPPUtilities::f1_score(std::vector<real_t> y_hat, std::vector<real_t> y) {
	return 2 * precision(y_hat, y) * recall(y_hat, y) / (precision(y_hat, y) + recall(y_hat, y));
}

void MLPPUtilities::_bind_methods() {
	ClassDB::bind_method(D_METHOD("weight_initializationv", "weights", "type"), &MLPPUtilities::weight_initializationv, WEIGHT_DISTRIBUTION_TYPE_DEFAULT);
	ClassDB::bind_method(D_METHOD("weight_initializationm", "weights", "type"), &MLPPUtilities::weight_initializationm, WEIGHT_DISTRIBUTION_TYPE_DEFAULT);
	ClassDB::bind_method(D_METHOD("bias_initializationr"), &MLPPUtilities::bias_initializationr);
	ClassDB::bind_method(D_METHOD("bias_initializationv", "z"), &MLPPUtilities::bias_initializationv);

	ClassDB::bind_method(D_METHOD("performance_vec", "y_hat", "output_set"), &MLPPUtilities::performance_vec);
	ClassDB::bind_method(D_METHOD("performance_mat", "y_hat", "y"), &MLPPUtilities::performance_mat);
	ClassDB::bind_method(D_METHOD("performance_pool_int_array_vec", "y_hat", "output_set"), &MLPPUtilities::performance_pool_int_array_vec);

	ClassDB::bind_method(D_METHOD("create_mini_batchesm", "input_set", "n_mini_batch"), &MLPPUtilities::create_mini_batchesm_bind);
	ClassDB::bind_method(D_METHOD("create_mini_batchesmv", "input_set", "output_set", "n_mini_batch"), &MLPPUtilities::create_mini_batchesmv_bind);
	ClassDB::bind_method(D_METHOD("create_mini_batchesmm", "input_set", "output_set", "n_mini_batch"), &MLPPUtilities::create_mini_batchesmm_bind);

	BIND_ENUM_CONSTANT(WEIGHT_DISTRIBUTION_TYPE_DEFAULT);
	BIND_ENUM_CONSTANT(WEIGHT_DISTRIBUTION_TYPE_XAVIER_NORMAL);
	BIND_ENUM_CONSTANT(WEIGHT_DISTRIBUTION_TYPE_XAVIER_UNIFORM);
	BIND_ENUM_CONSTANT(WEIGHT_DISTRIBUTION_TYPE_HE_NORMAL);
	BIND_ENUM_CONSTANT(WEIGHT_DISTRIBUTION_TYPE_HE_UNIFORM);
	BIND_ENUM_CONSTANT(WEIGHT_DISTRIBUTION_TYPE_LE_CUN_NORMAL);
	BIND_ENUM_CONSTANT(WEIGHT_DISTRIBUTION_TYPE_LE_CUN_UNIFORM);
	BIND_ENUM_CONSTANT(WEIGHT_DISTRIBUTION_TYPE_UNIFORM);
}
