/*************************************************************************/
/*  data.cpp                                                             */
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

#include "data.h"

#ifdef USING_SFW
#include "sfw.h"
#else
#include "core/os/file_access.h"
#endif

#include "../core/lin_alg.h"
#include "../core/stat.h"

// TODO move this to core?
#include "../modules/softmax_net/softmax_net.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

Ref<MLPPVector> MLPPDataESimple::get_input() {
	return _input;
}
void MLPPDataESimple::set_input(const Ref<MLPPVector> &val) {
	_input = val;
}

Ref<MLPPVector> MLPPDataESimple::get_output() {
	return _output;
}
void MLPPDataESimple::set_output(const Ref<MLPPVector> &val) {
	_output = val;
}

void MLPPDataESimple::instance_data() {
	_input.instance();
	_output.instance();
}

void MLPPDataESimple::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input"), &MLPPDataESimple::get_input);
	ClassDB::bind_method(D_METHOD("set_input", "val"), &MLPPDataESimple::set_input);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_input", "get_input");

	ClassDB::bind_method(D_METHOD("get_output"), &MLPPDataESimple::get_input);
	ClassDB::bind_method(D_METHOD("set_output", "val"), &MLPPDataESimple::set_output);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output", "get_output");

	ClassDB::bind_method(D_METHOD("instance_data"), &MLPPDataESimple::instance_data);
}

Ref<MLPPMatrix> MLPPDataSimple::get_input() {
	return _input;
}
void MLPPDataSimple::set_input(const Ref<MLPPMatrix> &val) {
	_input = val;
}

Ref<MLPPVector> MLPPDataSimple::get_output() {
	return _output;
}
void MLPPDataSimple::set_output(const Ref<MLPPVector> &val) {
	_output = val;
}

void MLPPDataSimple::instance_data() {
	_input.instance();
	_output.instance();
}

void MLPPDataSimple::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input"), &MLPPDataSimple::get_input);
	ClassDB::bind_method(D_METHOD("set_input", "val"), &MLPPDataSimple::set_input);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input", "get_input");

	ClassDB::bind_method(D_METHOD("get_output"), &MLPPDataSimple::get_input);
	ClassDB::bind_method(D_METHOD("set_output", "val"), &MLPPDataSimple::set_output);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output", PROPERTY_HINT_RESOURCE_TYPE, "MLPPVector"), "set_output", "get_output");

	ClassDB::bind_method(D_METHOD("instance_data"), &MLPPDataSimple::instance_data);
}

Ref<MLPPMatrix> MLPPDataComplex::get_input() {
	return _input;
}
void MLPPDataComplex::set_input(const Ref<MLPPMatrix> &val) {
	_input = val;
}

Ref<MLPPMatrix> MLPPDataComplex::get_output() {
	return _output;
}
void MLPPDataComplex::set_output(const Ref<MLPPMatrix> &val) {
	_output = val;
}

void MLPPDataComplex::instance_data() {
	_input.instance();
	_output.instance();
}

void MLPPDataComplex::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_input"), &MLPPDataComplex::get_input);
	ClassDB::bind_method(D_METHOD("set_input", "val"), &MLPPDataComplex::set_input);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "input", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_input", "get_input");

	ClassDB::bind_method(D_METHOD("get_output"), &MLPPDataComplex::get_input);
	ClassDB::bind_method(D_METHOD("set_output", "val"), &MLPPDataComplex::set_output);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "output", PROPERTY_HINT_RESOURCE_TYPE, "MLPPMatrix"), "set_output", "get_output");

	ClassDB::bind_method(D_METHOD("instance_data"), &MLPPDataComplex::instance_data);
}

// Loading Datasets
Ref<MLPPDataSimple> MLPPData::load_breast_cancer(const String &path) {
	const int BREAST_CANCER_SIZE = 30; // k = 30

	Ref<MLPPDataSimple> data;
	data.instance();
	data->instance_data();

	set_data_supervised(BREAST_CANCER_SIZE, path, data->get_input(), data->get_output());

	return data;
}

Ref<MLPPDataSimple> MLPPData::load_breast_cancer_svc(const String &path) {
	const int BREAST_CANCER_SIZE = 30; // k = 30

	Ref<MLPPDataSimple> data;
	data.instance();
	data->instance_data();

	set_data_supervised(BREAST_CANCER_SIZE, path, data->get_input(), data->get_output());

	return data;
}

Ref<MLPPDataComplex> MLPPData::load_iris(const String &path) {
	const int IRIS_SIZE = 4;
	const int ONE_HOT_NUM = 3;

	Ref<MLPPVector> temp_output_set;
	temp_output_set.instance();

	Ref<MLPPDataComplex> data;
	data.instance();
	data->instance_data();

	set_data_supervised(IRIS_SIZE, path, data->get_input(), temp_output_set);
	data->set_output(one_hot_rep(temp_output_set, ONE_HOT_NUM));

	return data;
}

Ref<MLPPDataComplex> MLPPData::load_wine(const String &path) {
	const int WINE_SIZE = 4;
	const int ONE_HOT_NUM = 3;

	Ref<MLPPVector> temp_output_set;
	temp_output_set.instance();

	Ref<MLPPDataComplex> data;
	data.instance();
	data->instance_data();

	set_data_supervised(WINE_SIZE, path, data->get_input(), temp_output_set);
	data->set_output(one_hot_rep(temp_output_set, ONE_HOT_NUM));

	return data;
}

Ref<MLPPDataComplex> MLPPData::load_mnist_train(const String &path) {
	const int MNIST_SIZE = 784;
	const int ONE_HOT_NUM = 10;

	Ref<MLPPVector> temp_output_set;
	temp_output_set.instance();

	Ref<MLPPDataComplex> data;
	data.instance();
	data->instance_data();

	set_data_supervised(MNIST_SIZE, path, data->get_input(), temp_output_set);
	data->set_output(one_hot_rep(temp_output_set, ONE_HOT_NUM));

	return data;
}

Ref<MLPPDataComplex> MLPPData::load_mnist_test(const String &path) {
	const int MNIST_SIZE = 784;
	const int ONE_HOT_NUM = 10;

	Ref<MLPPVector> temp_output_set;
	temp_output_set.instance();

	Ref<MLPPDataComplex> data;
	data.instance();
	data->instance_data();

	set_data_supervised(MNIST_SIZE, path, data->get_input(), temp_output_set);
	data->set_output(one_hot_rep(temp_output_set, ONE_HOT_NUM));

	return data;
}

Ref<MLPPDataSimple> MLPPData::load_california_housing(const String &path) {
	const int CALIFORNIA_HOUSING_SIZE = 13; // k = 30

	Ref<MLPPDataSimple> data;
	data.instance();
	data->instance_data();

	set_data_supervised(CALIFORNIA_HOUSING_SIZE, path, data->get_input(), data->get_output());

	return data;
}

Ref<MLPPDataESimple> MLPPData::load_fires_and_crime(const String &path) {
	// k is implicitly 1.

	Ref<MLPPDataESimple> data;
	data.instance();
	data->instance_data();

	set_data_simple(path, data->get_input(), data->get_output());

	return data;
}

// MULTIVARIATE SUPERVISED

void MLPPData::set_data_supervised(int k, const String &file_name, Ref<MLPPMatrix> input_set, Ref<MLPPVector> output_set) {
	ERR_FAIL_COND(!input_set.is_valid() || !output_set.is_valid());

	MLPPLinAlg alg;

	Vector<Vector<real_t>> input_set_tmp;
	Vector<real_t> output_set_tmp;

#ifdef USING_SFW
	FileAccess *file = FileAccess::create_and_open(file_name, FileAccess::READ);
#else
	FileAccess *file = FileAccess::open(file_name, FileAccess::READ);
#endif

	ERR_FAIL_COND(!file);

	while (!file->eof_reached()) {
		Vector<String> ll = file->get_csv_line();

		Vector<real_t> row;

		for (int i = 0; i < k; ++i) {
			row.push_back(static_cast<real_t>(ll[i].to_double()));
		}

		input_set_tmp.push_back(row);
		output_set_tmp.push_back(static_cast<real_t>(ll[k].to_double()));
	}

	file->close();
	memdelete(file);

	output_set->set_from_vector(output_set_tmp);
	input_set->set_from_vectors(input_set_tmp);
}

void MLPPData::set_data_unsupervised(int k, const String &file_name, Ref<MLPPMatrix> input_set) {
	ERR_FAIL_COND(!input_set.is_valid());

	MLPPLinAlg alg;

	Vector<Vector<real_t>> input_set_tmp;
	input_set_tmp.resize(k);

#ifdef USING_SFW
	FileAccess *file = FileAccess::create_and_open(file_name, FileAccess::READ);
#else
	FileAccess *file = FileAccess::open(file_name, FileAccess::READ);
#endif

	ERR_FAIL_COND(!file);

	while (!file->eof_reached()) {
		Vector<String> ll = file->get_csv_line();

		for (int i = 0; i < k; ++i) {
			input_set_tmp.write[i].push_back(static_cast<real_t>(ll[i].to_double()));
		}
	}

	file->close();
	memdelete(file);

	input_set->set_from_vectors(input_set_tmp);
	input_set = alg.transposenm(input_set);
}

void MLPPData::set_data_simple(const String &file_name, Ref<MLPPVector> input_set, Ref<MLPPVector> output_set) {
	ERR_FAIL_COND(!input_set.is_valid() || !output_set.is_valid());

#ifdef USING_SFW
	FileAccess *file = FileAccess::create_and_open(file_name, FileAccess::READ);
#else
	FileAccess *file = FileAccess::open(file_name, FileAccess::READ);
#endif

	ERR_FAIL_COND(!file);

	Vector<real_t> input_set_tmp;
	Vector<real_t> output_set_tmp;

	while (!file->eof_reached()) {
		Vector<String> ll = file->get_csv_line();

		for (int i = 0; i < ll.size(); i += 2) {
			input_set_tmp.push_back(static_cast<real_t>(ll[i].to_double()));
			output_set_tmp.push_back(static_cast<real_t>(ll[i + 1].to_double()));
		}
	}

	file->close();
	memdelete(file);

	input_set->set_from_vector(input_set_tmp);
	output_set->set_from_vector(output_set_tmp);
}

MLPPData::SplitComplexData MLPPData::train_test_split(Ref<MLPPDataComplex> data, real_t test_size) {
	SplitComplexData res;

	res.train.instance();
	res.train->instance_data();
	res.test.instance();
	res.test->instance_data();

	ERR_FAIL_COND_V(!data.is_valid(), res);

	Ref<MLPPMatrix> orig_input = data->get_input();
	Ref<MLPPMatrix> orig_output = data->get_output();

	ERR_FAIL_COND_V(!orig_input.is_valid(), res);
	ERR_FAIL_COND_V(!orig_output.is_valid(), res);

	Size2i orig_input_size = orig_input->size();
	Size2i orig_output_size = orig_output->size();

	int is = MIN(orig_input_size.y, orig_output_size.y);

	Array indices;
	indices.resize(is);

	for (int i = 0; i < is; ++i) {
		indices[i] = i;
	}

	indices.shuffle();

	Ref<MLPPVector> orig_input_row_tmp;
	orig_input_row_tmp.instance();
	orig_input_row_tmp->resize(orig_input_size.x);

	Ref<MLPPVector> orig_output_row_tmp;
	orig_output_row_tmp.instance();
	orig_output_row_tmp->resize(orig_output_size.x);

	int test_input_number = test_size * is; // implicit usage of floor

	Ref<MLPPMatrix> res_test_input = res.test->get_input();
	Ref<MLPPMatrix> res_test_output = res.test->get_output();

	res_test_input->resize(Size2i(orig_input_size.x, test_input_number));
	res_test_output->resize(Size2i(orig_output_size.x, test_input_number));

	for (int i = 0; i < test_input_number; ++i) {
		int index = indices[i];

		orig_input->row_get_into_mlpp_vector(index, orig_input_row_tmp);
		orig_output->row_get_into_mlpp_vector(index, orig_output_row_tmp);

		res_test_input->row_set_mlpp_vector(i, orig_input_row_tmp);
		res_test_output->row_set_mlpp_vector(i, orig_output_row_tmp);
	}

	Ref<MLPPMatrix> res_train_input = res.train->get_input();
	Ref<MLPPMatrix> res_train_output = res.train->get_output();

	int train_input_number = is - test_input_number;

	res_train_input->resize(Size2i(orig_input_size.x, train_input_number));
	res_train_output->resize(Size2i(orig_output_size.x, train_input_number));

	for (int i = 0; i < train_input_number; ++i) {
		int index = indices[test_input_number + i];

		orig_input->row_get_into_mlpp_vector(index, orig_input_row_tmp);
		orig_output->row_get_into_mlpp_vector(index, orig_output_row_tmp);

		res_train_input->row_set_mlpp_vector(i, orig_input_row_tmp);
		res_train_output->row_set_mlpp_vector(i, orig_output_row_tmp);
	}

	return res;
}
Array MLPPData::train_test_split_bind(const Ref<MLPPDataComplex> &data, real_t test_size) {
	SplitComplexData res = train_test_split(data, test_size);

	Array arr;
	arr.push_back(res.train);
	arr.push_back(res.test);

	return arr;
}

// Images
std::vector<std::vector<real_t>> MLPPData::rgb2gray(std::vector<std::vector<std::vector<real_t>>> input) {
	/*
	std::vector<std::vector<real_t>> grayScale;
	grayScale.resize(input[0].size());
	for (uint32_t i = 0; i < grayScale.size(); i++) {
		grayScale[i].resize(input[0][i].size());
	}
	for (uint32_t i = 0; i < grayScale.size(); i++) {
		for (uint32_t j = 0; j < grayScale[i].size(); j++) {
			grayScale[i][j] = 0.299 * input[0][i][j] + 0.587 * input[1][i][j] + 0.114 * input[2][i][j];
		}
	}
	return grayScale;
	*/

	return std::vector<std::vector<real_t>>();
}

std::vector<std::vector<std::vector<real_t>>> MLPPData::rgb2ycbcr(std::vector<std::vector<std::vector<real_t>>> input) {
	/*
	MLPPLinAlgOld alg;
	std::vector<std::vector<std::vector<real_t>>> YCbCr;
	YCbCr = alg.resize(YCbCr, input);
	for (uint32_t i = 0; i < YCbCr[0].size(); i++) {
		for (uint32_t j = 0; j < YCbCr[0][i].size(); j++) {
			YCbCr[0][i][j] = 0.299 * input[0][i][j] + 0.587 * input[1][i][j] + 0.114 * input[2][i][j];
			YCbCr[1][i][j] = -0.169 * input[0][i][j] - 0.331 * input[1][i][j] + 0.500 * input[2][i][j];
			YCbCr[2][i][j] = 0.500 * input[0][i][j] - 0.419 * input[1][i][j] - 0.081 * input[2][i][j];
		}
	}
	return YCbCr;
	*/

	return std::vector<std::vector<std::vector<real_t>>>();
}

// Conversion formulas available here:
// https://www.rapidtables.com/convert/color/rgb-to-hsv.html
std::vector<std::vector<std::vector<real_t>>> MLPPData::rgb2hsv(std::vector<std::vector<std::vector<real_t>>> input) {
	/*
	MLPPLinAlgOld alg;
	std::vector<std::vector<std::vector<real_t>>> HSV;
	HSV = alg.resize(HSV, input);
	for (uint32_t i = 0; i < HSV[0].size(); i++) {
		for (uint32_t j = 0; j < HSV[0][i].size(); j++) {
			real_t rPrime = input[0][i][j] / 255;
			real_t gPrime = input[1][i][j] / 255;
			real_t bPrime = input[2][i][j] / 255;

			real_t cMax = alg.max({ rPrime, gPrime, bPrime });
			real_t cMin = alg.min({ rPrime, gPrime, bPrime });
			real_t delta = cMax - cMin;

			// H calculation.
			if (delta == 0) {
				HSV[0][i][j] = 0;
			} else {
				if (cMax == rPrime) {
					HSV[0][i][j] = 60 * fmod(((gPrime - bPrime) / delta), 6);
				} else if (cMax == gPrime) {
					HSV[0][i][j] = 60 * ((bPrime - rPrime) / delta + 2);
				} else { // cMax == bPrime
					HSV[0][i][j] = 60 * ((rPrime - gPrime) / delta + 6);
				}
			}

			// S calculation.
			if (cMax == 0) {
				HSV[1][i][j] = 0;
			} else {
				HSV[1][i][j] = delta / cMax;
			}

			// V calculation.
			HSV[2][i][j] = cMax;
		}
	}
	return HSV;
	*/

	return std::vector<std::vector<std::vector<real_t>>>();
}

// http://machinethatsees.blogspot.com/2013/07/how-to-convert-rgb-to-xyz-or-vice-versa.html
std::vector<std::vector<std::vector<real_t>>> MLPPData::rgb2xyz(std::vector<std::vector<std::vector<real_t>>> input) {
	/*
	MLPPLinAlgOld alg;
	std::vector<std::vector<std::vector<real_t>>> XYZ;
	XYZ = alg.resize(XYZ, input);
	std::vector<std::vector<real_t>> RGB2XYZ = { { 0.4124564, 0.3575761, 0.1804375 }, { 0.2126726, 0.7151522, 0.0721750 }, { 0.0193339, 0.1191920, 0.9503041 } };
	return alg.vector_wise_tensor_product(input, RGB2XYZ);
	*/

	return std::vector<std::vector<std::vector<real_t>>>();
}

std::vector<std::vector<std::vector<real_t>>> MLPPData::xyz2rgb(std::vector<std::vector<std::vector<real_t>>> input) {
	/*
	MLPPLinAlgOld alg;
	std::vector<std::vector<std::vector<real_t>>> XYZ;
	XYZ = alg.resize(XYZ, input);
	std::vector<std::vector<real_t>> RGB2XYZ = alg.inverse({ { 0.4124564, 0.3575761, 0.1804375 }, { 0.2126726, 0.7151522, 0.0721750 }, { 0.0193339, 0.1191920, 0.9503041 } });
	return alg.vector_wise_tensor_product(input, RGB2XYZ);
	*/

	return std::vector<std::vector<std::vector<real_t>>>();
}

// TEXT-BASED & NLP
std::string MLPPData::toLower(std::string text) {
	for (uint32_t i = 0; i < text.size(); i++) {
		text[i] = tolower(text[i]);
	}
	return text;
}

std::vector<char> MLPPData::split(std::string text) {
	std::vector<char> split_data;
	for (uint32_t i = 0; i < text.size(); i++) {
		split_data.push_back(text[i]);
	}
	return split_data;
}

Vector<String> MLPPData::split_sentences(String data) {
	Vector<String> sentences;

	int start_index = 0;

	for (int i = 0; i < data.length() - 1; ++i) {
		if (data[i] == '.' && data[i + 1] != '.') {
			continue;
		}

		if (data[i] == '.') {
			sentences.push_back(data.substr_index(start_index, i));
			start_index = i + 1;
		}
	}

	if (start_index != data.length() - 1) {
		sentences.push_back(data.substr_index(start_index, data.length() - 1));
	}

	return sentences;
}

Vector<String> MLPPData::remove_spaces(Vector<String> data) {
	for (int i = 0; i < data.size(); i++) {
		data.write[i] = data[i].replace(" ", "");
	}
	return data;
}

Vector<String> MLPPData::remove_empty(Vector<String> data) {
	for (int i = 0; i < data.size(); ++i) {
		if (data[i].empty()) {
			data.remove(i);
		}
	}

	return data;
}

Vector<String> MLPPData::segment(String text) {
	Vector<String> segmented_data;
	int prev_delim = 0;

	for (int i = 0; i < text.length(); i++) {
		if (text[i] == ' ') {
			segmented_data.push_back(text.substr(prev_delim, i - prev_delim));
			prev_delim = i + 1;
		} else if (text[i] == ',' || text[i] == '!' || text[i] == '.' || text[i] == '-') {
			segmented_data.push_back(text.substr(prev_delim, i - prev_delim));
			String punc;
			punc += text[i];
			segmented_data.push_back(punc);
			prev_delim = i + 2;
			i++;
		} else if (i == text.length() - 1) {
			segmented_data.push_back(text.substr(prev_delim, text.length() - prev_delim)); // hehe oops- forgot this
		}
	}

	return segmented_data;
}

Vector<int> MLPPData::tokenize(String text) {
	int max_num = 0;
	bool new_num = true;
	Vector<String> segmented_data = segment(text);
	Vector<int> tokenized_data;
	tokenized_data.resize(segmented_data.size());

	for (int i = 0; i < segmented_data.size(); i++) {
		for (int j = i - 1; j >= 0; j--) {
			if (segmented_data[i] == segmented_data[j]) {
				tokenized_data.write[i] = tokenized_data[j];
				new_num = false;
			}
		}
		if (!new_num) {
			new_num = true;
		} else {
			max_num++;
			tokenized_data.write[i] = max_num;
		}
	}

	return tokenized_data;
}

Vector<String> MLPPData::remove_stop_words(String text) {
	Vector<String> segmented_data = remove_spaces(segment(text.to_lower()));

	for (int i = 0; i < stop_words.size(); i++) {
		for (int j = 0; j < segmented_data.size(); j++) {
			if (segmented_data[j] == stop_words[i]) {
				segmented_data.remove(j);
				--j;
			}
		}
	}

	return segmented_data;
}

Vector<String> MLPPData::remove_stop_words_vec(Vector<String> segmented_data) {
	for (int i = 0; i < segmented_data.size(); i++) {
		for (int j = 0; j < stop_words.size(); j++) {
			if (segmented_data[i] == stop_words[j]) {
				segmented_data.remove(i);
				--i;
			}
		}
	}

	return segmented_data;
}

String MLPPData::stemming(String text) {
	int padding_size = 4;
	String padding = " "; // our padding

	text += String(padding).repeat(padding_size); // ' ' will be our padding value

	for (int i = 0; i < text.length(); i++) {
		for (int j = 0; j < suffixes.size(); j++) {
			if (text.substr(i, suffixes[j].length()) == suffixes[j] && (text[i + suffixes[j].length()] == ' ' || text[i + suffixes[j].length()] == ',' || text[i + suffixes[j].length()] == '-' || text[i + suffixes[j].length()] == '.' || text[i + suffixes[j].length()] == '!')) {
				text.erase(i, suffixes[j].length());
			}
		}
	}

	return text;
}

Ref<MLPPMatrix> MLPPData::bag_of_words(Vector<String> sentences, BagOfWordsType type) {
	/*
	STEPS OF BOW:
		1) To lowercase (done by remove_stop_words function by def)
		2) Removing stop words
		3) Obtain a list of the used words
		4) Create a one hot encoded vector of the words and sentences
		5) Sentence.size() x list.size() matrix
	*/

	Vector<String> word_list = remove_empty(remove_stop_words_vec(create_word_list(sentences)));

	Vector<Vector<String>> segmented_sentences;
	segmented_sentences.resize(sentences.size());

	for (int i = 0; i < sentences.size(); i++) {
		segmented_sentences.write[i] = remove_stop_words(sentences[i]);
	}

	Ref<MLPPMatrix> bow;
	bow.instance();
	bow->resize(Size2i(word_list.size(), segmented_sentences.size()));
	bow->fill(0);

	for (int i = 0; i < segmented_sentences.size(); i++) {
		for (int j = 0; j < segmented_sentences[i].size(); j++) {
			for (int k = 0; k < word_list.size(); k++) {
				if (segmented_sentences[i][j] == word_list[k]) {
					if (type == BAG_OF_WORDS_TYPE_BINARY) {
						bow->element_set(i, k, 1);
					} else {
						bow->element_set(i, k, bow->element_get(i, k) + 1);
					}
				}
			}
		}
	}

	return bow;
}

Ref<MLPPMatrix> MLPPData::tfidf(Vector<String> sentences) {
	Vector<String> word_list = remove_empty(remove_stop_words_vec(create_word_list(sentences)));

	Vector<Vector<String>> segmented_sentences;
	segmented_sentences.resize(sentences.size());

	for (int i = 0; i < sentences.size(); i++) {
		segmented_sentences.write[i] = remove_stop_words(sentences[i]);
	}

	Ref<MLPPMatrix> TF;
	TF.instance();
	TF->resize(Size2i(word_list.size(), segmented_sentences.size()));

	Vector<int> frequency;
	frequency.resize(word_list.size());
	frequency.fill(0);

	Ref<MLPPVector> TF_row;
	TF_row.instance();
	TF_row->resize(word_list.size());

	for (int i = 0; i < segmented_sentences.size(); i++) {
		Vector<bool> present;
		present.resize(word_list.size());
		present.fill(false);

		for (int j = 0; j < segmented_sentences[i].size(); j++) {
			for (int k = 0; k < word_list.size(); k++) {
				if (segmented_sentences[i][j] == word_list[k]) {
					TF->element_set(i, k, TF->element_get(i, k) + 1);

					if (!present[k]) {
						frequency.write[k]++;
						present.write[k] = true;
					}
				}
			}
		}

		TF->row_get_into_mlpp_vector(i, TF_row);
		TF_row->scalar_multiply(real_t(1) / real_t(segmented_sentences[i].size()));
		TF->row_set_mlpp_vector(i, TF_row);
	}

	Vector<real_t> IDF;
	IDF.resize(frequency.size());

	for (int i = 0; i < IDF.size(); i++) {
		IDF.write[i] = Math::log((real_t)segmented_sentences.size() / (real_t)frequency[i]);
	}

	Ref<MLPPMatrix> TFIDF;
	TFIDF.instance();
	Size2i tfidf_size = Size2i(word_list.size(), segmented_sentences.size());
	TFIDF->resize(tfidf_size);

	for (int i = 0; i < tfidf_size.y; i++) {
		for (int j = 0; j < tfidf_size.x; j++) {
			TFIDF->element_set(i, j, TF->element_get(i, j) * IDF[j]);
		}
	}

	return TFIDF;
}

MLPPData::WordsToVecResult MLPPData::word_to_vec(Vector<String> sentences, WordToVecType type, int windowSize, int dimension, real_t learning_rate, int max_epoch) {
	WordsToVecResult res;

	res.word_list = remove_empty(remove_stop_words_vec(create_word_list(sentences)));

	Vector<Vector<String>> segmented_sentences;
	segmented_sentences.resize(sentences.size());

	for (int i = 0; i < sentences.size(); i++) {
		segmented_sentences.write[i] = remove_stop_words(sentences[i]);
	}

	Vector<String> inputStrings;
	Vector<String> outputStrings;

	for (int i = 0; i < segmented_sentences.size(); i++) {
		for (int j = 0; j < segmented_sentences[i].size(); j++) {
			for (int k = windowSize; k > 0; k--) {
				int jmk = (int)j - k;

				if (jmk >= 0) {
					inputStrings.push_back(segmented_sentences[i][j]);

					outputStrings.push_back(segmented_sentences[i][jmk]);
				}
				if (j + k <= segmented_sentences[i].size() - 1) {
					inputStrings.push_back(segmented_sentences[i][j]);
					outputStrings.push_back(segmented_sentences[i][j + k]);
				}
			}
		}
	}

	int input_size = inputStrings.size();

	inputStrings.append_array(outputStrings);

	Ref<MLPPMatrix> bow = bag_of_words(inputStrings, BAG_OF_WORDS_TYPE_BINARY);
	Size2i bow_size = bow->size();

	Ref<MLPPMatrix> input_set;
	Ref<MLPPMatrix> output_set;

	input_set.instance();
	output_set.instance();

	input_set->resize(Size2i(bow_size.x, input_size));

	Ref<MLPPVector> row_tmp;
	row_tmp.instance();
	row_tmp->resize(bow_size.x);

	for (int i = 0; i < input_size; i++) {
		bow->row_get_into_mlpp_vector(i, row_tmp);
		input_set->row_set_mlpp_vector(i, row_tmp);
	}

	output_set->resize(Size2i(bow_size.x, bow_size.y - input_size));
	Size2i output_set_size = output_set->size();

	for (int i = 0; i < output_set_size.y; i++) {
		bow->row_get_into_mlpp_vector(i + input_size, row_tmp);
		input_set->row_set_mlpp_vector(i, row_tmp);
	}

	MLPPSoftmaxNet *model;

	if (type == WORD_TO_VEC_TYPE_SKIPGRAM) {
		model = memnew(MLPPSoftmaxNet(output_set, input_set, dimension));
	} else { // else = CBOW. We maintain it is a default.
		model = memnew(MLPPSoftmaxNet(input_set, output_set, dimension));
	}

	model->train_gradient_descent(learning_rate, max_epoch);

	res.word_embeddings = model->get_embeddings();
	memdelete(model);

	return res;
}

Ref<MLPPMatrix> MLPPData::lsa(Vector<String> sentences, int dim) {
	MLPPLinAlg alg;

	Ref<MLPPMatrix> doc_word_data = bag_of_words(sentences, BAG_OF_WORDS_TYPE_BINARY);

	MLPPLinAlg::SVDResult svr_res = alg.svd(doc_word_data);

	Ref<MLPPMatrix> S_trunc = alg.zeromatnm(dim, dim);
	Ref<MLPPMatrix> Vt_trunc;
	Vt_trunc.instance();
	Vt_trunc->resize(Size2i(svr_res.Vt->size().x, dim));

	Ref<MLPPVector> row_rmp;
	row_rmp.instance();
	row_rmp->resize(svr_res.Vt->size().x);

	for (int i = 0; i < dim; i++) {
		S_trunc->element_set(i, i, svr_res.S->element_get(i, i));

		svr_res.Vt->row_get_into_mlpp_vector(i, row_rmp);
		Vt_trunc->row_set_mlpp_vector(i, row_rmp);
	}

	Ref<MLPPMatrix> embeddings = S_trunc->multn(Vt_trunc);
	return embeddings;
}

Vector<String> MLPPData::create_word_list(Vector<String> sentences) {
	String combined_text = "";

	for (int i = 0; i < sentences.size(); i++) {
		if (i != 0) {
			combined_text += " ";
		}

		combined_text += sentences[i];
	}

	return remove_spaces(vec_to_set(remove_stop_words(combined_text)));
}

// EXTRA
void MLPPData::setInputNames(std::string fileName, std::vector<std::string> &inputNames) {
	std::string inputNameTemp;
	std::ifstream dataFile(fileName);
	if (!dataFile.is_open()) {
		std::cout << fileName << " failed to open." << std::endl;
	}

	while (std::getline(dataFile, inputNameTemp)) {
		inputNames.push_back(inputNameTemp);
	}

	dataFile.close();
}

Ref<MLPPMatrix> MLPPData::feature_scaling(const Ref<MLPPMatrix> &p_X) {
	Ref<MLPPMatrix> X = p_X->transposen();

	Size2i x_size = X->size();

	LocalVector<real_t> max_elements;
	LocalVector<real_t> min_elements;

	max_elements.resize(x_size.y);
	min_elements.resize(x_size.y);

	Ref<MLPPVector> row_tmp;
	row_tmp.instance();
	row_tmp->resize(x_size.x);

	for (int i = 0; i < x_size.y; ++i) {
		X->row_get_into_mlpp_vector(i, row_tmp);

		max_elements[i] = row_tmp->max_element();
		min_elements[i] = row_tmp->min_element();
	}

	for (int i = 0; i < x_size.y; i++) {
		real_t maxe = max_elements[i];
		real_t mine = min_elements[i];

		for (int j = 0; j < x_size.x; j++) {
			real_t xij = X->element_get(i, j);

			X->element_set(i, j, (xij - mine) / (maxe - mine));
		}
	}

	return X->transposen();
}

Ref<MLPPMatrix> MLPPData::mean_centering(const Ref<MLPPMatrix> &p_X) {
	MLPPStat stat;

	Ref<MLPPMatrix> X;
	X.instance();
	X->resize(p_X->size());

	Size2i x_size = X->size();

	Ref<MLPPVector> x_row_tmp;
	x_row_tmp.instance();
	x_row_tmp->resize(x_size.x);

	for (int i = 0; i < x_size.y; ++i) {
		p_X->row_get_into_mlpp_vector(i, x_row_tmp);

		real_t mean_i = stat.meanv(x_row_tmp);

		for (int j = 0; j < x_size.x; ++j) {
			X->element_set(i, j, p_X->element_get(i, j) - mean_i);
		}
	}

	return X;
}

Ref<MLPPMatrix> MLPPData::mean_normalization(const Ref<MLPPMatrix> &p_X) {
	MLPPLinAlg alg;
	MLPPStat stat;

	// (X_j - mu_j) / std_j, for every j

	Ref<MLPPMatrix> X = mean_centering(p_X);
	Size2i x_size = X->size();

	Ref<MLPPVector> x_row_tmp;
	x_row_tmp.instance();
	x_row_tmp->resize(x_size.x);

	for (int i = 0; i < x_size.y; i++) {
		X->row_get_into_mlpp_vector(i, x_row_tmp);

		x_row_tmp->scalar_multiply((real_t)1 / stat.standard_deviationv(x_row_tmp));

		X->row_set_mlpp_vector(i, x_row_tmp);
	}

	return X;
}

Ref<MLPPMatrix> MLPPData::one_hot_rep(const Ref<MLPPVector> &temp_output_set, int n_class) {
	ERR_FAIL_COND_V(!temp_output_set.is_valid(), Ref<MLPPMatrix>());

	Ref<MLPPMatrix> output_set;
	output_set.instance();

	int temp_output_set_size = temp_output_set->size();
	const real_t *temp_output_set_ptr = temp_output_set->ptr();

	output_set->resize(Size2i(n_class, temp_output_set_size));

	for (int i = 0; i < temp_output_set_size; ++i) {
		for (int j = 0; j <= n_class - 1; ++j) {
			if (static_cast<int>(temp_output_set_ptr[i]) == j) {
				output_set->element_set(i, j, 1);
			} else {
				output_set->element_set(i, j, 0);
			}
		}
	}

	return output_set;
}

std::vector<real_t> MLPPData::reverseOneHot(std::vector<std::vector<real_t>> tempOutputSet) {
	std::vector<real_t> outputSet;
	//uint32_t n_class = tempOutputSet[0].size();
	for (uint32_t i = 0; i < tempOutputSet.size(); i++) {
		int current_class = 1;
		for (uint32_t j = 0; j < tempOutputSet[i].size(); j++) {
			if (tempOutputSet[i][j] == 1) {
				break;
			} else {
				current_class++;
			}
		}
		outputSet.push_back(current_class);
	}

	return outputSet;
}

void MLPPData::load_default_suffixes() {
	// Our list of suffixes which we use to compare against
	suffixes = String("eer er ion ity ment ness or sion ship th able ible al ant ary ful ic ious ous ive less y ed en ing ize ise ly ward wise").split_spaces();
}

void MLPPData::load_default_stop_words() {
	stop_words = String("i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t can will just don should now").split_spaces();
}

void MLPPData::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_breast_cancer", "path"), &MLPPData::load_breast_cancer);
	ClassDB::bind_method(D_METHOD("load_breast_cancer_svc", "path"), &MLPPData::load_breast_cancer_svc);
	ClassDB::bind_method(D_METHOD("load_iris", "path"), &MLPPData::load_iris);
	ClassDB::bind_method(D_METHOD("load_wine", "path"), &MLPPData::load_wine);
	ClassDB::bind_method(D_METHOD("load_mnist_train", "path"), &MLPPData::load_mnist_train);
	ClassDB::bind_method(D_METHOD("load_mnist_test", "path"), &MLPPData::load_mnist_test);
	ClassDB::bind_method(D_METHOD("load_california_housing", "path"), &MLPPData::load_california_housing);
	ClassDB::bind_method(D_METHOD("load_fires_and_crime", "path"), &MLPPData::load_fires_and_crime);

	ClassDB::bind_method(D_METHOD("train_test_split", "data", "test_size"), &MLPPData::train_test_split_bind);
}
