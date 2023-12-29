#ifndef MLPP_DATA_H
#define MLPP_DATA_H

/*************************************************************************/
/*  data.h                                                               */
/*************************************************************************/
/*                         This file is part of:                         */
/*                    PMLPP Machine Learning Library                     */
/*                   https://github.com/Relintai/pmlpp                   */
/*************************************************************************/
/* Copyright (c) 2022-present Péter Magyar.                              */
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

#include "core/math/math_defs.h"

#include "core/string/ustring.h"
#include "core/variant/array.h"

#include "core/object/reference.h"

#include "../lin_alg/mlpp_matrix.h"
#include "../lin_alg/mlpp_vector.h"

#include <string>
#include <tuple>
#include <vector>

class MLPPDataESimple : public Reference {
	GDCLASS(MLPPDataESimple, Reference);

public:
	Ref<MLPPVector> get_input();
	void set_input(const Ref<MLPPVector> &val);

	Ref<MLPPVector> get_output();
	void set_output(const Ref<MLPPVector> &val);

	void instance_data();

protected:
	static void _bind_methods();

	Ref<MLPPVector> _input;
	Ref<MLPPVector> _output;
};

class MLPPDataSimple : public Reference {
	GDCLASS(MLPPDataSimple, Reference);

public:
	Ref<MLPPMatrix> get_input();
	void set_input(const Ref<MLPPMatrix> &val);

	Ref<MLPPVector> get_output();
	void set_output(const Ref<MLPPVector> &val);

	void instance_data();

protected:
	static void _bind_methods();

	Ref<MLPPMatrix> _input;
	Ref<MLPPVector> _output;
};

class MLPPDataComplex : public Reference {
	GDCLASS(MLPPDataComplex, Reference);

public:
	Ref<MLPPMatrix> get_input();
	void set_input(const Ref<MLPPMatrix> &val);

	Ref<MLPPMatrix> get_output();
	void set_output(const Ref<MLPPMatrix> &val);

	void instance_data();

protected:
	static void _bind_methods();

	Ref<MLPPMatrix> _input;
	Ref<MLPPMatrix> _output;
};

class MLPPData : public Reference {
	GDCLASS(MLPPData, Reference);

public:
	// Load Datasets
	Ref<MLPPDataSimple> load_breast_cancer(const String &path);
	Ref<MLPPDataSimple> load_breast_cancer_svc(const String &path);
	Ref<MLPPDataComplex> load_iris(const String &path);
	Ref<MLPPDataComplex> load_wine(const String &path);
	Ref<MLPPDataComplex> load_mnist_train(const String &path);
	Ref<MLPPDataComplex> load_mnist_test(const String &path);
	Ref<MLPPDataSimple> load_california_housing(const String &path);
	Ref<MLPPDataESimple> load_fires_and_crime(const String &path);

	void set_data_supervised(int k, const String &file_name, Ref<MLPPMatrix> input_set, Ref<MLPPVector> output_set);
	void set_data_unsupervised(int k, const String &file_name, Ref<MLPPMatrix> input_set);
	void set_data_simple(const String &file_name, Ref<MLPPVector> input_set, Ref<MLPPVector> output_set);

	struct SplitComplexData {
		Ref<MLPPDataComplex> train;
		Ref<MLPPDataComplex> test;
	};

	SplitComplexData train_test_split(Ref<MLPPDataComplex> data, real_t test_size);
	Array train_test_split_bind(const Ref<MLPPDataComplex> &data, real_t test_size);

	// Images
	std::vector<std::vector<real_t>> rgb2gray(std::vector<std::vector<std::vector<real_t>>> input);
	std::vector<std::vector<std::vector<real_t>>> rgb2ycbcr(std::vector<std::vector<std::vector<real_t>>> input);
	std::vector<std::vector<std::vector<real_t>>> rgb2hsv(std::vector<std::vector<std::vector<real_t>>> input);
	std::vector<std::vector<std::vector<real_t>>> rgb2xyz(std::vector<std::vector<std::vector<real_t>>> input);
	std::vector<std::vector<std::vector<real_t>>> xyz2rgb(std::vector<std::vector<std::vector<real_t>>> input);

	// Text-Based & NLP
	std::string toLower(std::string text);
	std::vector<char> split(std::string text);
	Vector<String> split_sentences(String data);
	Vector<String> remove_spaces(Vector<String> data);
	Vector<String> remove_empty(Vector<String> data);
	Vector<String> segment(String text);
	Vector<int> tokenize(String text);
	Vector<String> remove_stop_words(String text);
	Vector<String> remove_stop_words_vec(Vector<String> segmented_data);

	String stemming(String text);

	enum BagOfWordsType {
		BAG_OF_WORDS_TYPE_DEFAULT = 0,
		BAG_OF_WORDS_TYPE_BINARY,
	};

	Ref<MLPPMatrix> bag_of_words(Vector<String> sentences, BagOfWordsType type = BAG_OF_WORDS_TYPE_DEFAULT);
	Ref<MLPPMatrix> tfidf(Vector<String> sentences);

	struct WordsToVecResult {
		Ref<MLPPMatrix> word_embeddings;
		Vector<String> word_list;
	};

	enum WordToVecType {
		WORD_TO_VEC_TYPE_CBOW = 0,
		WORD_TO_VEC_TYPE_SKIPGRAM,
	};

	WordsToVecResult word_to_vec(Vector<String> sentences, WordToVecType type, int windowSize, int dimension, real_t learning_rate, int max_epoch);

	Ref<MLPPMatrix> lsa(Vector<String> sentences, int dim);

	Vector<String> create_word_list(Vector<String> sentences);

	// Extra
	void setInputNames(std::string fileName, std::vector<std::string> &inputNames);
	Ref<MLPPMatrix> feature_scaling(const Ref<MLPPMatrix> &X);
	Ref<MLPPMatrix> mean_centering(const Ref<MLPPMatrix> &X);
	Ref<MLPPMatrix> mean_normalization(const Ref<MLPPMatrix> &X);
	Ref<MLPPMatrix> one_hot_rep(const Ref<MLPPVector> &temp_output_set, int n_class);
	std::vector<real_t> reverseOneHot(std::vector<std::vector<real_t>> tempOutputSet);

	template <class T>
	std::vector<T> vecToSet(std::vector<T> inputSet) {
		std::vector<T> setInputSet;
		for (uint32_t i = 0; i < inputSet.size(); i++) {
			bool new_element = true;
			for (uint32_t j = 0; j < setInputSet.size(); j++) {
				if (setInputSet[j] == inputSet[i]) {
					new_element = false;
				}
			}
			if (new_element) {
				setInputSet.push_back(inputSet[i]);
			}
		}
		return setInputSet;
	}

	template <class T>
	Vector<T> vec_to_set(Vector<T> input_set) {
		Vector<T> set_input_set;

		for (int i = 0; i < input_set.size(); i++) {
			bool new_element = true;

			for (int j = 0; j < set_input_set.size(); j++) {
				if (set_input_set[j] == input_set[i]) {
					new_element = false;
				}
			}

			if (new_element) {
				set_input_set.push_back(input_set[i]);
			}
		}

		return set_input_set;
	}

	Ref<MLPPVector> vec_to_setnv(const Ref<MLPPVector> &input_set) {
		Vector<real_t> set_input_set;

		for (int i = 0; i < input_set->size(); i++) {
			bool new_element = true;

			for (int j = 0; j < set_input_set.size(); j++) {
				if (set_input_set[j] == input_set->element_get(i)) {
					new_element = false;
				}
			}

			if (new_element) {
				set_input_set.push_back(input_set->element_get(i));
			}
		}

		Ref<MLPPVector> ret;
		ret.instance();
		ret->set_from_vector(set_input_set);

		return ret;
	}

	void load_default_suffixes();
	void load_default_stop_words();

	Vector<String> suffixes;
	Vector<String> stop_words;

protected:
	static void _bind_methods();
};

#endif /* Data_hpp */
