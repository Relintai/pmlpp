
#ifndef MLPP_DATA_OLD_H
#define MLPP_DATA_OLD_H

//
//  Data.hpp
//  MLP
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/math/math_defs.h"

#include <string>
#include <tuple>
#include <vector>

class MLPPDataOld {
public:
	// Load Datasets
	std::tuple<std::vector<std::vector<real_t>>, std::vector<real_t>> loadBreastCancer();
	std::tuple<std::vector<std::vector<real_t>>, std::vector<real_t>> loadBreastCancerSVC();
	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> loadIris();
	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> loadWine();
	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> loadMnistTrain();
	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> loadMnistTest();
	std::tuple<std::vector<std::vector<real_t>>, std::vector<real_t>> loadCaliforniaHousing();
	std::tuple<std::vector<real_t>, std::vector<real_t>> loadFiresAndCrime();

	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> trainTestSplit(std::vector<std::vector<real_t>> inputSet, std::vector<std::vector<real_t>> outputSet, real_t testSize);

	// Supervised
	void setData(int k, std::string fileName, std::vector<std::vector<real_t>> &inputSet, std::vector<real_t> &outputSet);
	void printData(std::vector<std::string> inputName, std::string outputName, std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet);

	// Unsupervised
	void setData(int k, std::string fileName, std::vector<std::vector<real_t>> &inputSet);
	void printData(std::vector<std::string> inputName, std::vector<std::vector<real_t>> inputSet);

	// Simple
	void setData(std::string fileName, std::vector<real_t> &inputSet, std::vector<real_t> &outputSet);
	void printData(std::string &inputName, std::string &outputName, std::vector<real_t> &inputSet, std::vector<real_t> &outputSet);

	// Images
	std::vector<std::vector<real_t>> rgb2gray(std::vector<std::vector<std::vector<real_t>>> input);
	std::vector<std::vector<std::vector<real_t>>> rgb2ycbcr(std::vector<std::vector<std::vector<real_t>>> input);
	std::vector<std::vector<std::vector<real_t>>> rgb2hsv(std::vector<std::vector<std::vector<real_t>>> input);
	std::vector<std::vector<std::vector<real_t>>> rgb2xyz(std::vector<std::vector<std::vector<real_t>>> input);
	std::vector<std::vector<std::vector<real_t>>> xyz2rgb(std::vector<std::vector<std::vector<real_t>>> input);

	// Text-Based & NLP
	std::string toLower(std::string text);
	std::vector<char> split(std::string text);
	std::vector<std::string> splitSentences(std::string data);
	std::vector<std::string> removeSpaces(std::vector<std::string> data);
	std::vector<std::string> removeNullByte(std::vector<std::string> data);
	std::vector<std::string> segment(std::string text);
	std::vector<real_t> tokenize(std::string text);
	std::vector<std::string> removeStopWords(std::string text);
	std::vector<std::string> removeStopWords(std::vector<std::string> segmented_data);

	std::string stemming(std::string text);

	std::vector<std::vector<real_t>> BOW(std::vector<std::string> sentences, std::string = "Default");
	std::vector<std::vector<real_t>> TFIDF(std::vector<std::string> sentences);

	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::string>> word2Vec(std::vector<std::string> sentences, std::string type, int windowSize, int dimension, real_t learning_rate, int max_epoch);

	struct WordsToVecResult {
		std::vector<std::vector<real_t>> word_embeddings;
		std::vector<std::string> word_list;
	};

	WordsToVecResult word_to_vec(std::vector<std::string> sentences, std::string type, int windowSize, int dimension, real_t learning_rate, int max_epoch);

	std::vector<std::vector<real_t>> LSA(std::vector<std::string> sentences, int dim);

	std::vector<std::string> createWordList(std::vector<std::string> sentences);

	// Extra
	void setInputNames(std::string fileName, std::vector<std::string> &inputNames);
	std::vector<std::vector<real_t>> featureScaling(std::vector<std::vector<real_t>> X);
	std::vector<std::vector<real_t>> meanNormalization(std::vector<std::vector<real_t>> X);
	std::vector<std::vector<real_t>> meanCentering(std::vector<std::vector<real_t>> X);
	std::vector<std::vector<real_t>> oneHotRep(std::vector<real_t> tempOutputSet, int n_class);
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

protected:
	static void _bind_methods();
};

#endif /* Data_hpp */
