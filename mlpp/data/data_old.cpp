//
//  Data.cpp
//  MLP
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "data_old.h"

#include "core/os/file_access.h"

#include "../lin_alg/lin_alg_old.h"
#include "../softmax_net/softmax_net_old.h"
#include "../stat/stat_old.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

// Loading Datasets
std::tuple<std::vector<std::vector<real_t>>, std::vector<real_t>> MLPPDataOld::loadBreastCancer() {
	const int BREAST_CANCER_SIZE = 30; // k = 30
	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;

	setData(BREAST_CANCER_SIZE, "MLPP/Data/Datasets/BreastCancer.csv", inputSet, outputSet);
	return { inputSet, outputSet };
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<real_t>> MLPPDataOld::loadBreastCancerSVC() {
	const int BREAST_CANCER_SIZE = 30; // k = 30
	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;

	setData(BREAST_CANCER_SIZE, "MLPP/Data/Datasets/BreastCancerSVM.csv", inputSet, outputSet);
	return { inputSet, outputSet };
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPDataOld::loadIris() {
	const int IRIS_SIZE = 4;
	const int ONE_HOT_NUM = 3;
	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> tempOutputSet;

	setData(IRIS_SIZE, "/Users/marcmelikyan/Desktop/Data/Iris.csv", inputSet, tempOutputSet);
	std::vector<std::vector<real_t>> outputSet = oneHotRep(tempOutputSet, ONE_HOT_NUM);
	return { inputSet, outputSet };
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPDataOld::loadWine() {
	const int WINE_SIZE = 4;
	const int ONE_HOT_NUM = 3;
	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> tempOutputSet;

	setData(WINE_SIZE, "MLPP/Data/Datasets/Iris.csv", inputSet, tempOutputSet);
	std::vector<std::vector<real_t>> outputSet = oneHotRep(tempOutputSet, ONE_HOT_NUM);
	return { inputSet, outputSet };
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPDataOld::loadMnistTrain() {
	const int MNIST_SIZE = 784;
	const int ONE_HOT_NUM = 10;
	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> tempOutputSet;

	setData(MNIST_SIZE, "MLPP/Data/Datasets/MnistTrain.csv", inputSet, tempOutputSet);
	std::vector<std::vector<real_t>> outputSet = oneHotRep(tempOutputSet, ONE_HOT_NUM);
	return { inputSet, outputSet };
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPDataOld::loadMnistTest() {
	const int MNIST_SIZE = 784;
	const int ONE_HOT_NUM = 10;
	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> tempOutputSet;

	setData(MNIST_SIZE, "MLPP/Data/Datasets/MnistTest.csv", inputSet, tempOutputSet);
	std::vector<std::vector<real_t>> outputSet = oneHotRep(tempOutputSet, ONE_HOT_NUM);
	return { inputSet, outputSet };
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<real_t>> MLPPDataOld::loadCaliforniaHousing() {
	const int CALIFORNIA_HOUSING_SIZE = 13; // k = 30
	std::vector<std::vector<real_t>> inputSet;
	std::vector<real_t> outputSet;

	setData(CALIFORNIA_HOUSING_SIZE, "MLPP/Data/Datasets/CaliforniaHousing.csv", inputSet, outputSet);
	return { inputSet, outputSet };
}

std::tuple<std::vector<real_t>, std::vector<real_t>> MLPPDataOld::loadFiresAndCrime() {
	std::vector<real_t> inputSet; // k is implicitly 1.
	std::vector<real_t> outputSet;

	setData("MLPP/Data/Datasets/FiresAndCrime.csv", inputSet, outputSet);
	return { inputSet, outputSet };
}

// Note that inputs and outputs should be pairs (technically), but this
// implementation will separate them. (My implementation keeps them tied together.)
// Not yet sure whether this is intentional or not (or it's something like a compiler specific difference)
std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> MLPPDataOld::trainTestSplit(std::vector<std::vector<real_t>> inputSet, std::vector<std::vector<real_t>> outputSet, real_t testSize) {
	std::random_device rd;
	std::default_random_engine generator(rd());

	std::shuffle(inputSet.begin(), inputSet.end(), generator); // inputSet random shuffle
	std::shuffle(outputSet.begin(), outputSet.end(), generator); // outputSet random shuffle)

	std::vector<std::vector<real_t>> inputTestSet;
	std::vector<std::vector<real_t>> outputTestSet;

	int testInputNumber = testSize * inputSet.size(); // implicit usage of floor
	int testOutputNumber = testSize * outputSet.size(); // implicit usage of floor

	for (int i = 0; i < testInputNumber; i++) {
		inputTestSet.push_back(inputSet[i]);
		inputSet.erase(inputSet.begin());
	}

	for (int i = 0; i < testOutputNumber; i++) {
		outputTestSet.push_back(outputSet[i]);
		outputSet.erase(outputSet.begin());
	}

	return { inputSet, outputSet, inputTestSet, outputTestSet };
}

// MULTIVARIATE SUPERVISED

void MLPPDataOld::setData(int k, std::string fileName, std::vector<std::vector<real_t>> &inputSet, std::vector<real_t> &outputSet) {
	MLPPLinAlgOld alg;
	std::string inputTemp;
	std::string outputTemp;

	inputSet.resize(k);

	std::ifstream dataFile(fileName);
	if (!dataFile.is_open()) {
		std::cout << fileName << " failed to open." << std::endl;
	}

	std::string line;
	while (std::getline(dataFile, line)) {
		std::stringstream ss(line);

		for (int i = 0; i < k; i++) {
			std::getline(ss, inputTemp, ',');
			inputSet[i].push_back(std::stod(inputTemp));
		}

		std::getline(ss, outputTemp, ',');
		outputSet.push_back(std::stod(outputTemp));
	}
	inputSet = alg.transpose(inputSet);
	dataFile.close();
}

void MLPPDataOld::printData(std::vector<std::string> inputName, std::string outputName, std::vector<std::vector<real_t>> inputSet, std::vector<real_t> outputSet) {
	MLPPLinAlgOld alg;
	inputSet = alg.transpose(inputSet);
	for (uint32_t i = 0; i < inputSet.size(); i++) {
		std::cout << inputName[i] << std::endl;
		for (uint32_t j = 0; j < inputSet[i].size(); j++) {
			std::cout << inputSet[i][j] << std::endl;
		}
	}

	std::cout << outputName << std::endl;
	for (uint32_t i = 0; i < outputSet.size(); i++) {
		std::cout << outputSet[i] << std::endl;
	}
}

// UNSUPERVISED

void MLPPDataOld::setData(int k, std::string fileName, std::vector<std::vector<real_t>> &inputSet) {
	MLPPLinAlgOld alg;
	std::string inputTemp;

	inputSet.resize(k);

	std::ifstream dataFile(fileName);
	if (!dataFile.is_open()) {
		std::cout << fileName << " failed to open." << std::endl;
	}

	std::string line;
	while (std::getline(dataFile, line)) {
		std::stringstream ss(line);

		for (int i = 0; i < k; i++) {
			std::getline(ss, inputTemp, ',');
			inputSet[i].push_back(std::stod(inputTemp));
		}
	}
	inputSet = alg.transpose(inputSet);
	dataFile.close();
}

void MLPPDataOld::printData(std::vector<std::string> inputName, std::vector<std::vector<real_t>> inputSet) {
	MLPPLinAlgOld alg;
	inputSet = alg.transpose(inputSet);
	for (uint32_t i = 0; i < inputSet.size(); i++) {
		std::cout << inputName[i] << std::endl;
		for (uint32_t j = 0; j < inputSet[i].size(); j++) {
			std::cout << inputSet[i][j] << std::endl;
		}
	}
}

// SIMPLE

void MLPPDataOld::setData(std::string fileName, std::vector<real_t> &inputSet, std::vector<real_t> &outputSet) {
	std::string inputTemp, outputTemp;

	std::ifstream dataFile(fileName);
	if (!dataFile.is_open()) {
		std::cout << "The file failed to open." << std::endl;
	}

	std::string line;

	while (std::getline(dataFile, line)) {
		std::stringstream ss(line);

		std::getline(ss, inputTemp, ',');
		std::getline(ss, outputTemp, ',');

		inputSet.push_back(std::stod(inputTemp));
		outputSet.push_back(std::stod(outputTemp));
	}

	dataFile.close();
}

void MLPPDataOld::printData(std::string &inputName, std::string &outputName, std::vector<real_t> &inputSet, std::vector<real_t> &outputSet) {
	std::cout << inputName << std::endl;
	for (uint32_t i = 0; i < inputSet.size(); i++) {
		std::cout << inputSet[i] << std::endl;
	}

	std::cout << outputName << std::endl;
	for (uint32_t i = 0; i < inputSet.size(); i++) {
		std::cout << outputSet[i] << std::endl;
	}
}

// Images
std::vector<std::vector<real_t>> MLPPDataOld::rgb2gray(std::vector<std::vector<std::vector<real_t>>> input) {
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
}

std::vector<std::vector<std::vector<real_t>>> MLPPDataOld::rgb2ycbcr(std::vector<std::vector<std::vector<real_t>>> input) {
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
}

// Conversion formulas available here:
// https://www.rapidtables.com/convert/color/rgb-to-hsv.html
std::vector<std::vector<std::vector<real_t>>> MLPPDataOld::rgb2hsv(std::vector<std::vector<std::vector<real_t>>> input) {
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
}

// http://machinethatsees.blogspot.com/2013/07/how-to-convert-rgb-to-xyz-or-vice-versa.html
std::vector<std::vector<std::vector<real_t>>> MLPPDataOld::rgb2xyz(std::vector<std::vector<std::vector<real_t>>> input) {
	MLPPLinAlgOld alg;
	std::vector<std::vector<std::vector<real_t>>> XYZ;
	XYZ = alg.resize(XYZ, input);
	std::vector<std::vector<real_t>> RGB2XYZ = { { 0.4124564, 0.3575761, 0.1804375 }, { 0.2126726, 0.7151522, 0.0721750 }, { 0.0193339, 0.1191920, 0.9503041 } };
	return alg.vector_wise_tensor_product(input, RGB2XYZ);
}

std::vector<std::vector<std::vector<real_t>>> MLPPDataOld::xyz2rgb(std::vector<std::vector<std::vector<real_t>>> input) {
	MLPPLinAlgOld alg;
	std::vector<std::vector<std::vector<real_t>>> XYZ;
	XYZ = alg.resize(XYZ, input);
	std::vector<std::vector<real_t>> RGB2XYZ = alg.inverse({ { 0.4124564, 0.3575761, 0.1804375 }, { 0.2126726, 0.7151522, 0.0721750 }, { 0.0193339, 0.1191920, 0.9503041 } });
	return alg.vector_wise_tensor_product(input, RGB2XYZ);
}

// TEXT-BASED & NLP
std::string MLPPDataOld::toLower(std::string text) {
	for (uint32_t i = 0; i < text.size(); i++) {
		text[i] = tolower(text[i]);
	}
	return text;
}

std::vector<char> MLPPDataOld::split(std::string text) {
	std::vector<char> split_data;
	for (uint32_t i = 0; i < text.size(); i++) {
		split_data.push_back(text[i]);
	}
	return split_data;
}

std::vector<std::string> MLPPDataOld::splitSentences(std::string data) {
	std::vector<std::string> sentences;
	std::string currentStr = "";

	for (uint32_t i = 0; i < data.length(); i++) {
		currentStr.push_back(data[i]);
		if (data[i] == '.' && data[i + 1] != '.') {
			sentences.push_back(currentStr);
			currentStr = "";
			i++;
		}
	}
	return sentences;
}

std::vector<std::string> MLPPDataOld::removeSpaces(std::vector<std::string> data) {
	for (uint32_t i = 0; i < data.size(); i++) {
		auto it = data[i].begin();
		for (uint32_t j = 0; j < data[i].length(); j++) {
			if (data[i][j] == ' ') {
				data[i].erase(it);
			}
			it++;
		}
	}
	return data;
}

std::vector<std::string> MLPPDataOld::removeNullByte(std::vector<std::string> data) {
	for (uint32_t i = 0; i < data.size(); i++) {
		if (data[i] == "\0") {
			data.erase(data.begin() + i);
		}
	}
	return data;
}

std::vector<std::string> MLPPDataOld::segment(std::string text) {
	std::vector<std::string> segmented_data;
	int prev_delim = 0;
	for (uint32_t i = 0; i < text.length(); i++) {
		if (text[i] == ' ') {
			segmented_data.push_back(text.substr(prev_delim, i - prev_delim));
			prev_delim = i + 1;
		} else if (text[i] == ',' || text[i] == '!' || text[i] == '.' || text[i] == '-') {
			segmented_data.push_back(text.substr(prev_delim, i - prev_delim));
			std::string punc;
			punc.push_back(text[i]);
			segmented_data.push_back(punc);
			prev_delim = i + 2;
			i++;
		} else if (i == text.length() - 1) {
			segmented_data.push_back(text.substr(prev_delim, text.length() - prev_delim)); // hehe oops- forgot this
		}
	}

	return segmented_data;
}

std::vector<real_t> MLPPDataOld::tokenize(std::string text) {
	int max_num = 0;
	bool new_num = true;
	std::vector<std::string> segmented_data = segment(text);
	std::vector<real_t> tokenized_data;
	tokenized_data.resize(segmented_data.size());
	for (uint32_t i = 0; i < segmented_data.size(); i++) {
		for (int j = i - 1; j >= 0; j--) {
			if (segmented_data[i] == segmented_data[j]) {
				tokenized_data[i] = tokenized_data[j];
				new_num = false;
			}
		}
		if (!new_num) {
			new_num = true;
		} else {
			max_num++;
			tokenized_data[i] = max_num;
		}
	}
	return tokenized_data;
}

std::vector<std::string> MLPPDataOld::removeStopWords(std::string text) {
	std::vector<std::string> stopWords = { "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now" };
	std::vector<std::string> segmented_data = removeSpaces(segment(toLower(text)));

	for (uint32_t i = 0; i < stopWords.size(); i++) {
		for (uint32_t j = 0; j < segmented_data.size(); j++) {
			if (segmented_data[j] == stopWords[i]) {
				segmented_data.erase(segmented_data.begin() + j);
			}
		}
	}
	return segmented_data;
}

std::vector<std::string> MLPPDataOld::removeStopWords(std::vector<std::string> segmented_data) {
	std::vector<std::string> stopWords = { "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now" };
	for (uint32_t i = 0; i < segmented_data.size(); i++) {
		for (uint32_t j = 0; j < stopWords.size(); j++) {
			if (segmented_data[i] == stopWords[j]) {
				segmented_data.erase(segmented_data.begin() + i);
			}
		}
	}
	return segmented_data;
}

std::string MLPPDataOld::stemming(std::string text) {
	// Our list of suffixes which we use to compare against
	std::vector<std::string> suffixes = { "eer", "er", "ion", "ity", "ment", "ness", "or", "sion", "ship", "th", "able", "ible", "al", "ant", "ary", "ful", "ic", "ious", "ous", "ive", "less", "y", "ed", "en", "ing", "ize", "ise", "ly", "ward", "wise" };
	int padding_size = 4;
	char padding = ' '; // our padding

	for (int i = 0; i < padding_size; i++) {
		text[text.length() + i] = padding; // ' ' will be our padding value
	}

	for (uint32_t i = 0; i < text.size(); i++) {
		for (uint32_t j = 0; j < suffixes.size(); j++) {
			if (text.substr(i, suffixes[j].length()) == suffixes[j] && (text[i + suffixes[j].length()] == ' ' || text[i + suffixes[j].length()] == ',' || text[i + suffixes[j].length()] == '-' || text[i + suffixes[j].length()] == '.' || text[i + suffixes[j].length()] == '!')) {
				text.erase(i, suffixes[j].length());
			}
		}
	}

	return text;
}

std::vector<std::vector<real_t>> MLPPDataOld::BOW(std::vector<std::string> sentences, std::string type) {
	/*
	STEPS OF BOW:
		1) To lowercase (done by removeStopWords function by def)
		2) Removing stop words
		3) Obtain a list of the used words
		4) Create a one hot encoded vector of the words and sentences
		5) Sentence.size() x list.size() matrix
	*/

	std::vector<std::string> wordList = removeNullByte(removeStopWords(createWordList(sentences)));

	std::vector<std::vector<std::string>> segmented_sentences;
	segmented_sentences.resize(sentences.size());

	for (uint32_t i = 0; i < sentences.size(); i++) {
		segmented_sentences[i] = removeStopWords(sentences[i]);
	}

	std::vector<std::vector<real_t>> bow;

	bow.resize(sentences.size());
	for (uint32_t i = 0; i < bow.size(); i++) {
		bow[i].resize(wordList.size());
	}

	for (uint32_t i = 0; i < segmented_sentences.size(); i++) {
		for (uint32_t j = 0; j < segmented_sentences[i].size(); j++) {
			for (uint32_t k = 0; k < wordList.size(); k++) {
				if (segmented_sentences[i][j] == wordList[k]) {
					if (type == "Binary") {
						bow[i][k] = 1;
					} else {
						bow[i][k]++;
					}
				}
			}
		}
	}
	return bow;
}

std::vector<std::vector<real_t>> MLPPDataOld::TFIDF(std::vector<std::string> sentences) {
	MLPPLinAlgOld alg;
	std::vector<std::string> wordList = removeNullByte(removeStopWords(createWordList(sentences)));

	std::vector<std::vector<std::string>> segmented_sentences;
	segmented_sentences.resize(sentences.size());

	for (uint32_t i = 0; i < sentences.size(); i++) {
		segmented_sentences[i] = removeStopWords(sentences[i]);
	}

	std::vector<std::vector<real_t>> TF;
	std::vector<int> frequency;
	frequency.resize(wordList.size());
	TF.resize(segmented_sentences.size());
	for (uint32_t i = 0; i < TF.size(); i++) {
		TF[i].resize(wordList.size());
	}
	for (uint32_t i = 0; i < segmented_sentences.size(); i++) {
		std::vector<bool> present(wordList.size(), false);
		for (uint32_t j = 0; j < segmented_sentences[i].size(); j++) {
			for (uint32_t k = 0; k < wordList.size(); k++) {
				if (segmented_sentences[i][j] == wordList[k]) {
					TF[i][k]++;
					if (!present[k]) {
						frequency[k]++;
						present[k] = true;
					}
				}
			}
		}
		TF[i] = alg.scalarMultiply(real_t(1) / real_t(segmented_sentences[i].size()), TF[i]);
	}

	std::vector<real_t> IDF;
	IDF.resize(frequency.size());

	for (uint32_t i = 0; i < IDF.size(); i++) {
		IDF[i] = std::log((real_t)segmented_sentences.size() / (real_t)frequency[i]);
	}

	std::vector<std::vector<real_t>> TFIDF;
	TFIDF.resize(segmented_sentences.size());
	for (uint32_t i = 0; i < TFIDF.size(); i++) {
		TFIDF[i].resize(wordList.size());
	}

	for (uint32_t i = 0; i < TFIDF.size(); i++) {
		for (uint32_t j = 0; j < TFIDF[i].size(); j++) {
			TFIDF[i][j] = TF[i][j] * IDF[j];
		}
	}

	return TFIDF;
}

std::tuple<std::vector<std::vector<real_t>>, std::vector<std::string>> MLPPDataOld::word2Vec(std::vector<std::string> sentences, std::string type, int windowSize, int dimension, real_t learning_rate, int max_epoch) {
	std::vector<std::string> wordList = removeNullByte(removeStopWords(createWordList(sentences)));

	std::vector<std::vector<std::string>> segmented_sentences;
	segmented_sentences.resize(sentences.size());

	for (uint32_t i = 0; i < sentences.size(); i++) {
		segmented_sentences[i] = removeStopWords(sentences[i]);
	}

	std::vector<std::string> inputStrings;
	std::vector<std::string> outputStrings;

	for (uint32_t i = 0; i < segmented_sentences.size(); i++) {
		for (uint32_t j = 0; j < segmented_sentences[i].size(); j++) {
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

	uint32_t inputSize = inputStrings.size();

	inputStrings.insert(inputStrings.end(), outputStrings.begin(), outputStrings.end());

	std::vector<std::vector<real_t>> BOW = MLPPDataOld::BOW(inputStrings, "Binary");

	std::vector<std::vector<real_t>> inputSet;
	std::vector<std::vector<real_t>> outputSet;

	for (uint32_t i = 0; i < inputSize; i++) {
		inputSet.push_back(BOW[i]);
	}

	for (uint32_t i = inputSize; i < BOW.size(); i++) {
		outputSet.push_back(BOW[i]);
	}

	MLPPSoftmaxNetOld *model;

	if (type == "Skipgram") {
		model = new MLPPSoftmaxNetOld(outputSet, inputSet, dimension);
	} else { // else = CBOW. We maintain it is a default.
		model = new MLPPSoftmaxNetOld(inputSet, outputSet, dimension);
	}

	model->gradientDescent(learning_rate, max_epoch, false);

	std::vector<std::vector<real_t>> wordEmbeddings = model->getEmbeddings();
	delete model;
	return { wordEmbeddings, wordList };
}

struct WordsToVecResult {
	std::vector<std::vector<real_t>> word_embeddings;
	std::vector<std::string> word_list;
};

MLPPDataOld::WordsToVecResult MLPPDataOld::word_to_vec(std::vector<std::string> sentences, std::string type, int windowSize, int dimension, real_t learning_rate, int max_epoch) {
	WordsToVecResult res;

	res.word_list = removeNullByte(removeStopWords(createWordList(sentences)));

	std::vector<std::vector<std::string>> segmented_sentences;
	segmented_sentences.resize(sentences.size());

	for (uint32_t i = 0; i < sentences.size(); i++) {
		segmented_sentences[i] = removeStopWords(sentences[i]);
	}

	std::vector<std::string> inputStrings;
	std::vector<std::string> outputStrings;

	for (uint32_t i = 0; i < segmented_sentences.size(); i++) {
		for (uint32_t j = 0; j < segmented_sentences[i].size(); j++) {
			for (int k = windowSize; k > 0; k--) {
				if (j - k >= 0) {
					inputStrings.push_back(segmented_sentences[i][j]);

					outputStrings.push_back(segmented_sentences[i][j - k]);
				}
				if (j + k <= segmented_sentences[i].size() - 1) {
					inputStrings.push_back(segmented_sentences[i][j]);
					outputStrings.push_back(segmented_sentences[i][j + k]);
				}
			}
		}
	}

	uint32_t inputSize = inputStrings.size();

	inputStrings.insert(inputStrings.end(), outputStrings.begin(), outputStrings.end());

	std::vector<std::vector<real_t>> BOW = MLPPDataOld::BOW(inputStrings, "Binary");

	std::vector<std::vector<real_t>> inputSet;
	std::vector<std::vector<real_t>> outputSet;

	for (uint32_t i = 0; i < inputSize; i++) {
		inputSet.push_back(BOW[i]);
	}

	for (uint32_t i = inputSize; i < BOW.size(); i++) {
		outputSet.push_back(BOW[i]);
	}

	MLPPSoftmaxNetOld *model;

	if (type == "Skipgram") {
		model = new MLPPSoftmaxNetOld(outputSet, inputSet, dimension);
	} else { // else = CBOW. We maintain it is a default.
		model = new MLPPSoftmaxNetOld(inputSet, outputSet, dimension);
	}

	model->gradientDescent(learning_rate, max_epoch, false);

	res.word_embeddings = model->getEmbeddings();
	delete model;

	return res;
}

std::vector<std::vector<real_t>> MLPPDataOld::LSA(std::vector<std::string> sentences, int dim) {
	MLPPLinAlgOld alg;
	std::vector<std::vector<real_t>> docWordData = BOW(sentences, "Binary");

	MLPPLinAlgOld::SVDResultOld svr_res = alg.SVD(docWordData);
	std::vector<std::vector<real_t>> S_trunc = alg.zeromat(dim, dim);
	std::vector<std::vector<real_t>> Vt_trunc;
	for (int i = 0; i < dim; i++) {
		S_trunc[i][i] = svr_res.S[i][i];
		Vt_trunc.push_back(svr_res.Vt[i]);
	}

	std::vector<std::vector<real_t>> embeddings = alg.matmult(S_trunc, Vt_trunc);
	return embeddings;
}

std::vector<std::string> MLPPDataOld::createWordList(std::vector<std::string> sentences) {
	std::string combinedText = "";
	for (uint32_t i = 0; i < sentences.size(); i++) {
		if (i != 0) {
			combinedText += " ";
		}
		combinedText += sentences[i];
	}

	return removeSpaces(vecToSet(removeStopWords(combinedText)));
}

// EXTRA
void MLPPDataOld::setInputNames(std::string fileName, std::vector<std::string> &inputNames) {
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

std::vector<std::vector<real_t>> MLPPDataOld::featureScaling(std::vector<std::vector<real_t>> X) {
	MLPPLinAlgOld alg;
	X = alg.transpose(X);
	std::vector<real_t> max_elements, min_elements;
	max_elements.resize(X.size());
	min_elements.resize(X.size());

	for (uint32_t i = 0; i < X.size(); i++) {
		max_elements[i] = alg.max(X[i]);
		min_elements[i] = alg.min(X[i]);
	}

	for (uint32_t i = 0; i < X.size(); i++) {
		for (uint32_t j = 0; j < X[i].size(); j++) {
			X[i][j] = (X[i][j] - min_elements[i]) / (max_elements[i] - min_elements[i]);
		}
	}
	return alg.transpose(X);
}

std::vector<std::vector<real_t>> MLPPDataOld::meanNormalization(std::vector<std::vector<real_t>> X) {
	MLPPLinAlgOld alg;
	MLPPStatOld stat;
	// (X_j - mu_j) / std_j, for every j

	X = meanCentering(X);
	for (uint32_t i = 0; i < X.size(); i++) {
		X[i] = alg.scalarMultiply(1 / stat.standardDeviation(X[i]), X[i]);
	}
	return X;
}

std::vector<std::vector<real_t>> MLPPDataOld::meanCentering(std::vector<std::vector<real_t>> X) {
	MLPPStatOld stat;
	for (uint32_t i = 0; i < X.size(); i++) {
		real_t mean_i = stat.mean(X[i]);
		for (uint32_t j = 0; j < X[i].size(); j++) {
			X[i][j] -= mean_i;
		}
	}
	return X;
}

std::vector<std::vector<real_t>> MLPPDataOld::oneHotRep(std::vector<real_t> tempOutputSet, int n_class) {
	std::vector<std::vector<real_t>> outputSet;
	outputSet.resize(tempOutputSet.size());
	for (uint32_t i = 0; i < tempOutputSet.size(); i++) {
		for (int j = 0; j <= n_class - 1; j++) {
			if (tempOutputSet[i] == j) {
				outputSet[i].push_back(1);
			} else {
				outputSet[i].push_back(0);
			}
		}
	}
	return outputSet;
}

std::vector<real_t> MLPPDataOld::reverseOneHot(std::vector<std::vector<real_t>> tempOutputSet) {
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
