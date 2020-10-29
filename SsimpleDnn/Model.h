#pragma once
#ifndef MODEL_H
#define MODEL_H
#include "Neuron.h"
#include <opencv2/core/core.hpp>
#include <vector>
#include <set>
using namespace std;
using namespace cv;
class Model {
public:
	vector<vector<Neuron>> NN;
	vector<float> loss;
	unsigned batch_size;
	unsigned epoch;
	double learning_rate;
	unsigned output_num;
	float loss_epoch;
	float accuracy;
	Size input_size;
	int input_num;
	set<int> Set_Label;
	vector<Mat> vecImages;
	vector<int> vecLabels;

	Model(double learning_rate, unsigned batch_size,unsigned epoch,unsigned output_num, Size input_size);

	friend class Neuron;
	vector<float> Get_active(int Layer_num);
	vector<float> Feed_forward(vector<float> input);
	void Update_paramter();
	void Back_propagation(vector<float> label);
	void Train_data(const string& Image_path);
	void Test_data(const string& Image_path);
	void Evalution_model(int epoch_num);
	void Add_layer(int num);
	bool Judge_result(vector<float> input, signed res);
	void Shuffle_data(vector<Mat> &vecImage,vector<int> &vecLabel);
	int Get_Label(string Imagename);
};
#endif // !MODEL_H
