#pragma once
#ifndef NEURON_H
#define NEURON_H
#include <vector>
using namespace std;
class Neuron {
private:
	vector<float> weight;
	vector<vector<float>> Batchweight;
	float bias;
	vector<float> Batchbias;
	float theta;
	float z;
	float a;
public:
	~Neuron();
	friend class Model;
	void Initialize(int num);
	void Active(vector<float> input);
	float Relu(float z);
	float Sigmoid(float z);
	int Relu_derivative(float z);
};
#endif // !NEURON_H
