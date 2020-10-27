#include "Neuron.h"
#include<cstdlib>
#include<math.h>
#include<iostream>
#include<vector>
#define N 9999;
using namespace std;
Neuron::~Neuron()
{

}

void Neuron::Initialize(int num) 
{
	for (unsigned i = 0; i < num; i++) 
	{
		float temp = (2.0 * rand() / RAND_MAX - 1.0);
		weight.push_back(temp);
	}
	bias = rand()/(float)(10000);
	//bias = 0;
	Batchweight.resize(num);
	/*for (const auto it : weight)
		cout << it << " ";
	cout << endl;
	cout << bias << endl;*/
}

void Neuron::Active(vector<float> input) 
{
	z = 0;
	for (unsigned i = 0; i < input.size(); i++)
		z += (weight[i] * input[i]);
	z += bias;
	a = Sigmoid(z);
}

float Neuron::Relu(float z) 
{
	return z > 0 ? z : 0;
}

float Neuron::Sigmoid(float z) 
{
	return 1.0/(1.0 + exp(-z));
}

int Neuron::Relu_derivative(float z)
{
	return z > 0 ? 1 : 0;
}