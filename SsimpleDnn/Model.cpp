#include "Model.h"
#include "Neuron.h"
#include <set>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;

Model::Model(double learning_rate,unsigned batch_size, unsigned epoch,unsigned output_num,Size input_size)
{
	int input_num = input_size.height*input_size.width;
	vector<Neuron> layer(input_num);
	NN.push_back(layer);
	this->learning_rate = learning_rate;
	this->batch_size = batch_size;
	this->epoch = epoch;
	this->output_num = output_num;
	this->input_size.height = input_size.height;
	this->input_size.width = input_size.width;
}

void Model::Add_layer(int Layer_size) 
{
	int size = NN.back().size();
	vector<Neuron> layer;
	for (unsigned i = 0; i < Layer_size; i++) 
	{
		Neuron n;
		n.Initialize(size);
		layer.push_back(n);
	}
	NN.push_back(layer);
}

vector<float> Model::Get_active(int Layer_num) 
{
	vector<float> res;
	for (int i = 0; i < NN[Layer_num].size(); i++)
		res.push_back(NN[Layer_num][i].a);
	return res;
}

vector<float> Model::Feed_forward(vector<float> input) 
{
	//将数据输入到一层
	for (unsigned i = 0; i < input.size(); i++) 
		NN[0][i].a = input[i]/255.0;
	//前向传播
	for (unsigned i = 1; i < NN.size()-1; i++) 
		for (unsigned j = 0; j < NN[i].size(); j++)
			NN[i][j].Active(Get_active(i - 1));
	float sum = 0;
	vector<float> last = Get_active(NN.size() - 2);
	for (unsigned j = 0; j < NN.back().size(); j++) 
	{
		NN.back()[j].z = 0;
		for (unsigned i = 0; i < last.size(); i++) 
		{
			NN.back()[j].z += (NN.back()[j].weight[i] * last[i]);
		}
		NN.back()[j].z += NN.back()[j].bias;
		sum += exp(NN.back()[j].z);
	}
	for (unsigned j = 0; j < NN.back().size(); j++)
	{
		NN.back()[j].a = (exp(NN.back()[j].z) / sum);
	}
	//取出前向传播结果
	vector<float> answer;
	for (const auto it : NN.back())
		answer.push_back(it.a);
	return answer;
}

void Model::Back_propagation(vector<float> label) 
{
	//求取输出层的theta
	for (unsigned i = 0; i < NN.back().size(); i++) 
	{
		if(label[i]==1.0)
			NN.back()[i].theta = (NN.back()[i].a-1);
		else
			NN.back()[i].theta = NN.back()[i].a;
		//NN.back()[i].theta = -(label[i] - NN.back()[i].a)*(1 - NN.back()[i].a)*NN.back()[i].a;
	}

	//求取各层的theta
	for (unsigned i = NN.size() - 2; i > 0; i--) 
	{
		for (unsigned j = 0; j < NN[i].size(); j++) 
		{
			NN[i][j].theta = 0;
			for (unsigned k = 0; k < NN[i + 1].size(); k++) 
			{
				//NN[i][j].theta += NN[i + 1][k].theta * NN[i + 1][k].weight[j] * NN[i][j].Relu_derivative(NN[i][j].z);
				NN[i][j].theta += NN[i + 1][k].theta * NN[i + 1][k].weight[j] * (1 - NN[i][j].a) * NN[i][j].a;
			}		
		}
	}
		

	//将权重和偏置更新值放入vector
	for (unsigned i = NN.size() - 1; i > 0; i--) 
	{
		for (unsigned j = 0; j < NN[i].size(); j++) 
		{
			for (unsigned k = 0; k < NN[i-1].size(); k++) 
			{
				NN[i][j].Batchweight[k].push_back(NN[i][j].theta * NN[i-1][k].a);
			}
			NN[i][j].Batchbias.push_back(NN[i][j].theta);
		}	
	}
}

void Model::Update_paramter() {
	for (unsigned i = NN.size() - 1; i > 0; i--)
	{
		for (unsigned j = 0; j < NN[i].size(); j++)
		{
			for (unsigned k = 0; k < NN[i - 1].size(); k++)
			{
				float batchweight = 0;
				for (auto it : NN[i][j].Batchweight[k])
					batchweight += it;
				NN[i][j].Batchweight[k].clear();
				NN[i][j].weight[k] -= batchweight / batch_size * learning_rate;
			}
			float batchbias = 0;
			for (auto it : NN[i][j].Batchbias)
				batchbias += it;
			NN[i][j].Batchbias.clear();
			NN[i][j].bias -= batchbias / batch_size * learning_rate;
		}
	}
}

void Model::Train_data(const string& Image_path)
{
	ifstream inImages(Image_path);
	string imageName;
	//vector<string> vecImages;
	
	while(inImages >> imageName)
	{
		Mat src = imread(imageName, 0);
		resize(src, src, input_size);
		vecImages.push_back(src);
		vecLabels.push_back(Get_Label(imageName));
		Set_Label.insert(Get_Label(imageName));
	}
	inImages.close();
	int total_num = vecImages.size();
	cout << "-----------------Start Training------------------" << endl;
	cout << "Total_num:" << total_num<<endl;
	cout << "The answer :";
	for (auto it : Set_Label)
		cout << it << " ";
	cout << endl;
	while (epoch-- > 0)
	{
		Shuffle_data(vecImages,vecLabels);
		for (unsigned i = 0; i < total_num; i++)
		{	
			Mat src = vecImages[i];
			vector<float> vec(src.reshape(0,1));
			vector<float> output = Feed_forward(vec);
			vector<float> label(output_num, 0);
			auto it = Set_Label.find(vecLabels[i]);
			unsigned count = 1;
			while (++it!= Set_Label.end())
				count++;
			label[output_num - count] = 1;
			Back_propagation(label);
			if ((i+1)% batch_size == 0)
			{
				Update_paramter();
			}
		}
		Evalution_model();
	}
	cout << "-----------------End Training----------------- "<<endl;
}

void Model::Test_data(const string& Image_path) 
{
	ifstream inImages(Image_path);
	string imageName;
	vector<string> vecImages;
	int true_num = 0;
	while (inImages >> imageName)
	{
		vecImages.push_back(imageName);
		Set_Label.insert(Get_Label(imageName));
	}
	inImages.close();
	int total_num = vecImages.size();
	cout << "-----------------Start Testing----------------" << endl;
	cout << "Total_num:" << total_num << endl;
	for (unsigned i = 0; i < total_num; i++)
	{
		Mat src = imread(vecImages[i], 0);
		resize(src, src, input_size);
		int ImageLebel = Get_Label(vecImages[i]);
		vector<float> vec(src.reshape(0, 1));
		vector<float> ans = Feed_forward(vec);
		if (Judge_result(ans, ImageLebel))
			true_num++;
		for (auto it : ans)
			cout << it << " ";
		cout<<endl;
	}
	cout << "accuracy:" << (float)true_num /total_num << endl;
	cout << "-----------------End Testing----------------" << endl;
}
void Model::Evalution_model() 
{
	int total_num = vecImages.size();
	int True_num = 0;
	for (unsigned i = 0; i < total_num; i++)
	{
		float loss_temp = 0;
		Mat src = vecImages[i];
		vector<float> vec(src.reshape(0, 1));
		vector<float> output = Feed_forward(vec);
		vector<float> label(output_num, 0);
		auto it = Set_Label.find(vecLabels[i]);
		unsigned count = 1;
		while (++it != Set_Label.end())
			count++;
		label[output_num - count] = 1;
		for (unsigned i = 0; i < NN.back().size(); i++)
		{
			//loss_temp += pow((label[i] - NN.back()[i].a), 2) / 2;
			loss_temp += (-label[i] * log(NN.back()[i].a));
		}
		loss.push_back(loss_temp);
		//if (Judge_result(output, ImageLebel))
		if (Judge_result(output, vecLabels[i]))
			True_num++;
	}
	float Total_loss = 0;
	for (auto it : loss)
		Total_loss += it;
	loss.clear();
	float loss_epoch = Total_loss / total_num;
	float accuracy = (float)True_num / total_num;
	cout << "Epoch_num:" <<epoch+1<<" "<< "loss:" <<loss_epoch<<" "<<"accuracy: "<<accuracy<< endl;
}

bool Model::Judge_result(vector<float> input, signed answer) 
{
	unsigned count = 0;
	int res = 0;
	for (unsigned j = 1; j < input.size(); j++)
	{
		if (input[j] > input[count])
			count = j;
	}
	auto it = Set_Label.begin();
	for (unsigned i = 0; i <= count; i++)
		res = *it++;
	if (res != answer)
		return false;
	return true;
}

void Model::Shuffle_data(vector<Mat> &vecImage, vector<int>& vecLabel)
{
	int size = vecImage.size()*0.3;
	for (unsigned i = 0; i < size; i++) 
	{
		int temp1 = rand() % vecImage.size();
		int temp2 = rand() % vecImage.size();

		Mat temp ;
		int change;

		temp = vecImage[temp1];
		vecImage[temp1] = vecImage[temp2];
		vecImage[temp2] = temp;

		change = vecLabel[temp1];
		vecLabel[temp1] = vecLabel[temp2];
		vecLabel[temp2] = change;
	}
}

int Model::Get_Label(string Imagename) 
{
	int len = 0,begin = 0,end = 0;
	while (Imagename[len++] != '/') begin = len;
	while (Imagename[len++] != '/') end = len;
	string s(Imagename.substr(begin + 1, end - begin - 1));
	return stoi(s);
}
