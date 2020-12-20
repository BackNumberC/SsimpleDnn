#include<iostream>
#include<vector>
#include "Model.h"
#include "Neuron.h"
#include<set>
#include "XmlHelp.h"
#include<map>
//#define TRAIN
using namespace std;
int main() {
	  
#ifdef TRAIN
	//模型初始化（自动生成输入层）
	//learning_rate  batch_size  epoch  output_num  Data_size
	Model model(0.1, 100, 10, 10, Size(28, 28));

	//添加全连接层（包括输出层）
	model.Add_layer(10);

	//设置训练集和测试集
	model.Train_data("10kind.txt/myImageList.txt");
	model.Test_data("10kind.txt/myImagetestList.txt");
	
	//生成XML文件存储模型
	XMLFile xmlFile("Binary10kinds.xml");
	xmlFile.CreateXML(model);
#endif // TRAIN

#ifndef TRAIN
	XMLFile xmlFile("Binary10kinds.xml");
	Model model = xmlFile.LoadXML();
	model.Test_data("10kind.txt/myImagetestList.txt");
#endif // !TRAIN

	return 0;
}