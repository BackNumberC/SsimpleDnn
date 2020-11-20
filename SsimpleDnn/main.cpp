#include<iostream>
#include<vector>
#include "Model.h"
#include "Neuron.h"
#include<set>
#include "XmlHelp.h"
#include<map>
#define TRAIN
using namespace std;
int main() {
	  
#ifdef TRAIN
	//learning_rate  batch_size  epoch  output_num  Data_size
	Model model(0.1, 100, 10, 10, Size(28, 28));
	//model.Add_layer(36);
	model.Add_layer(10);
	model.Train_data("10kind.txt/myImageList.txt");
	model.Test_data("10kind.txt/myImagetestList.txt");

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