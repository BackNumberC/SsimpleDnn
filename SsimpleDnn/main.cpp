#include<iostream>
#include<vector>
#include "Model.h"
#include "Neuron.h"
#include<set>
#include "XmlHelp.h"
#include<map>
using namespace std;
int main() {
	//learning_rate  batch_size  epoch  output_num  Data_size  
	/*Model model(0.1,10,10,2,Size(28, 28));
	model.Add_layer(36);
	model.Add_layer(2);
	model.Train_data("2kind.txt/myImageList.txt");
	model.Test_data("2kind.txt/myImagetestList.txt");*/

	XMLFile xmlFile("Binary2kinds.xml");
	//xmlFile.CreateXML(model);
	Model model = xmlFile.LoadXML();
	model.Test_data("2kind.txt/myImagetestList.txt");
	return 0;
}