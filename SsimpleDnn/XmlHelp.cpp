#include "tinyxml.h"
#include <string>
#include <map>
#include "Model.h"
#include "XmlHelp.h"
#include "Math.h"
#include <iostream>
#pragma warning(disable:4996)
using namespace std;

XMLFile::XMLFile(const char* xmlFileName)
{
	m_xmlFileName = new char[20];
	strcpy(m_xmlFileName, xmlFileName);

	//创建XML文档指针
	m_pDocument = new TiXmlDocument(m_xmlFileName);
	m_pDeclaration = NULL;
}

XMLFile::~XMLFile()
{
	if (m_xmlFileName != NULL)
		delete m_xmlFileName;

	if (m_pDocument != NULL)
		delete m_pDocument;

	if (m_pDeclaration != NULL)
		delete m_pDeclaration;
}

void XMLFile::CreateXML(Model& model)
{
	cout << "---------------Start Saving----------------" << endl;
	if (NULL == m_pDocument)
	{
		return;
	}

	//声明XML
	m_pDeclaration = new TiXmlDeclaration("1.0", "", "");
	if (NULL == m_pDeclaration)
	{
		return;
	}
	m_pDocument->LinkEndChild(m_pDeclaration);

	//创建根节点
	TiXmlElement* pRoot = new TiXmlElement("NetworkNeuron");
	if (NULL == pRoot)
	{
		return;
	}
	//关联XML文档，成为XML文档的根节点
	m_pDocument->LinkEndChild(pRoot);

	pRoot->SetAttribute("Epoch", model.epoch);
	pRoot->SetAttribute("Batch_size", model.batch_size);
	pRoot->SetAttribute("Output_num", model.output_num);
	pRoot->SetAttribute("Input_num", model.input_num);

	string bias = to_string(model.accuracy);
	TiXmlElement* pAccuracy = new TiXmlElement("Accuracy");
	TiXmlText* pAccuracyText = new TiXmlText(bias.c_str());
	pAccuracy->LinkEndChild(pAccuracyText);

	string loss = to_string(model.loss_epoch);
	TiXmlElement* ploss = new TiXmlElement("Loss");
	TiXmlText* plossText = new TiXmlText(loss.c_str());
	ploss->LinkEndChild(plossText);

	string rate = to_string(model.learning_rate);
	TiXmlElement* pRate = new TiXmlElement("Learning_rate");
	TiXmlText* pRateText = new TiXmlText(rate.c_str());
	pRate->LinkEndChild(pRateText);

	pRoot->LinkEndChild(pAccuracy);
	pRoot->LinkEndChild(ploss);
	pRoot->LinkEndChild(pRate);


	int size = model.NN.size();
	for (int i = 1; i < size; i++) 
	{
		//创建孩子节点
		TiXmlElement* pLayer = new TiXmlElement("Layer");
		if (NULL == pLayer)
		{
			return;
		}
		pLayer->SetAttribute("No", i);
		pLayer->SetAttribute("Num", size);

		int length = model.NN[i].size();
		for (int k = 0; k < length; k++) 
		{
			TiXmlElement* pNeuron = new TiXmlElement("Neuron");
			pNeuron->SetAttribute("No", k+1);
			pNeuron->SetAttribute("Num", length);

			TiXmlElement* pWeight = new TiXmlElement("Weight");
			TiXmlText* pWeightText;

			int num = model.NN[i][k].weight.size();
			for (int j = 0; j < num; j++) 
			{
				string wieght = to_string(model.NN[i][k].weight[j]) + "  ";
				pWeightText = new TiXmlText(wieght.c_str());
				pWeight->LinkEndChild(pWeightText);
			}

			string bias = to_string(model.NN[i][k].bias);
			TiXmlElement* pBias = new TiXmlElement("Bias");
			TiXmlText* pBiasText = new TiXmlText(bias.c_str());
			pBias->LinkEndChild(pBiasText);

			pNeuron->LinkEndChild(pWeight);
			pNeuron->LinkEndChild(pBias);
			pLayer->LinkEndChild(pNeuron);
		}
		pRoot->LinkEndChild(pLayer);
	}
	m_pDocument->SaveFile(m_xmlFileName);

	cout << "-----------------End Saving----------------" << endl;
}

Model XMLFile::LoadXML() 
{
	m_pDocument->LoadFile(m_xmlFileName);
	
	TiXmlElement* pNN = m_pDocument->RootElement();

	vector<int> Param;
	TiXmlAttribute* pAttr = NULL;
	for (pAttr = pNN->FirstAttribute(); pAttr != NULL; pAttr = pAttr->Next())
	{
		Param.push_back(atoi(pAttr->Value()));
	}

	TiXmlElement* pAcc = pNN->FirstChildElement();
	TiXmlElement* ploss = pAcc->NextSiblingElement();
	TiXmlElement* pRate = ploss->NextSiblingElement();

	//加载Model参数
	Model model(atof(pRate->GetText()), Param.at(0), Param.at(1), Param.at(2), Size(sqrt(Param.at(3)), sqrt(Param.at(3))));

	//加载Neuron参数
	TiXmlElement* pLayer = pRate->NextSiblingElement();
	while (pLayer) 
	{
		vector<Neuron> layer;
		TiXmlElement* pNeuron = pLayer->FirstChildElement();
		while (pNeuron) 
		{
			Neuron n;

			TiXmlElement* pWeight = pNeuron->FirstChildElement();
			TiXmlElement* pBias = pWeight->NextSiblingElement();

			string pAcc_Value = pWeight->GetText();
			istringstream in(pAcc_Value);
			vector<float> result((istream_iterator<float>(in)), istream_iterator<float>());

			n.weight = result;
			n.bias = atof(pBias->GetText());
			layer.push_back(n);
			pNeuron = pNeuron->NextSiblingElement();
		}
		model.NN.push_back(layer);
		pLayer = pLayer->NextSiblingElement();
	}

	return model;
}