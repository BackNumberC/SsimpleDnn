#pragma once
#ifndef XMLHELP_H
#define XMLHELP_H
#include "tinyxml.h"
#include <map>
using namespace std;
class XMLFile
{
public:
	XMLFile(const char* xmlFileName);
	~XMLFile();
	void CreateXML(Model& model);//����XML�ļ�
	Model LoadXML();// ����XML�ļ�
	

private:
	char* m_xmlFileName;
	TiXmlDocument* m_pDocument;
	TiXmlDeclaration* m_pDeclaration;
};

#endif // !XMLHELP_H

