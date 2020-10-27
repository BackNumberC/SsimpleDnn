#pragma once
#ifndef MINST_H
#define MINST_H
#include<inttypes.h>
#include<string>
using namespace std;
uint32_t swap_endian(uint32_t val);
void readAndSave(const string& mnist_img_path, const string& mnist_label_path)
#endif // !MINST_H
