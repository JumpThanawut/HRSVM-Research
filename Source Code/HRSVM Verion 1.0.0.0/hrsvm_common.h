#ifndef _HRSVM_COMMON_H
#define _HRSVM_COMMON_H
/*
#ifdef __cplusplus
extern "C" {
#endif
*/

#include "svm.h"
#include<vector>
#include<map>

void RSVM(svm_problem &prob, svm_model* &model, int randomSeed=-1);
void feature_selection(svm_problem prob, svm_node* x_space, int max_index, int elements, std::map<int, int> &selected);

#endif