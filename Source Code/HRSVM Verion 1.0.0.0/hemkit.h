#ifndef _HEMKIT_H
#define _HEMKIT_H
#include<string>
void hierarchy_measure(std::string hf, std::string truef, std::string predf, double &LCA_F, double &LCA_P, double &LCA_R, double &ACC);
void hierPrecRecF1(std::string hf, std::string truef, std::string predf, FILE *out); 
#endif