/* 
 * File:   main.cpp
 * Author: Aris Kosmopoulos
 *
 * Created on November 9, 2011, 3:24 PM
 */

#include <cstdlib>
#include <iostream>
#include "graph.h"
#include "distanceCopmuter.h"
#include <fstream>
#include <string>
#include<vector>
#include<string>
#include <map>
#include <set>
#include<string.h>
using namespace std;

/*
 * 
 */

void hierarchy_measure(string hf, string truef, string predf, double &LCA_F, double &LCA_P, double &LCA_R, double &ACC) 
{      
    char tmp[2000], tmp2[2000];    
    strcpy(tmp, hf.c_str());
    strcpy(tmp2, truef.c_str());
    Graph G (tmp, tmp2);
    //cout << "Number of G nodes: " << G.getG().size() <<endl;   

    strcpy(tmp, truef.c_str());
    ifstream finGS (tmp);

    strcpy(tmp, predf.c_str());
    ifstream finPr (tmp);
   
    int maximumDistance = 2000000000;// atoi(tmp);   
    int maxError = 5;//atoi(tmp);

    string line1,line2;
    double sum = 0.0;
    double sumP = 0.0;
    double sumR = 0.0;
    double MGIAdistance = 0.0;
    int count = 0;
    while (getline(finGS,line1)) {
        if (line1 != "") {
	    
            getline(finPr,line2);
            set<int> gsLCA;
            set<int> prLCA;
	    set<int> gsMGIA = fillSet(line1);
            set<int> prMGIA = fillSet(line2);
	    
	    set<int> Allancs;    
	    set<int>::iterator s_iter,s_iter2;
            for (s_iter=prMGIA.begin();s_iter!=prMGIA.end();s_iter++) {                
                    set<int> ancs = G.getAllAncestors(*s_iter,maximumDistance);
		    
                    Allancs = addSets (Allancs,ancs);                
            }
            for (s_iter=prMGIA.begin();s_iter!=prMGIA.end();s_iter++) {
	      s_iter2 = Allancs.find(*s_iter);
	      if (s_iter2 == Allancs.end())
		prLCA.insert(*s_iter);
	    }
	    Allancs.clear();
	    for (s_iter=gsMGIA.begin();s_iter!=gsMGIA.end();s_iter++) {                
                    set<int> ancs = G.getAllAncestors(*s_iter,maximumDistance);                    
                    Allancs = addSets (Allancs,ancs);                
            }
            for (s_iter=gsMGIA.begin();s_iter!=gsMGIA.end();s_iter++) {
	      s_iter2 = Allancs.find(*s_iter);
	      if (s_iter2 == Allancs.end())
		gsLCA.insert(*s_iter);
	    }
	    double p=0.0;
	    double r=0.0;
	    double LCA_F=0.0;
	    double MGIA=0.0;
	    
	    if (gsLCA.size() != gsMGIA.size() || prLCA.size() != prMGIA.size()) {
	      DistanceComputer dcLCA (G,prLCA,gsLCA, maximumDistance,maxError);
	      DistanceComputer dcMGIA (G,prMGIA, gsMGIA, maximumDistance,maxError);
	      LCA_F = dcLCA.getF(p,r);	      
	      sum += LCA_F;
	      sumP += p;	      
	      sumR += r;
	      MGIA = dcMGIA.getMGIA();
	      MGIAdistance += MGIA;
	    }
	    else {	     
	      DistanceComputer dc (G,prMGIA, gsMGIA, maximumDistance,maxError);
	      
	      LCA_F = dc.getF(p,r);	      
	      sum += LCA_F;
	      sumP += p;	      
	      sumR += r;
	      
	      MGIA = dc.getMGIA();	      
	      MGIAdistance += MGIA;
	      
	    }

            count ++;    
        }
    }    

    LCA_F = sum/count;
    LCA_P = sumP/count;
    LCA_R = sumR/count;
    ACC = MGIAdistance/count;
    /*
    fprintf(out, "LCA F : %g\n", sum, sum/count);
    fprintf(out, "LCA P : %g\n", sumP/count);
    fprintf(out, "LCA R : %g\n", sumR/count);
    fprintf(out, "MGIA  : %g\n", MGIAdistance/count);
    */
    finGS.close();
    finPr.close();
}

void hierPrecRecF1 (string hf, string truef, string predf, FILE *out) //(int argc, char** argv, vector<double* >& resultsPerInstance, FILE *out) 
{
    char tmp[2000], tmp2[2000];    
    strcpy(tmp, hf.c_str());
    strcpy(tmp2, truef.c_str());
    Graph G (tmp, tmp2);
    //cout << "Number of G nodes: " << G.getG().size() <<endl;   
    strcpy(tmp, truef.c_str());
    ifstream finGS (tmp);

    strcpy(tmp, predf.c_str());
    ifstream finPr (tmp);

    int maximumDistance = 2000000000;

    string line1,line2;
    double sumPre = 0.0;
    double sumRec = 0.0;
    double sumF = 0.0;
    //double HammingDist = 0.0;
    int HammingDist = 0;
    int count = 0;
    
    map<int,set<int> > ancestors;
    map<int,set<int> >::iterator anc_iter;
    set<int>::iterator s_iter;

    while (getline(finGS,line1)) { 
        if (line1 != "") {
            
            getline(finPr,line2);
            set<int> gs = fillSet(line1);
            set<int> pr = fillSet(line2);
            
            set<int> AncsP;
            set<int> AncsT;            
            
            for (s_iter=gs.begin();s_iter!=gs.end();s_iter++) {
                anc_iter = ancestors.find(*s_iter);
                if (anc_iter == ancestors.end()) {
                    set<int> ancs = G.getAllAncestors(*s_iter,maximumDistance);
                    ancs.insert((*s_iter));
                    ancestors[*s_iter] = ancs;                    
                    AncsT = addSets (AncsT,ancs);
                }
                else
                    AncsT = addSets (AncsT,anc_iter->second);
            }
                
            for (s_iter=pr.begin();s_iter!=pr.end();s_iter++) {
                anc_iter = ancestors.find(*s_iter);
                if (anc_iter == ancestors.end()) {
                    set<int> ancs = G.getAllAncestors(*s_iter,maximumDistance);
                    ancs.insert((*s_iter));
                    ancestors[*s_iter] = ancs;
                    AncsP = addSets (AncsP,ancs);
                }
                else
                    AncsP = addSets (AncsP,anc_iter->second);
            }    
            set<int> PTInter = getIntrOfSets (AncsP,AncsT);
	    /*
	    s_iter = PTInter.find(0);
	    if (s_iter!=PTInter.end())
	      PTInter.erase(s_iter);
	    s_iter = AncsP.find(0);
	    if (s_iter!=AncsP.end())
	      AncsP.erase(s_iter);
	    s_iter = AncsT.find(0);
	    if (s_iter!=AncsT.end())
	      AncsT.erase(s_iter);
	    */
	    
//            cout << "PTInter = " <<  PTInter.size() << endl;
//            cout << "AncsP = " <<  AncsP.size() << endl;
//            cout << "AncsT = " <<  AncsT.size() << endl;
            double precision = (PTInter.size()/((double) AncsP.size()));            
            double recall = (PTInter.size()/((double) AncsT.size()));
            double F1 = 0.0;
            if (precision != 0.0 || recall != 0.0)
                F1 = ((2*precision*recall)/(precision+recall));
            sumF += F1;
            sumPre += precision;
            sumRec += recall;
            
            //ForHammingDistance
            set<int> uniqueP = getSubsetMinusCommon (AncsP,PTInter);
            set<int> uniqueT = getSubsetMinusCommon (AncsT,PTInter);
            double HD = (uniqueP.size() + uniqueT.size());                                                
            HammingDist += HD;
            count ++;
        }
    }     
    fprintf(out, "count                  : %d\n", count);
    fprintf(out, "Hierarchical Precision : %g\n", sumPre/count);
    fprintf(out, "Hierarchical Recall    : %g\n", sumRec/count);
    fprintf(out, "Hierarchical F1        : %g\n", sumF/count);
    fprintf(out, "SDL with all ancestors : %g\n", HammingDist/(double)count);
    finGS.close();
    finPr.close();
}


int main(int argc, char *argv[])
{
 double f, p, r, acc;

 hierarchy_measure(string(argv[1]), string(argv[2]), string(argv[3]), f, p, r, acc);
 printf("%lg\n%lg\n%lg\n%lg", f, p, r, acc);
}
