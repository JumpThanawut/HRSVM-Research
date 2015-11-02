#include<stdio.h>
#include<iostream>
#include<vector>
#include<algorithm>
#include<stdlib.h>
#include<float.h>
#include<time.h>
#include<set>
#include<map>
#include "svm.h"
#include "hrsvm_common.h"
using namespace std;

const double epsilon = 1E-5;

typedef pair<int, double> mypair;
struct Cmp2
{
	bool operator()(const mypair &a, const mypair &b)
	{
		return a.second < b.second;
	}
}mypairCmp;

/*--------------RSVM Module-----------------*/
/*  Author    : Piyapan Poomsirivilai       */
/*  Supervisor: Peerapon Vateekul           */
/*  Date      : 28 July 2013                */
/*------------------------------------------*/  

void RSVM(svm_problem &prob, svm_model* &model, int randomSeed /*= -1*/)
{
	typedef vector<vector<int> > IVec2d;
	typedef vector<IVec2d> IVec3d;

	vector<mypair> pred; //prediction value from training example (collected from prediction files).
	vector<double> b; //break points
	vector<double> pIdx, nIdx; //positive class index of sorted "pred" variable.
	vector<int> pIdxgr, nIdxgr;
	const int K=5; //number of parition in Partitioning process. K must >= 1
	int i, j, k, l;

	if(randomSeed >= 0)
		srand(randomSeed);
	else
		srand(time(NULL));
			
	{
		vector<int> Pmap, Nmap;
		for(i=0;i<prob.l;i++)
			if(myabs(prob.y[i]-1.0) < epsilon)
				Pmap.push_back(i);
			else
				Nmap.push_back(i);

		printf("RSVM : positive size = %d, negative size = %d\n", (int)Pmap.size(), (int)Nmap.size());

		for(i=0;i<(int)Pmap.size();i++)
			swap(Pmap[rand()%Pmap.size()], Pmap[rand()%Pmap.size()]);
		for(i=0;i<(int)Nmap.size();i++)
			swap(Nmap[rand()%Nmap.size()], Nmap[rand()%Nmap.size()]);

		//resampling positive for 2/3.
		for(i=0;i<(int)Pmap.size();i+=3)
			pred.push_back(make_pair(1, svm_predict(model, prob.x[Pmap[i]])));  
		for(i=1;i<(int)Pmap.size();i+=3)
			pred.push_back(make_pair(1, svm_predict(model, prob.x[Pmap[i]])));  
		for(i=2;i<(int)Pmap.size() && (int)pred.size()<200;i+=3)
			pred.push_back(make_pair(1, svm_predict(model, prob.x[Pmap[i]])));  

		//resampling negative for 1/2.
		j = 0; 
		for(i=0;i<(int)Nmap.size();i+=2,j++)
			pred.push_back(make_pair(-1, svm_predict(model, prob.x[Nmap[i]])));  
		for(i=1;i<(int)Nmap.size()&&j<400;i+=2,j++)
			pred.push_back(make_pair(-1, svm_predict(model, prob.x[Nmap[i]])));  
	}

	//sort svm_score in ascending order.
	sort(pred.begin(), pred.end(), mypairCmp); 
	
	//put svm_score of truly positive class in to pIdx (in ascending order) 
	//put svm_score of truly negative class in to nIdx (in ascending order) 
	for(i=0;i<(int)pred.size();i++)
		if(pred[i].first == 1)
			pIdx.push_back(pred[i].second);
		else
			nIdx.push_back(pred[i].second);
	
	//mapping groups.
	pIdxgr.resize(pIdx.size());
	nIdxgr.resize(nIdx.size());
	for(i=0;i<K;i++)
		for(j=i;j<(int)pIdx.size();j+=K)
			pIdxgr[j] = i;
	
	for(i=0;i<K;i++)
		for(j=i;j<(int)nIdx.size();j+=K)
			nIdxgr[j] = i;

	//swap it randomly.
	for(i=0;i<(int)pIdxgr.size();i++)
		swap(pIdxgr[rand()%pIdxgr.size()], pIdxgr[rand()%pIdxgr.size()]);
	for(i=0;i<(int)nIdxgr.size();i++)
		swap(nIdxgr[rand()%nIdxgr.size()], nIdxgr[rand()%nIdxgr.size()]);

	//sumInP[i][j] means number of group i in PIdxgr[0..j]; K is number of partitions
	vector<vector<int> > sumInP(K, vector<int>(pIdx.size())), sumInN(K, vector<int>(nIdx.size()));
	for(j=0;j<K;j++)
		if(pIdxgr[0] == j)
			sumInP[j][0] = 1;
		else
			sumInP[j][0] = 0;

	for(j=0;j<K;j++)
		for(i=1;i<(int)pIdxgr.size();i++)
			if(pIdxgr[i] == j)
				sumInP[j][i] = sumInP[j][i-1]+1;
			else
				sumInP[j][i] = sumInP[j][i-1];

	for(j=0;j<K;j++)
		if(nIdxgr[0] == j)
			sumInN[j][0] = 1;
		else
			sumInN[j][0] = 0;

	for(j=0;j<K;j++)
		for(i=1;i<(int)nIdxgr.size();i++)  
			if(nIdxgr[i] == j)
				sumInN[j][i] = sumInN[j][i-1]+1;
			else
				sumInN[j][i] = sumInN[j][i-1];
	
	//seting up breakpoints
	b.clear();
	b.push_back(0.0);
	for(i=0;i<(int)pred.size()-1;i++)
		if(pred[i].first != pred[i+1].first) 
			b.push_back((pred[i].second+pred[i+1].second)/2.0);

	vector<int> posbIdx(b.size()), negbIdx(b.size());
	for(i=0;i<(int)b.size();i++)
	{
		vector<double>::iterator posbIdxit, negbIdxit;
		posbIdxit = lower_bound(pIdx.begin(), pIdx.end(), b[i]);   
		negbIdxit = lower_bound(nIdx.begin(), nIdx.end(), b[i]);

		posbIdx[i] = posbIdxit - pIdx.begin();
		negbIdx[i] = negbIdxit - nIdx.begin();
	}

	double sumB=0.0;

	for(l=1;l<=K;l++)
		for(i=0;i<l;i++)
		{
			double maxF = -1.0;
			double maxB = 0.0;
			for(j=0;j<(int)b.size();j++)
			{
				int tp, tn, fp, fn;
				tp = tn = fp = fn = 0;

				if(posbIdx[j]-1 >= 0)
					for(k=i;k<K;k+=l)
						fn += sumInP[k][posbIdx[j]-1];
				for(k=i;k<K;k+=l)
					tp += sumInP[k][pIdx.size()-1];
				tp-=fn;

				if(negbIdx[j]-1 >= 0)
					for(k=i;k<K;k+=l)
						tn += sumInN[k][negbIdx[j]-1];
				for(k=i;k<K;k+=l)
					fp += sumInN[k][nIdx.size()-1];
				fp-=tn;

				double F = 0.0;
				if(tp != 0 || fp != 0 || fn != 0)
					F = 2*tp/(double)(2*tp+fp+fn);

				if(F > maxF)
				{
					maxF = F;
					maxB = b[j];
				}
			}

			sumB+=maxB;
		}

	double RSVMTheta = sumB/(double)(K*(K+1)/2), newF;

	{
		vector<double>::iterator posbIdxit, negbIdxit;
		posbIdxit = lower_bound(pIdx.begin(), pIdx.end(), RSVMTheta);   
		negbIdxit = lower_bound(nIdx.begin(), nIdx.end(), RSVMTheta);

		int posbIdx_, negbIdx_;
		posbIdx_ = posbIdxit - pIdx.begin();
		negbIdx_ = negbIdxit - nIdx.begin();

		int tp_, tn_, fp_, fn_;
		tp_ = tn_ = fp_ = fn_ = 0;

		if(posbIdx_-1 >= 0)
			for(k=0;k<K;k++)
				fn_ += sumInP[k][posbIdx_-1];
		for(k=0;k<K;k++)
			tp_ += sumInP[k][pIdx.size()-1];
		tp_-=fn_;

		if(negbIdx_-1 >= 0)
			for(k=0;k<K;k++)
				tn_ += sumInN[k][negbIdx_-1];
		for(k=0;k<K;k++)
			fp_ += sumInN[k][nIdx.size()-1];
		fp_-=tn_;

		newF = 0.0;
		if(tp_ != 0 || fp_ != 0 || fn_ != 0)
			newF = 2*tp_/(double)(2*tp_+fp_+fn_);
	}

	{
		vector<double>::iterator posbIdxit, negbIdxit;
		posbIdxit = lower_bound(pIdx.begin(), pIdx.end(), 0.0);   
		negbIdxit = lower_bound(nIdx.begin(), nIdx.end(), 0.0);

		int posbIdx_, negbIdx_;
		posbIdx_ = posbIdxit - pIdx.begin();
		negbIdx_ = negbIdxit - nIdx.begin();

		int tp_, tn_, fp_, fn_;
		tp_ = tn_ = fp_ = fn_ = 0;

		if(posbIdx_-1 >= 0)
			for(k=0;k<K;k++)
				fn_ += sumInP[k][posbIdx_-1];
		for(k=0;k<K;k++)
			tp_ += sumInP[k][pIdx.size()-1];
		tp_-=fn_;

		if(negbIdx_-1 >= 0)
			for(k=0;k<K;k++)
				tn_ += sumInN[k][negbIdx_-1];
		for(k=0;k<K;k++)
			fp_ += sumInN[k][nIdx.size()-1];
		fp_-=tn_;

		double oldF = 0.0;
		if(tp_ != 0 || fp_ != 0 || fn_ != 0)
			oldF = 2*tp_/(double)(2*tp_+fp_+fn_);

		if(newF < oldF)
			RSVMTheta = 0.0;
	}

	printf("RSVM: Old Rho = %lg, New Rho = %lg\n", model->rho[0], model->rho[0]+RSVMTheta);
	model->rho[0]+=RSVMTheta;
}

void feature_selection(svm_problem prob, svm_node* x_space, int max_index, int elements, map<int, int> &selected)
{
	//using Fscore feature selection algorithm from Yi-Wei Chen and Chih-Jen Lin;
	//source code (python): http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#feature_selection_tool
	
	//sum[i] means the sum of i-th feature; max_index is the number of feature. 
	//sumpos[i] means the sum of i-th feature of only positive examples.
	//sumsqpos[i] means the quadratic sum of i-th feature of only positive examples.
	//sumneg[i] means the sum of i-th feature of only negative examples.
	//sumsqneg[i] means the quadratic sum of i-th feature of only negative examples.
	//numpos, numneg is the number of positive, negative examples.
	vector<double> sum(max_index+1, 0.0), sumpos(max_index+1, 0.0), sumneg(max_index+1, 0.0), sumsqpos(max_index+1, 0.0), sumsqneg(max_index+1, 0.0);
	int numpos = 0, numneg = 0;
	int i, j, k, N, f;
	
	N = prob.l;
	for(i=0;i<N;i++)
	{
		j=0;  
		
		if(prob.y[i] > 0.0)
			numpos++;
		else
			numneg++;
		
		while(prob.x[i][j].index != -1)
		{
			sum[prob.x[i][j].index]+=prob.x[i][j].value;
			if(prob.y[i] > 0.0)
			{
				sumpos[prob.x[i][j].index]+=prob.x[i][j].value;
				sumsqpos[prob.x[i][j].index]+=prob.x[i][j].value*prob.x[i][j].value;
			}
			else
			{
				sumneg[prob.x[i][j].index]+=prob.x[i][j].value;
				sumsqneg[prob.x[i][j].index]+=prob.x[i][j].value*prob.x[i][j].value;
			}
			
			j++;
		}
	}

	vector<pair<int, double> > F(max_index+1);
	double sumF = 0.0, minF=DBL_MAX;
	for(i=0;i<=max_index;i++)
	{
		double SB = 0.0;
		SB += numpos*(sumpos[i]/(double)numpos - sum[i]/(double)N)*(sumpos[i]/(double)numpos - sum[i]/(double)N);
		SB += numneg*(sumneg[i]/(double)numneg - sum[i]/(double)N)*(sumneg[i]/(double)numneg - sum[i]/(double)N);

		double SW = 1E-6;
		SW += sumsqpos[i] - sumpos[i]*sumpos[i]/(double)numpos;
		SW += sumsqneg[i] - sumneg[i]*sumneg[i]/(double)numneg;

		F[i].first = i;
		F[i].second = SB/SW;
		//printf("%lg\n", F[i].second);
		sumF+=F[i].second;
		if(F[i].second < minF)
			minF = F[i].second;
	}

	map<int, int>::iterator it;

	{
		//cumulative sum method (97%)

		sort(F.begin(), F.end(), mypairCmp);

		double collect = 0.0;
		i=F.size()-1;

		while(i >= 0 && (i+20 > (int)F.size() || collect < 0.97*sumF))
		{
			selected[F[i].first] = 0;
			collect+=F[i].second;
			i--;
			j++;
		}

		j = 1;
		for(it = selected.begin();it!=selected.end();it++)
		it->second = j++;
	}

	printf("cut-out %d features\n", max_index+1-(int)selected.size());

	//delete irrelevant feature from x_space and re-number the feature.
	if(x_space != NULL)
	{
		int k=1;
		for(i=0;i<elements;i++)
		{
			j = i;

			while(x_space[i].index != -1)
			{
				it = selected.find(x_space[i].index);
				if(it != selected.end())
				{
					if(i != j)
						x_space[j] = x_space[i];

					x_space[j].index = it->second;
					j++;
				}

				i++;
			}

			x_space[j].index = -1;
		}
	}
	else
	{
		for(i=0;i<N;i++)
		{
			j = 0;
			k = 0;
			while(prob.x[i][j].index != -1)
			{
				it = selected.find(prob.x[i][j].index);
				if(it != selected.end())
				{
					if(k != j)
						prob.x[i][k] = prob.x[i][j];

					prob.x[i][k].index = it->second;
					k++;
				}

				j++;
			}

			prob.x[i][k].index = -1;   
		}
	}

}

