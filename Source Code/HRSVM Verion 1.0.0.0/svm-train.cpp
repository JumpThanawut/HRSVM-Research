#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <ctype.h>
#include <errno.h>
#include <vector>
#include <iostream>
#include <sstream> 
#include <algorithm>
#include <iterator>
#include <time.h>
#include<stdlib.h>
#include<set>
#include<float.h>
#include "svm.h"
#include "hrsvm_common.h"
#include<math.h>
#include<map>
#include<limits.h>
#include<sys/stat.h>
#if defined(_WIN32)
	#include<direct.h>
#endif

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;

//class_list[i] contains example index that belongs to class i.
// (In case of hierarchical, class_list[i] will contains example index j if and only if class i is the leaf of example j).
// class_list = 2D, where row=class i, column=example j
vector <int> *class_list;

//DAG graph which parent[i] contains the parent of index i, child[i] contains the child of index i.
vector<vector<int> > parent, child;

//variable tr_class contains only leaf. (using shrink_class_list)
//tr_class[i] contains leaf class that example i belongs. 
//  (In case of multilabel/binary/multiclass, which in flat classification, all class are considered leaf class.)
vector<vector<int> > tr_class;   

//root variable contains node which doesn't have any parents.
vector<int> root;
//deep[i] contains the all possible level of node i
vector<set<int> > depth;

//class_count[i] = number of examples that belongs to class i;
vector<int> class_count;

int max_depth = 0;
int class_no, randomSeed;

//use for map from string label to int and int to string label.
map<string, int> classMap;
map<int, string> numMap;
set<int> rare_class;
int rare_class_threshold = -1; //class that have number of data lower than threshold will be considered as rare class.

char model_folder_name[1024];
char model_file_name[1024];
char model_file_name_without_path[1024];

//check if file exist.
bool FileExist( const std::string& Name )
{
	#ifdef OS_WINDOWS
		struct _stat buf;
		int Result = _stat( Name.c_str(), &buf );
	#else
		struct stat buf;
		int Result = stat( Name.c_str(), &buf );
	#endif
		return Result == 0;
}

//exec function will call external command specified by user in variable "cmd"
//the output of this function is the output of the command.
string exec(const char* cmd) 
{
	#ifdef OS_WINDOWS
		FILE* pipe = _popen(cmd, "r");
		if (!pipe) return "ERROR";
		char buffer[128];
		std::string result = "";
		while(!feof(pipe)) {
			if(fgets(buffer, 128, pipe) != NULL)
				result += buffer;
		}
		_pclose(pipe);
		return result;
	#else
		FILE* pipe = popen(cmd, "r");
		if (!pipe) return "ERROR";
		char buffer[128];
		std::string result = "";
		while(!feof(pipe)) {
			if(fgets(buffer, 128, pipe) != NULL)
				result += buffer;
		}
		pclose(pipe);
		return result;
	#endif
}

void get_ancestor_list(int i, set<int> &ans)
{
	if(ans.find(i) != ans.end())
		return;

	int j;
	ans.insert(i);
	for(j=0;j<(int)parent[i].size();j++)
		get_ancestor_list(parent[i][j], ans);
}

bool valid_model_cmp (int i,int j) 
{ 
	return numMap[i].compare(numMap[j]) < 0; 
}

bool x_space_cmp(const svm_node &x, const svm_node &y)
{
	return x.index < y.index;
}

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: svm-train [options] training_set_file model_file\n"
	"options:\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"-a enable_R-SVM : Enable/disable R-SVM algorithm (default 0)\n"
	"-f enable_feature_selection : Enable/disable feature selection algorithm (default 0)\n"
	"-l minimum_example_in_class : Set the threshold which program will remove classes\n"
	"                              that have number of examples lower than threshold (default 0)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-v n: n-fold cross validation mode\n"
	"-x n: specify random seed in k-fold cross validation mode (default: random)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name, char *model_folder_name);
void read_problem(const char *filename);
void original_read_problem(const char *filename);
void do_cross_validation(char *input_file_name);

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int do_RSVM;
char hierarchical_classification_path[1050];
int nr_fold;
int max_index, elements; //use in read_problem()

static char *line = NULL;
static int max_line_len;

void set_kernel(int numOfFeature, int numOfExample, int numOfPosExample, struct svm_parameter &param)
{
	param.nr_weight = 0;

	if(numOfExample < numOfFeature)
	{
		param.svm_type = C_SVC; //C-SVC
		param.kernel_type = 0; //linear kernel
	}
	else if(numOfExample < 30*numOfFeature)
	{
		param.svm_type = C_SVC; //C-SVC
		param.kernel_type = 2; //gaussian kernel
		param.gamma = 0.1;
	}
	else
	{
		param.svm_type = C_SVC; //C-SVC
		param.kernel_type = 2; 
		param.gamma = 0.1;
	}

	if(numOfPosExample*100 > 15*numOfExample)
		param.C = 1.0;  
	else
		param.C = 2.0;
}

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void count_and_find_rare_class(int threshold, set<int> &ans)
{
	int i, j;

	class_count.clear();
	if(param.multiclass_classification || param.multilabel_classification)
	{  
		for(i=0;i<class_no;i++)
		{
			class_count.push_back((int)class_list[i].size());
			if((int)class_list[i].size() < threshold)
				ans.insert(i);
		}
	}
	else if(param.hierarchical_classification)
	{
		vector<int> count(class_no, 0);
		set<int>::iterator it;

		for(i=0;i<prob.l;i++)
		{
			set<int> tr_cl_ext;
			for(j=0;j<(int)tr_class[i].size();j++)
			{
				get_ancestor_list(tr_class[i][j], tr_cl_ext);
				tr_cl_ext.insert(tr_class[i][j]);
			}
	
			for(it=tr_cl_ext.begin();it!=tr_cl_ext.end();it++)
				count[*it]++;
		}

		class_count = count;
		if(threshold > 0)
			for(i=0;i<class_no;i++)
				if(count[i] < threshold)
					ans.insert(i);
	}
}

int main(int argc, char **argv)
{
	char input_file_name[1024];
	const char *error_msg;
	char model_file_name_temp[1024];
	int i, j;

	if(randomSeed >= 0)
		srand(randomSeed);
	else
		srand(time(NULL));
	parse_command_line(argc, argv, input_file_name, model_file_name, model_folder_name);

	// read_problem: output is &prob
	// &prob = (x, y); where x is 2D-array (row=example,col=feature) and value -1 = stopping value
	// prob.x[i][0].index, prob.x[i][0].value and y={-1,1} only since the original SVM is binary
	if(param.binary_classification)
		original_read_problem(input_file_name);
	else
		read_problem(input_file_name);
	error_msg = svm_check_parameter(&prob,&param);
	count_and_find_rare_class(rare_class_threshold, rare_class);

	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}

	//if it is k-fold cross validation.
	if(cross_validation)
	{
		do_cross_validation(input_file_name);     
	}
	else if(param.binary_classification)
	{
		#if defined(_WIN32)
			_mkdir(model_folder_name);    
		#else 
			mkdir(model_folder_name, 0777); // notice that 777 is different than 0777
		#endif

		map<int, int> selected_feature;
		if(param.do_feature_selection)
			feature_selection(prob, x_space, max_index, elements, selected_feature);

		model = svm_train(&prob,&param);

		//model->l is #support vectors; don't do R-SVM if there is no SVs (0).
		if(do_RSVM && model->l > 0)
			RSVM(prob, model, randomSeed);

		sprintf(model_file_name_temp, "%s.%s", model_file_name, numMap[1].c_str());
		if(svm_save_model(model_file_name_temp,model))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			exit(1);
		}
		svm_free_and_destroy_model(&model);

		FILE *center_model = fopen(model_file_name, "wb");
		fprintf(center_model, "classification_type : binary\n");
		fprintf(center_model, "model : %s %s\n", numMap[1].c_str(), numMap[-1].c_str());

		fprintf(center_model, "RSVM              : %d\n", do_RSVM);
		fprintf(center_model, "Feature_selection : %d\n", param.do_feature_selection);
		fprintf(center_model, "%d ", (int)selected_feature.size());
		map<int, int>::iterator it;
		for(it=selected_feature.begin();it!=selected_feature.end();it++)
			fprintf(center_model, "%X %X ", it->first, it->second);
		fprintf(center_model, "\n");

		fclose(center_model);
	}
	else if(param.multiclass_classification || param.multilabel_classification)
	{
		vector<svm_node> original_x_space;
		svm_problem probTmp = prob;
		int N = prob.l;
		vector<map<int, int> > selected_feature(class_no);

		#if defined(_WIN32)
			_mkdir(model_folder_name);    
		#else 
			mkdir(model_folder_name, 0777); // notice that 777 is different than 0777
		#endif

		if(param.do_feature_selection)
			original_x_space.assign(x_space, x_space+elements);

		//iterate each class and create one-vs-all classifier
		for(i=0;i<class_no;i++) 
		{
			printf("\nclass %s : ", numMap[i].c_str());
			if(rare_class.find(i) == rare_class.end())
				printf("creating model\n");
			else
			{
				printf("rare class -- skipped\n");
				continue;
			}

			vector<double> y(N);
			vector<svm_node*> train;
			vector<double> trainCl;

			for(j=0;j<N;j++) 
				y[j] = -1.0;
			for(j=0;j<(int)class_list[i].size();j++)
				y[class_list[i][j]] = 1.0;

			for(j=0;j<N;j++)
			{
				train.push_back(prob.x[j]);
				trainCl.push_back(y[j]);
			}

			prob.l = (int)train.size();
			prob.x = &train[0];
			prob.y = &trainCl[0];

			if((int)class_list[i].size() == 0)
				printf("WARNING: class %s does not has training data. See README for details.\n", numMap[i].c_str());    

			if(param.do_feature_selection)
				feature_selection(prob, x_space, max_index, elements, selected_feature[i]);

			model = svm_train(&prob,&param); 
			if(model->nr_class == 1) 
				printf("WARNING: class %s has training data in only one class. See README for details.\n", numMap[i].c_str());
			printf("Total nSV: %d\n", model->l);
			fflush(stdout);

			if(do_RSVM && model->l > 0)
				RSVM(prob, model, randomSeed);

			if(param.do_feature_selection)
				copy(original_x_space.begin(), original_x_space.end(), x_space);

			sprintf(model_file_name_temp, "%s.%s", model_file_name, numMap[i].c_str());
			if(svm_save_model(model_file_name_temp,model))
			{
				fprintf(stderr, "can't save model to file %s\n", model_file_name_temp);
				exit(1);
			}

			prob = probTmp;
			svm_free_and_destroy_model(&model);
		}

		//create model file.
		FILE *center_model = fopen(model_file_name, "wb");

		if(center_model == NULL)
		{
			fprintf(stderr,"can't write model to file %s\n", model_file_name);
			exit(1);
		}

		if(param.multiclass_classification)
			fprintf(center_model, "classification_type : multiclass\n");
		else
			fprintf(center_model, "classification_type : multilabel\n");

		fprintf(center_model, "#valid_model : %d\n", (int)(numMap.size()-rare_class.size()));
		fprintf(center_model, "model : ");
		for(i=0;i<(int)classMap.size();i++)
			if(rare_class.find(i) == rare_class.end())
				fprintf(center_model, "%s ", numMap[i].c_str());
		fprintf(center_model, "\n");

		fprintf(center_model, "RSVM              : %d\n", do_RSVM);
		fprintf(center_model, "Feature_selection : %d\n", param.do_feature_selection);
		for(i=0;i<class_no;i++)
			if(rare_class.find(i) == rare_class.end())
			{
				fprintf(center_model, "%d ", (int)selected_feature[i].size());
				map<int, int>::iterator it;
				for(it=selected_feature[i].begin();it!=selected_feature[i].end();it++)
					fprintf(center_model, "%X %X ", it->first, it->second);
				fprintf(center_model, "\n");
			}

		if((int)(numMap.size()-rare_class.size()) == 0)
			printf("WARNING: All classes are considered as rare class. No classifer is created.\n"); 

		fclose(center_model);
	}
	else if(param.hierarchical_classification)
	{
		vector<svm_node> original_x_space;
		svm_problem probTmp = prob;
		int N = prob.l, c, k;
		vector<int> valid_model;
		vector<map<int, int> > selected_feature(class_no);

		#if defined(_WIN32)
			_mkdir(model_folder_name);    
		#else 
			mkdir(model_folder_name, 0777); // notice that 777 is different than 0777
		#endif

		if(param.do_feature_selection)
			original_x_space.assign(x_space, x_space+elements);

		//iterate each class and create classifier
		for(i=0;i<class_no;i++) 
		{
			printf("\nclass %s : ", numMap[i].c_str());
			if(rare_class.find(i) == rare_class.end())
				printf("creating model\n");
			else
			{
				printf("rare class -- skipped\n");
				continue;
			}

			vector<double> y(N);
			vector<svm_node*> train;
			vector<double> trainCl;
			set<int> added_example;

			for(j=0;j<N;j++) 
				y[j] = -1.0;

			//collect superclasses for each class node
			set<int> grandparent, grandgrandparent;
			{
				vector<int>::iterator it, it2, it3;
				for(it=parent[i].begin();it!=parent[i].end();it++)
					for(it2=parent[*it].begin();it2!=parent[*it].end();it2++)
					{
						grandparent.insert(*it2);
						for(it3=parent[*it2].begin();it3!=parent[*it2].end();it3++)
							grandgrandparent.insert(*it3);
					}
			}

			//create training data which consist of all data in parent level, 20% of grand parent level and 10 of grand-grand parent level
			//adding "each example" to training dataset of class i. 

			int grandparent_cnt=-1, grandgrandparent_cnt=-1, other_cnt=-1, posCnt=0;

			for(k=0;k<N;k++)
			{
				set<int> tr_cl_ext;
				for(j=0;j<(int)tr_class[k].size();j++)
				{
					get_ancestor_list(tr_class[k][j], tr_cl_ext);
					tr_cl_ext.insert(tr_class[k][j]);
				}
				if(tr_cl_ext.find(i) != tr_cl_ext.end())
				{
					y[k] = 1.0;
					posCnt++;
					train.push_back(prob.x[k]);
					trainCl.push_back(y[k]);
				}
				else
				{
					vector<int>::iterator it;
					set<int>::iterator it2;
					bool added = false;

					for(it=parent[i].begin();it!=parent[i].end();it++)
						if(tr_cl_ext.find(*it) != tr_cl_ext.end())
						{
							train.push_back(prob.x[k]);
							trainCl.push_back(y[k]);
							added = true;
							break;
						}      

					if(!added)
						for(it2=grandparent.begin();it2!=grandparent.end();it2++)
							if(tr_cl_ext.find(*it2) != tr_cl_ext.end())
							{
								grandparent_cnt++;
								if(grandparent_cnt%100 < 20) //20% of grandparent
								{
									train.push_back(prob.x[k]);
									trainCl.push_back(y[k]);
									added = true;
									break;
								}    
							}

					if(!added)
						for(it2=grandgrandparent.begin();it2!=grandgrandparent.end();it2++)
							if(tr_cl_ext.find(*it2) != tr_cl_ext.end())
							{
								grandgrandparent_cnt++;
								if(grandgrandparent_cnt%100 < 10) //10% of grand-grandparent
								{
									train.push_back(prob.x[k]);
									trainCl.push_back(y[k]);
									added = true;
									break;
								}    
							}
				}
			}

			prob.l = (int)train.size();
			prob.x = &train[0];
			prob.y = &trainCl[0];

			if(param.do_feature_selection)
				feature_selection(prob, x_space, max_index, elements, selected_feature[i]);

			set_kernel(max_index, prob.l, posCnt, param); 

			model = svm_train(&prob,&param); 
			if(model->nr_class == 1) 
				printf("WARNING: class %s has training data in only one class. See README for details.\n", numMap[i].c_str());
			printf("Total nSV: %d\n", model->l);
			fflush(stdout);

			if(do_RSVM && model->l > 0)
				RSVM(prob, model, randomSeed);

			if(param.do_feature_selection)
				copy(original_x_space.begin(), original_x_space.end(), x_space);

			sprintf(model_file_name_temp, "%s.%s", model_file_name, numMap[i].c_str());
			if(svm_save_model(model_file_name_temp,model))
			{
				fprintf(stderr, "can't save model to file %s\n", model_file_name_temp);
				exit(1);
			}

			prob = probTmp;
			valid_model.push_back(i);
			svm_free_and_destroy_model(&model);
		}

		FILE *center_model = fopen(model_file_name, "wb");
		if(center_model == NULL)
		{
				fprintf(stderr, "can't save model to file %s\n", model_file_name);
				exit(1);
		}

		fprintf(center_model, "classification_type : hierarchical\n");

		fprintf(center_model, "#model : %d\n", (int)numMap.size());
		fprintf(center_model, "model : ");
		for(i=0;i<(int)classMap.size();i++)
			fprintf(center_model, "%s ", numMap[i].c_str());
		fprintf(center_model, "\n");

		fprintf(center_model, "#valid_model : %d\n", (int)valid_model.size());
		fprintf(center_model, "model : ");
		for(i=0;i<(int)valid_model.size();i++)
			fprintf(center_model, "%d ", valid_model[i]);
		fprintf(center_model, "\n");

		fprintf(center_model, "RSVM              : %d\n", do_RSVM);
		fprintf(center_model, "Feature_selection : %d\n", param.do_feature_selection);
		for(i=0;i<class_no;i++)
		{
			fprintf(center_model, "%d ", (int)selected_feature[i].size());
			map<int, int>::iterator it;
			for(it=selected_feature[i].begin();it!=selected_feature[i].end();it++)
				fprintf(center_model, "%X %X ", it->first, it->second);
			fprintf(center_model, "\n");
		}

		if((int)(valid_model.size()) == 0)
			printf("WARNING: All classes are considered as rare class. No classifer is created.\n"); 

		fclose(center_model);

		sprintf(model_file_name_temp, "%s.hf", model_file_name);
		FILE *hierarchy_model = fopen(model_file_name_temp, "wb");
		if(hierarchy_model == NULL)
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name_temp);
			exit(1);
		}

		FILE *fp = fopen(hierarchical_classification_path,"r");
		if(fp == NULL)
		{
			fprintf(stderr, "can't open file %s\n", hierarchical_classification_path);
			exit(1);
		}

		char pstr[4096], cstr[4096];
		while(fscanf(fp, "%s", pstr) != EOF)
		{
			fscanf(fp, "%s", cstr);
			int ci = classMap[string(cstr)], pi = classMap[string(pstr)];

			fprintf(hierarchy_model, "%d %d\n", pi, ci);
		}

		fclose(fp);
		fclose(hierarchy_model);
	}

	if(!cross_validation)
	{
		printf("Finished.\n");
		printf("Model folder has been created to %s\n", model_folder_name);
	}

	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);
	return 0;
}

void do_cross_validation_predict_multilabel
		(int foldNum, 
			vector<int> &m, 
			double &accmacro, 
			double &precmicro, 
			double &precmacro, 
			double &reclmicro, 
			double &reclmacro, 
			double &f1micro, 
			double &f1macro, 
			vector<vector<int> > &prediction, 
			vector<map<int, double> > &svm_sc,
			vector<double> &acc, 
			vector<double> &prec, 
			vector<double> &recl, 
			vector<double> &f1,
			double &inductionTime,
			vector<int> &numOfSelectedFeature,
			vector<svm_parameter> &classParam)
{
	//treating first 2N/3 as train and last N/3 as test.
	int N, i, j;
	int sumTP, sumFP, sumFN, sumTN;
	vector<svm_node> original_x_space;
	vector<svm_node*> train, test;
	svm_problem probTmp = prob;
	vector<int> testIdx, trainIdx, cl;
	clock_t start, stop;
	vector<map<int, int> > selected_feature(class_no), selected_feature2(class_no);
	vector<vector<double> > useful_threshold_val(class_no);

	char model_folder_name_tmp[1100], model_file_name_tmp[1100], model_file_name_cl[1100];
	sprintf(model_folder_name_tmp, "%s_fold%d", model_folder_name, foldNum);
	sprintf(model_file_name_tmp, "%s/%s_fold%d", model_folder_name_tmp, model_file_name_without_path, foldNum);
	#if defined(_WIN32)
		_mkdir(model_folder_name_tmp);    
	#else 
		mkdir(model_folder_name_tmp, 0777); // notice that 777 is different than 0777
	#endif

	N = prob.l;
	f1micro=0.0;
	f1macro=0.0;
	inductionTime = 0.0;
	precmacro = reclmacro = accmacro = 0.0;
	sumTP=sumFP=sumFN=sumTN=0;

	start = clock();
	if(param.do_feature_selection)
		original_x_space.assign(x_space, x_space+elements);

	for(j=0;j<N;j++)
		if(m[j] == foldNum)
		{
			test.push_back(prob.x[j]);
			testIdx.push_back(j);
		}
		else
		{
			train.push_back(prob.x[j]);
			trainIdx.push_back(j);
		}

	cl.resize(testIdx.size());
	vector<double> maxp(testIdx.size(), -DBL_MAX);
	stop = clock();
	inductionTime+=(stop-start)/(double)CLOCKS_PER_SEC;
	numOfSelectedFeature.resize(class_no);

	for(i=0;i<class_no;i++) 
	{
		int posCnt = 0;

		start = clock();

		printf("\nclass %s : ", numMap[i].c_str());
		if(rare_class.find(i) == rare_class.end())
			printf("creating model\n");
		else
		{   
			printf("rare class -- skipped\n");
			continue;
		}

		vector<double> y(N);
		vector<double> trainCl, testCl;

		for(j=0;j<N;j++) 
			y[j] = -1.0;
		for(j=0;j<(int)class_list[i].size();j++)
		{
			posCnt++;
			y[class_list[i][j]] = 1.0;
		}

		for(j=0;j<N;j++)
			if(m[j] == foldNum)
			{
				testCl.push_back(y[j]);
			}
			else
			{
				trainCl.push_back(y[j]);
			}

		prob.l = (int)train.size();
		prob.x = &train[0];
		prob.y = &trainCl[0];

		//if label does not has training data, the classifier will say NO (Mr. NO) in all predictions.
		if((int)class_list[i].size() == 0)
			printf("WARNING: label %s does not has training data. See README for details.\n", numMap[i].c_str());   

		if(param.do_feature_selection)
		{
			feature_selection(prob, x_space, max_index, elements, selected_feature[i]);
			numOfSelectedFeature[i] = (int)selected_feature[i].size();
		}
		else
			numOfSelectedFeature[i] = max_index+1;

		svm_copy_param(classParam[i], param);
		
		model = svm_train(&prob,&classParam[i]); 
		if(model->nr_class == 1) 
				printf("WARNING: class %s has training data in only one class. See README for details.\n", numMap[i].c_str());
		printf("Total nSV: %d\n", model->l);
		fflush(stdout);

		classParam[i].do_RSVM = (do_RSVM && model->l > 0)?1:0;
		if(do_RSVM && model->l > 0)
			RSVM(prob, model, randomSeed);

		stop = clock();
		inductionTime+=(stop-start)/(double)CLOCKS_PER_SEC;

		if(param.multilabel_classification)
		{
			int tp, tn, fp, fn;
			tp = tn = fp = fn = 0;

			for(j=0;j<(int)test.size();j++)
			{
				vector<svm_node> data;
				double p = svm_predict(model, test[j]);
				svm_sc[testIdx[j]][i] = p;

				if(p > 0.0)
					prediction[testIdx[j]].push_back(i);  

				if(myabs(testCl[j]-1.0) < epsilon)
					if(p > 0.0)
						tp++;
					else 
						fn++;
				else 
					if(p > 0.0)
						fp++;
					else
						tn++;
			}

			sumTP+=tp;
			sumTN+=tn;
			sumFP+=fp;
			sumFN+=fn;

			double F, PREC, RECL, ACC;
			F = PREC = RECL = ACC = 0.0;
			if(tp != 0 || fp != 0 || fn != 0)
				F = 2*tp/(double)(2*tp+fp+fn);
			if(tp != 0 || fp != 0)
				PREC = tp/(double)(tp+fp);
			if(tp != 0 || fn != 0)
				RECL = tp/(double)(tp+fn);
			if((int)test.size() != 0)
				ACC = 1.0-(fp+fn)/(double)test.size();

			precmacro += PREC;
			reclmacro += RECL;
			accmacro += ACC;
			f1macro+=F;  
			prec[i] = PREC;
			recl[i] = RECL;
			acc[i] = ACC;
			f1[i] = F;
		}
		else //multiclass classification
		{
			for(j=0;j<(int)test.size();j++)
			{
				double p = svm_predict(model, test[j]);
				svm_sc[testIdx[j]][i] = p;

				if(p > maxp[j])
				{
					maxp[j] = p;
					cl[j] = i;
				}    
			}
		}

		start = clock();

		prob = probTmp;
		if(param.do_feature_selection)
			copy(original_x_space.begin(), original_x_space.end(), x_space);

		stop = clock();
		inductionTime+=(stop-start)/(double)CLOCKS_PER_SEC;

		sprintf(model_file_name_cl, "%s.%s", model_file_name_tmp, numMap[i].c_str());
		if(svm_save_model(model_file_name_cl,model))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name_cl);
			exit(1);
		}

		svm_free_and_destroy_model(&model);
	}

	//Writing model file.*****************************************************
	{
		FILE *center_model = fopen(model_file_name_tmp, "wb");

		if(center_model == NULL)
		{
			fprintf(stderr,"can't write model to file %s\n", model_file_name_tmp);
			exit(1);
		}

		if(param.multiclass_classification)
			fprintf(center_model, "classification_type : multiclass\n");
		else
			fprintf(center_model, "classification_type : multilabel\n");

		fprintf(center_model, "#valid_model : %d\n", (int)(numMap.size()-rare_class.size()));
		fprintf(center_model, "model : ");
		for(i=0;i<(int)classMap.size();i++)
			if(rare_class.find(i) == rare_class.end())
				fprintf(center_model, "%s ", numMap[i].c_str());
		fprintf(center_model, "\n");

		fprintf(center_model, "RSVM              : %d\n", do_RSVM);
		fprintf(center_model, "Feature_selection : %d\n", param.do_feature_selection);
		for(i=0;i<class_no;i++)
			if(rare_class.find(i) == rare_class.end())
			{
				fprintf(center_model, "%d ", (int)selected_feature[i].size());
				map<int, int>::iterator it;
				for(it=selected_feature[i].begin();it!=selected_feature[i].end();it++)
					fprintf(center_model, "%X %X ", it->first, it->second);
				fprintf(center_model, "\n");
			}

		fclose(center_model);
	}
	//END Writing model file.*****************************************************

	if(param.multiclass_classification)
	{
		vector<int> tp(class_no, 0), fp(class_no, 0), fn(class_no, 0);
		
		for(j=0;j<(int)testIdx.size();j++)
		{
			int tr_cl = -1;
			int pr_cl = cl[j];

			if(!tr_class[testIdx[j]].empty())
				tr_cl = tr_class[testIdx[j]][0];
			
			prediction[testIdx[j]].push_back(pr_cl);

			if(tr_cl >= 0 && tr_cl == pr_cl)
				tp[tr_cl]++;
			else
			{
				fp[pr_cl]++;
				if(tr_cl >= 0)
					fn[tr_cl]++;
			}
		}

		for(i=0;i<class_no;i++)
			if(rare_class.find(i) == rare_class.end())
			{
				sumTP+=tp[i];
				sumFP+=fp[i];
				sumFN+=fn[i];

				double F, PREC, RECL, ACC;
				F = PREC = RECL = ACC = 0.0;
				if(tp[i] != 0 || fp[i] != 0 || fn[i] != 0)
					F = 2*tp[i]/(double)(2*tp[i]+fp[i]+fn[i]);
				if(tp[i] != 0 || fp[i] != 0)
					PREC = tp[i]/(double)(tp[i]+fp[i]);
				if(tp[i] != 0 || fn[i] != 0)
					RECL = tp[i]/(double)(tp[i]+fn[i]);
				if((int)test.size() != 0)
					ACC = 1.0-(fp[i]+fn[i])/(double)test.size();

				precmacro += PREC;
				reclmacro += RECL;
				accmacro += ACC;
				f1macro+=F;  
				prec[i] = PREC;
				recl[i] = RECL;
				acc[i] = ACC;
				f1[i] = F;
			}
	}

	if((int)(classMap.size()-rare_class.size()) == 0)
		f1macro = precmacro = reclmacro = accmacro = 0.0;
	else
	{
		f1macro/=(double)(classMap.size() - rare_class.size());
		precmacro/=(double)(classMap.size() - rare_class.size());
		reclmacro/=(double)(classMap.size() - rare_class.size());
		accmacro/=(double)(classMap.size() - rare_class.size());
	}

	f1micro = precmicro = reclmicro = 0.0;
	if(sumTP != 0 || sumFP != 0 || sumFN != 0)
		f1micro = 2*sumTP/(double)(2*sumTP+sumFP+sumFN);
	if(sumTP != 0 || sumFP != 0)
		precmicro = sumTP/(double)(sumTP+sumFP);
	if(sumTP != 0 || sumFN != 0)
		reclmicro = sumTP/(double)(sumTP+sumFN);
}

void do_cross_validation_predict_hierarchical
		(int foldNum, 
			vector<int> &m, 
			double &accmacro, 
			double &precmicro, 
			double &precmacro, 
			double &reclmicro, 
			double &reclmacro, 
			double &f1micro, 
			double &f1macro, 
			double &hprecmicro, 
			double &hprecmacro,
			double &hreclmicro, 
			double &hreclmacro,
			double &hf1micro, 
			double &hf1macro, 
			vector<vector<int> > &prediction, 
			vector<map<int, double> > &svm_sc,
			vector<double> &acc, 
			vector<double> &prec, 
			vector<double> &recl, 
			vector<double> &f1,
			double &inductionTime,
			string truetmpfile,
			string predictiontmpfile,
			vector<int> &numOfSelectedFeature,
			vector<svm_parameter> &classParam,
			vector<int> &posPredCnt, vector<int> &negPredCnt)
{
	int N, i, j, k, c;
	int sumTP, sumFP, sumFN, sumTN;
	vector<int> test_data_idx, valid_model;
	vector<svm_model *> model;
	vector<svm_node> original_x_space;
	svm_problem probTmp = prob;
	set<int>::iterator it;
	clock_t start;
	vector<map<int, int> > selected_feature(class_no);

	char model_folder_name_tmp[1100], model_file_name_tmp[1100], model_file_name_cl[1100];
	sprintf(model_folder_name_tmp, "%s_fold%d", model_folder_name, foldNum);
	sprintf(model_file_name_tmp, "%s/%s_fold%d", model_folder_name_tmp, model_file_name_without_path, foldNum);
	#if defined(_WIN32)
		_mkdir(model_folder_name_tmp);    
	#else 
		mkdir(model_folder_name_tmp, 0777); // notice that 777 is different than 0777
	#endif

	inductionTime=0.0;
	start = clock();
	N = prob.l;
	f1micro=0.0;
	f1macro=0.0;
	precmacro = reclmacro = accmacro = 0.0;
	hf1micro=0.0;
	hf1macro=0.0;
	hprecmicro = hprecmacro = hreclmicro = hreclmacro = 0.0;
	model.resize(class_no);
	numOfSelectedFeature.resize(class_no);

	if(param.do_feature_selection)
		original_x_space.assign(x_space, x_space+elements);

	inductionTime+=(clock()-start)/(double)CLOCKS_PER_SEC;

	vector<svm_node*> test;
	//add testing set.
	for(j=0;j<N;j++)
		if(m[j] == foldNum)
		{
			test.push_back(prob.x[j]);
			test_data_idx.push_back(j);
		}   

	for(i=0;i<class_no;i++) 
	{
		start = clock();
		int posCnt = 0;
		model[i] = NULL;

		printf("\nclass %s : ", numMap[i].c_str());
		if(rare_class.find(i) == rare_class.end())
			printf("creating model\n");
		else
		{
			printf("rare class -- skipped\n");
			continue;
		}

		vector<double> y(N);
		vector<svm_node*> train, validation_set;
		vector<int> trainIdx;
		vector<double> trainCl, validation_set_cl;
		set<int> added_example;

		for(j=0;j<N;j++) 
			y[j] = -1.0;

		set<int> grandparent, grandgrandparent;
		{
			vector<int>::iterator it, it2, it3;

			for(it=parent[i].begin();it!=parent[i].end();it++)
				for(it2=parent[*it].begin();it2!=parent[*it].end();it2++)
				{
					grandparent.insert(*it2);

					for(it3=parent[*it2].begin();it3!=parent[*it2].end();it3++)
						grandgrandparent.insert(*it3);
				}
		}

		int parent_cnt=-1, grandparent_cnt=-1, grandgrandparent_cnt=-1, other_cnt=-1;  posCnt=0;
		for(k=0;k<N;k++)
		{
			set<int> tr_cl_ext;
			for(j=0;j<(int)tr_class[k].size();j++)
			{
				get_ancestor_list(tr_class[k][j], tr_cl_ext);
				tr_cl_ext.insert(tr_class[k][j]);
			}
			if(tr_cl_ext.find(i) != tr_cl_ext.end())
			{
				y[k] = 1.0;
				if(m[k] != foldNum)
				{
					posCnt++;
					train.push_back(prob.x[k]);
					trainCl.push_back(y[k]);
					trainIdx.push_back(k);

					validation_set.push_back(prob.x[k]);
					validation_set_cl.push_back(y[k]);
				}
			}
			else if(m[k] != foldNum)
			{
				vector<int>::iterator it;
				set<int>::iterator it2;
				bool added = false;

				for(it=parent[i].begin();it!=parent[i].end();it++)
					if(tr_cl_ext.find(*it) != tr_cl_ext.end())
					{
						parent_cnt++;
						train.push_back(prob.x[k]);
						trainCl.push_back(y[k]);
						trainIdx.push_back(k);

						validation_set.push_back(prob.x[k]);
						validation_set_cl.push_back(y[k]);
						added = true;
						break;
					}      

				if(!added)
					for(it2=grandparent.begin();it2!=grandparent.end();it2++)
						if(tr_cl_ext.find(*it2) != tr_cl_ext.end())
						{
							grandparent_cnt++;
							if(grandparent_cnt%100 < 20)
							{
								train.push_back(prob.x[k]);
								trainCl.push_back(y[k]);
								trainIdx.push_back(k);
								added = true;
								break;
							}    
						}

				if(!added)
					for(it2=grandgrandparent.begin();it2!=grandgrandparent.end();it2++)
						if(tr_cl_ext.find(*it2) != tr_cl_ext.end())
						{
							grandgrandparent_cnt++;
							if(grandgrandparent_cnt%100 < 10)
							{
								train.push_back(prob.x[k]);
								trainCl.push_back(y[k]);
								trainIdx.push_back(k);
								added = true;
								break;
							}    
						}
			}
		}

		if((int)train.size() == 0)
		{
			printf("WARNING: label %s does not has training data. See README for details.\n", numMap[i].c_str());
			//add 1 negative training data to make Mr.NO classifier.
			train.push_back(prob.x[0]);
			trainCl.push_back(-1.0);
		}

		prob.l = (int)train.size();
		prob.x = &train[0];
		prob.y = &trainCl[0];

		if(param.do_feature_selection)
		{
			feature_selection(prob, x_space, max_index, elements, selected_feature[i]);
			numOfSelectedFeature[i] = (int)selected_feature[i].size();
		}
		else
			numOfSelectedFeature[i] = max_index+1;

		int do_RSVM_tmp = do_RSVM;
		set_kernel(numOfSelectedFeature[i], parent_cnt+1, posCnt, param); 
		svm_copy_param(classParam[i], param);

		model[i] = svm_train(&prob,&classParam[i]); 
		if(model[i]->nr_class == 1) 
				printf("WARNING: class %s has training data in only one class. See README for details.\n", numMap[i].c_str());
		
		printf("Total nSV: %d\n", model[i]->l);
		fflush(stdout);

		classParam[i].do_RSVM = (do_RSVM && model[i]->l > 0)?1:0;
		if(do_RSVM && model[i]->l > 0)
			RSVM(prob, model[i], randomSeed);

		do_RSVM = do_RSVM_tmp;

		if(param.do_feature_selection)
			copy(original_x_space.begin(), original_x_space.end(), x_space);

		inductionTime+=(clock()-start)/(double)CLOCKS_PER_SEC;

		//Writing model file.*****************************************************
		sprintf(model_file_name_cl, "%s.%s", model_file_name_tmp, numMap[i].c_str());
		if(svm_save_model(model_file_name_cl,model[i]))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name_cl);
			exit(1);
		}
		//END writing model file.*****************************************************

		valid_model.push_back(i);
		prob = probTmp;
	}

	//Writing Center model file.*****************************************************
	{
		FILE *center_model = fopen(model_file_name_tmp, "wb");
		if(center_model == NULL)
		{
				fprintf(stderr, "can't save model to file %s\n", model_file_name_tmp);
				exit(1);
		}

		fprintf(center_model, "classification_type : hierarchical\n");

		fprintf(center_model, "#model : %d\n", (int)numMap.size());
		fprintf(center_model, "model : ");
		for(i=0;i<(int)classMap.size();i++)
			fprintf(center_model, "%s ", numMap[i].c_str());
		fprintf(center_model, "\n");

		fprintf(center_model, "#valid_model : %d\n", (int)valid_model.size());
		fprintf(center_model, "model : ");
		for(i=0;i<(int)valid_model.size();i++)
			fprintf(center_model, "%d ", valid_model[i]);
		fprintf(center_model, "\n");

		fprintf(center_model, "RSVM              : %d\n", do_RSVM);
		fprintf(center_model, "Feature_selection : %d\n", param.do_feature_selection);
		for(i=0;i<class_no;i++)
		{
			fprintf(center_model, "%d ", (int)selected_feature[i].size());
			map<int, int>::iterator it;
			for(it=selected_feature[i].begin();it!=selected_feature[i].end();it++)
				fprintf(center_model, "%X %X ", it->first, it->second);
			fprintf(center_model, "\n");
		}

		fclose(center_model);

		sprintf(model_file_name_cl, "%s.hf", model_file_name_tmp);
		FILE *hierarchy_model = fopen(model_file_name_cl, "wb");
		if(hierarchy_model == NULL)
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name_cl);
			exit(1);
		}

		FILE *fp = fopen(hierarchical_classification_path,"r");
		if(fp == NULL)
		{
			fprintf(stderr, "can't open file %s\n", hierarchical_classification_path);
			exit(1);
		}

		char pstr[4096], cstr[4096];
		while(fscanf(fp, "%s", pstr) != EOF)
		{
			fscanf(fp, "%s", cstr);
			int ci = classMap[string(cstr)], pi = classMap[string(pstr)];

			fprintf(hierarchy_model, "%d %d\n", pi, ci);
		}

		fclose(fp);
		fclose(hierarchy_model);
	}
	//END Writing Center model file.*****************************************************

	//Measure training/testing error.
	vector<int> tp(class_no, 0), fp(class_no, 0), fn(class_no, 0); 
	vector<int> htp(test.size(), 0), hfp(test.size(), 0), hfn(test.size(), 0); 

	FILE *truetmp = fopen(truetmpfile.c_str(), "wb");
	if(truetmp == NULL)
	{
		fprintf(truetmp, "can't create tmp file %s\n", truetmpfile.c_str());
		exit(1);
	}
	FILE *predtmp = fopen(predictiontmpfile.c_str(), "wb");
	if(predtmp == NULL)
	{
		fprintf(stderr, "can't create tmp file %s\n", predictiontmpfile.c_str());
		exit(1);
	}

	for(i=0;i<(int)test.size();i++)
	{
		set<int> pred_cl;

		int test_idx = test_data_idx[i];
		//variable tr_class contains only leaf. so extend version will contains all node.
		set<int> tr_cl_ext;

		for(j=0;j<(int)tr_class[test_idx].size();j++)
		{
			get_ancestor_list(tr_class[test_idx][j], tr_cl_ext);
			tr_cl_ext.insert(tr_class[test_idx][j]);
		}

		svm_predict_hierarchical_CV(root, model, test[i], selected_feature, child, pred_cl, svm_sc[test_idx], posPredCnt, negPredCnt, tr_cl_ext);
	
		for(it=pred_cl.begin();it!=pred_cl.end();it++)
			prediction[test_idx].push_back(*it);

		//delete rare_class from example.
		for(it=rare_class.begin();it!=rare_class.end();it++)
		{
			tr_cl_ext.erase(*it);
			pred_cl.erase(*it);
		}

		for(it=pred_cl.begin();it!=pred_cl.end();it++)
			if(tr_cl_ext.find(*it) == tr_cl_ext.end())
			{
				fp[*it]++;
				hfp[i]++;
			}
			else
			{
				htp[i]++;
				tp[*it]++;
			}

		for(it=tr_cl_ext.begin();it!=tr_cl_ext.end();it++)
			if(pred_cl.find(*it) == pred_cl.end())
			{
				hfn[i]++;
				fn[*it]++;
			}
		
		for(it=pred_cl.begin();it!=pred_cl.end();it++)
			fprintf(predtmp, "%d ", *it+1);
		for(it=tr_cl_ext.begin();it!=tr_cl_ext.end();it++)
			fprintf(truetmp, "%d ", *it+1);
		fprintf(predtmp, "\n");
		fprintf(truetmp, "\n");
	}
	fclose(truetmp);
	fclose(predtmp);

	//class based micro macro.
	sumTP = sumTN = sumFP = sumFN = 0; 
	for(i=0;i<class_no;i++)
		if(rare_class.find(i) == rare_class.end())
		{
			sumTP+=tp[i];
			sumFP+=fp[i];
			sumFN+=fn[i];

			double F, PREC, RECL, ACC;
			F = PREC = RECL = ACC = 0.0;
			if(tp[i] != 0 || fp[i] != 0 || fn[i] != 0)
				F = 2*tp[i]/(double)(2*tp[i]+fp[i]+fn[i]);
			if(tp[i] != 0 || fp[i] != 0)
				PREC = tp[i]/(double)(tp[i]+fp[i]);
			if(tp[i] != 0 || fn[i] != 0)
				RECL = tp[i]/(double)(tp[i]+fn[i]);
			if((int)test.size() != 0)
				ACC = 1.0-(fp[i]+fn[i])/(double)test.size();

			precmacro += PREC;
			reclmacro += RECL;
			accmacro += ACC;
			f1macro+=F;  
			prec[i] = PREC;
			recl[i] = RECL;
			acc[i] = ACC;
			f1[i] = F;
		}

	if((int)(classMap.size()-rare_class.size()) == 0)
		f1macro = precmacro = reclmacro = accmacro = 0.0;
	else
	{
		f1macro/=(double)(classMap.size() - rare_class.size());
		precmacro/=(double)(classMap.size() - rare_class.size());
		reclmacro/=(double)(classMap.size() - rare_class.size());
		accmacro/=(double)(classMap.size() - rare_class.size());
	}

	f1micro = precmicro = reclmicro = 0.0;
	if(sumTP != 0 || sumFP != 0 || sumFN != 0)
		f1micro = 2*sumTP/(double)(2*sumTP+sumFP+sumFN);
	if(sumTP != 0 || sumFP != 0)
		precmicro = sumTP/(double)(sumTP+sumFP);
	if(sumTP != 0 || sumFN != 0)
		reclmicro = sumTP/(double)(sumTP+sumFN);
	
	//example based micro macro.
	sumTP = sumTN = sumFP = sumFN = 0;
	for(i=0;i<(int)test.size();i++)
	{
		sumTP+=htp[i];
		sumFP+=hfp[i];
		sumFN+=hfn[i];

		double F, PREC, RECL;
		F = PREC = RECL = 0.0;
		if(htp[i] != 0 || hfp[i] != 0 || hfn[i] != 0)
			F = 2*htp[i]/(double)(2*htp[i]+hfp[i]+hfn[i]);
		if(htp[i] != 0 || hfp[i] != 0)
			PREC = htp[i]/(double)(htp[i]+hfp[i]);
		if(htp[i] != 0 || hfn[i] != 0)
			RECL = htp[i]/(double)(htp[i]+hfn[i]);

		hf1macro+=F;
		hprecmacro+=PREC;
		hreclmacro+=RECL;
	}

	if((int)test.size() == 0)
		hf1macro = hprecmacro = hreclmacro = 0.0;
	else
	{
		hf1macro/=(int)test.size();
		hprecmacro/=(int)test.size();
		hreclmacro/=(int)test.size();
	}

	hf1micro = hprecmicro = hreclmicro = 0.0;
	if(sumTP != 0 || sumFP != 0 || sumFN != 0)
		hf1micro = 2*sumTP/(double)(2*sumTP+sumFP+sumFN);
	if(sumTP != 0 || sumFP != 0)
		hprecmicro = sumTP/(double)(sumTP+sumFP);
	if(sumTP != 0 || sumFN != 0)
		hreclmicro = sumTP/(double)(sumTP+sumFN);

	for(i=0;i<(int)model.size();i++)
	{
		if(model[i] != NULL)
			svm_free_and_destroy_model(&model[i]);
	}
}

void do_cross_validation_predict_binary
	(int foldNum, 
		vector<int> &m, 
		double &ACC, 
		double &PREC, 
		double &RECL, 
		double &F,
		double &inductionTime,    
		vector<int> &numOfSelectedFeature,
		vector<vector<int> > &prediction,        
		vector<map<int, double> > &svm_sc)
{
	int N, j;
	vector<svm_node*> train, test;
	vector<svm_node> original_x_space;
	vector<int> test_data_idx;
	vector<double> trainCl, testCl;
	svm_problem probTmp = prob;
	clock_t start = clock();
	map<int, int> selected_feature;

	char model_folder_name_tmp[1100], model_file_name_tmp[1100], model_file_name_cl[1100];
	sprintf(model_folder_name_tmp, "%s_fold%d", model_folder_name, foldNum);
	sprintf(model_file_name_tmp, "%s/%s_fold%d", model_folder_name_tmp, model_file_name_without_path, foldNum);
	#if defined(_WIN32)
		_mkdir(model_folder_name_tmp);    
	#else 
		mkdir(model_folder_name_tmp, 0777); // notice that 777 is different than 0777
	#endif
	
	numOfSelectedFeature.resize(1);
	N = prob.l;
	if(param.do_feature_selection)
		original_x_space.assign(x_space, x_space+elements);

	for(j=0;j<N;j++)
		if(m[j] == foldNum)
		{
			test.push_back(prob.x[j]);
			testCl.push_back(prob.y[j]);
			test_data_idx.push_back(j);
		}
		else
		{
			train.push_back(prob.x[j]);
			trainCl.push_back(prob.y[j]);
		}

	prob.l = (int)train.size();
	prob.x = &train[0];
	prob.y = &trainCl[0];

	if(param.do_feature_selection)
	{
		feature_selection(prob, x_space, max_index, elements, selected_feature);
		numOfSelectedFeature[0] = (int)selected_feature.size();
	}
	else
		numOfSelectedFeature[0] = max_index+1;

	model = svm_train(&prob,&param);

	if(model->nr_class == 1) 
		printf("WARNING: training data in only one class. See README for details.\n");
	printf("Total nSV: %d\n", model->l);
	fflush(stdout);

	if(do_RSVM && model->l > 0)
		RSVM(prob, model, randomSeed);

	inductionTime=(clock()-start)/(double)CLOCKS_PER_SEC;

	//Writing model file.*****************************************************
	{
		sprintf(model_file_name_cl, "%s.%s", model_file_name_tmp, numMap[1].c_str());
		if(svm_save_model(model_file_name_cl,model))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name_cl);
			exit(1);
		}

		FILE *center_model = fopen(model_file_name_tmp, "wb");
		fprintf(center_model, "classification_type : binary\n");
		fprintf(center_model, "model : %s %s\n", numMap[1].c_str(), numMap[-1].c_str());

		fprintf(center_model, "RSVM              : %d\n", do_RSVM);
		fprintf(center_model, "Feature_selection : %d\n", param.do_feature_selection);
		fprintf(center_model, "%d ", (int)selected_feature.size());
		map<int, int>::iterator it;
		for(it=selected_feature.begin();it!=selected_feature.end();it++)
			fprintf(center_model, "%X %X ", it->first, it->second);
		fprintf(center_model, "\n");

		fclose(center_model);
	}
	//END Writing model file.*****************************************************

	int tp, tn, fp, fn;
	tp = tn = fp = fn = 0;
	for(j=0;j<(int)test.size();j++)
	{
		double p = svm_predict(model, test[j]);
		svm_sc[test_data_idx[j]][0] = p;
		prediction[test_data_idx[j]].push_back(p>0.0?1:-1);

		if(myabs(testCl[j]-1.0) < epsilon)
			if(p > 0.0)
				tp++;
			else 
				fn++;
		else 
			if(p > 0.0)
				fp++;
			else
				tn++;
	}

	F = PREC = RECL = ACC = 0.0;
	if(tp != 0 || fp != 0 || fn != 0)
		F = 2*tp/(double)(2*tp+fp+fn);
	if(tp != 0 || fp != 0)
		PREC = tp/(double)(tp+fp);
	if(tp != 0 || fn != 0)
		RECL = tp/(double)(tp+fn);
	if((int)test.size() != 0)
		ACC = 1.0-(fp+fn)/(double)test.size();

	prob = probTmp;
	if(param.do_feature_selection)
		copy(original_x_space.begin(), original_x_space.end(), x_space);
	
	svm_free_and_destroy_model(&model);
}

double getSD(const vector<int> &weight, const vector<double> &data, double mean)
{
	int i, sumw=0, nonZeroWeightCnt=0, N = (int)data.size();

	if(N <= 1)
		return 0.0;

	for(i=0;i<N;i++)
		if(weight[i] != 0)
		{
			sumw+=weight[i];
			nonZeroWeightCnt++;
		}

	double sum=0.0;
	for(i=0;i<N;i++)
		sum+=weight[i]*(data[i]-mean)*(data[i]-mean);

	return sqrt(sum/((nonZeroWeightCnt-1)*sumw/nonZeroWeightCnt));
}

void do_cross_validation(char *input_file_name)
{
	clock_t start, stop;
	const int K = nr_fold;  //K = number of fold
	int i, j, k, N = prob.l;
	vector<int> m(N);
	//measure of each fold.
	vector<double> f1micro(K), f1macro(K), precmicro(K), precmacro(K), reclmicro(K), reclmacro(K), accmacro(K), hprecmicro(K), hprecmacro(K), hreclmicro(K), hreclmacro(K), hf1micro(K), hf1macro(K);
	vector<double> ElbF1micro(K, 0.0), ElbF1macro(K, 0.0), Elbprecmicro(K, 0.0), Elbprecmacro(K, 0.0), Elbreclmicro(K, 0.0), Elbreclmacro(K, 0.0), LCA_F(K, 0.0), LCA_P(K, 0.0), LCA_R(K, 0.0), MGIA(K, 0.0);
	double sumaccmacro, sumf1micro, sumf1macro, sumprecmicro, sumprecmacro, sumreclmicro, sumreclmacro;
	double sumhprecmicro, sumhprecmacro;
	double sumhreclmicro, sumhreclmacro;
	double sumhf1micro, sumhf1macro;
	double sumElbF1micro, sumElbF1macro, sumElbprecmicro, sumElbprecmacro, sumElbreclmicro, sumElbreclmacro;
	double sumLCA_F, sumLCA_P, sumLCA_R, sumMGIA;
	//prediction[i] contains predicted class of example i;
	vector<vector<int> > prediction(N);
	vector<map<int, double> > svm_sc(N);
	//numOfSelectedFeature[i][j] is #selected feature in fold i, feature j 
	vector<vector<int> > numOfSelectedFeature(K);
	vector<int>::iterator it;
	//acc, prec, recl, f1 of each class.
	vector<double> sumf1(class_no, 0.0), sumacc(class_no, 0.0), sumprec(class_no, 0.0), sumrecl(class_no, 0.0);
	vector<double> inductionTime(K), testTime(K);
	vector<int> numberOfgrI(nr_fold);
	double sumInductionTime, sumTestTime;
	vector<svm_parameter> classParam(class_no);

	//hierarchical classification only.
	//posPredCnt[i] = number of examples that belongs to class i and has been go to classifier class i.
	//negPredCnt[i] = number of examples that does not belongs to class i and has been go to classifier class i.
	//posPredCnt and negPredCnt used to measures propagation error and blocking problem.
	vector<int> negPredCnt, posPredCnt;

	for(i=0;i<N;i++)
		m[i] = i%K;

	if(randomSeed >= 0)
		srand(randomSeed);
	for(i=0;i<N;i++)
		swap(m[rand()%N], m[rand()%N]);
	
	sumaccmacro = 0.0;
	sumInductionTime = sumTestTime = 0.0;
	sumf1micro = sumf1macro = 0.0;
	sumprecmicro = sumprecmacro = 0.0;
	sumreclmicro = sumreclmacro = 0.0;
	sumhf1micro = sumhf1macro = 0.0;
	sumElbF1micro = sumElbF1macro = 0.0;
	sumElbprecmicro = sumElbprecmacro = 0.0;
	sumElbreclmicro = sumElbreclmacro = 0.0;
	sumhprecmacro = sumhprecmicro = 0.0;
	sumhreclmicro = sumhreclmacro = 0.0;
	sumLCA_F = sumLCA_P = sumLCA_R = sumMGIA = 0.0;

	string reportfilename = string(input_file_name)+".report";
	string predictionfilename = string(input_file_name)+".prediction";
	string csvfilename = string(input_file_name)+".csv";

	int duplicate_file_index=0;
	if(FileExist(reportfilename) || FileExist(predictionfilename) || FileExist(csvfilename))
	{
		char tmp[100];
		string reportfilenametmp;
		string predictionfilenametmp;
		string csvfilenametmp;
		string strinputfilename = string(input_file_name);
		do
		{
			duplicate_file_index++;
			sprintf(tmp, "_%d", duplicate_file_index);
			reportfilenametmp = strinputfilename + tmp + ".report";
			predictionfilenametmp = strinputfilename + tmp + ".prediction"; 
			csvfilenametmp = strinputfilename+tmp+".csv";
		}
		while(FileExist(reportfilenametmp) || FileExist(predictionfilenametmp) || FileExist(csvfilenametmp));
		reportfilename = reportfilenametmp;
		predictionfilename = predictionfilenametmp;
		csvfilename = csvfilenametmp;
	}

	FILE *out = fopen(reportfilename.c_str(), "w");
	if(out == NULL)
	{
		fprintf(stderr, "error, cannot create file %s\n", reportfilename.c_str());
		exit(1);
	}

	fprintf(out, "Training data     : %s\n", input_file_name);
	if(param.hierarchical_classification)
		fprintf(out, "Hierarchical data : %s\n", hierarchical_classification_path);
	else
	{
		fprintf(out, "Kernel type       : %s\n", kernel_type_table[param.kernel_type]);
		if(param.kernel_type == POLY)
			fprintf(out, "Degree            : %d\n", param.degree);
		if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
			fprintf(out, "Gamma             : %lg\n", param.gamma);
		if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
			fprintf(out, "Coef0             : %lg\n", param.coef0);
	}
	fprintf(out, "Feature selection : %s\n", param.do_feature_selection?"on":"off");
	fprintf(out, "RSVM              : %s\n", do_RSVM?"on":"off");
	fprintf(out, "#Class            : %d\n", (int)numMap.size());
	fprintf(out, "#Feature          : %d\n", max_index+1);
	fprintf(out, "#Example          : %d\n", N);
	if(!param.binary_classification)
		fprintf(out, "#Rare class       : %d\n", (int)rare_class.size());
	if(param.hierarchical_classification)
		fprintf(out, "Max depth         : %d\n", max_depth);
	fprintf(out, "\n");

	if(param.binary_classification)
	{
		for(i=0;i<K;i++, fflush(stdout), fflush(out))
		{
			printf("\nFOLD %d\n", i+1);

			start = clock();
			do_cross_validation_predict_binary(i, m, 
												accmacro[i], precmacro[i], reclmacro[i], f1macro[i], 
												inductionTime[i], 
												numOfSelectedFeature[i],
												prediction, svm_sc);
			stop = clock();
			testTime[i] = (stop-start)/(double)CLOCKS_PER_SEC - inductionTime[i];

			numberOfgrI[i] = N/K + (i<N%K?1:0);
			sumf1[0]+=f1macro[i]*numberOfgrI[i];
			sumacc[0]+=accmacro[i]*numberOfgrI[i];
			sumrecl[0]+=reclmacro[i]*numberOfgrI[i];
			sumprec[0]+=precmacro[i]*numberOfgrI[i];
			sumInductionTime+=inductionTime[i]*numberOfgrI[i];
			sumTestTime+=testTime[i]*numberOfgrI[i];

			putchar('\n');
		}
	}
	else if(param.multilabel_classification || param.multiclass_classification)
	{
		vector<double> f1(class_no), acc(class_no), prec(class_no), recl(class_no);

		for(i=0;i<K;i++, fflush(stdout), fflush(out))
		{ 
			printf("\nFOLD %d\n", i+1);

			char tmp[100];
			if(duplicate_file_index == 0)
				sprintf(tmp, "_train_fold%d", i);  
			else
				sprintf(tmp, "_%d_train_fold%d", duplicate_file_index, i);  
			start = clock();

			do_cross_validation_predict_multilabel(i, m, accmacro[i], 
													precmicro[i], precmacro[i], 
													reclmicro[i], reclmacro[i], 
													f1micro[i], f1macro[i], 
													prediction, svm_sc, 
													acc, prec, recl, f1,
													inductionTime[i],
													numOfSelectedFeature[i],
													classParam);
			stop = clock();

			testTime[i] = (stop-start)/(double)CLOCKS_PER_SEC - inductionTime[i];
			
			numberOfgrI[i] = N/K + (i<N%K?1:0);
			sumf1micro+=f1micro[i]*numberOfgrI[i];
			sumf1macro+=f1macro[i]*numberOfgrI[i];
			sumaccmacro+=accmacro[i]*numberOfgrI[i];
			sumreclmicro+=reclmicro[i]*numberOfgrI[i];
			sumreclmacro+=reclmacro[i]*numberOfgrI[i];
			sumprecmicro+=precmicro[i]*numberOfgrI[i];
			sumprecmacro+=precmacro[i]*numberOfgrI[i];
			sumInductionTime+=inductionTime[i]*numberOfgrI[i];
			sumTestTime+=testTime[i]*numberOfgrI[i];

			for(j=0;j<class_no;j++)
			{
				sumf1[j]+=f1[j]*numberOfgrI[i];
				sumacc[j]+=acc[j]*numberOfgrI[i];
				sumrecl[j]+=recl[j]*numberOfgrI[i];
				sumprec[j]+=prec[j]*numberOfgrI[i];
			}

			putchar('\n');
		}
	}
	else //param.hierarchical_classification
	{
		vector<double> prec(class_no), acc(class_no), recl(class_no), f1(class_no);
		posPredCnt.clear();
		negPredCnt.clear();
		posPredCnt.assign(class_no, 0);
		negPredCnt.assign(class_no, 0);
		char duplicate_file_index_str[100];
		sprintf(duplicate_file_index_str, "%d", duplicate_file_index);  

		//create hierarchy tmp file for testing.
		string htmpFilename = string(input_file_name)+"_"+duplicate_file_index_str+".hftmp";
		FILE *htmpfile = fopen(htmpFilename.c_str(), "wb");
		if(htmpfile == NULL)
		{  
			fprintf(htmpfile, "can't create tmp file %s\n", (string(input_file_name)+"_"+duplicate_file_index_str+".hftmp").c_str());
			exit(1);
		}
		for(i=0;i<class_no;i++)
			for(it=child[i].begin();it!=child[i].end();it++)
				fprintf(htmpfile, "%d %d\n", i, *it);
		fclose(htmpfile);

		for(i=0;i<K;i++, fflush(stdout), fflush(out))
		{ 
			printf("\nFOLD %d\n", i+1);
			//fprintf(out, "FOLD %d\n", i+1);
			char tmp[100];
			if(duplicate_file_index == 0)
				sprintf(tmp, "_train_fold%d", i);  
			else
				sprintf(tmp, "_%d_train_fold%d", duplicate_file_index, i);  

			start = clock();
			string truetmpFilename = string(input_file_name)+"_"+duplicate_file_index_str+".truetmp";
			string predtmpFilename = string(input_file_name)+"_"+duplicate_file_index_str+".predtmp";

			do_cross_validation_predict_hierarchical(i, m, accmacro[i], 
													precmicro[i], precmacro[i], 
													reclmicro[i], reclmacro[i], 
													f1micro[i], f1macro[i],
													hprecmicro[i], hprecmacro[i],
													hreclmicro[i], hreclmacro[i],
													hf1micro[i], hf1macro[i], 
													prediction, svm_sc, acc, prec, recl, f1, 
													inductionTime[i],
													truetmpFilename, 
													predtmpFilename,
													numOfSelectedFeature[i],
													classParam,
													posPredCnt, negPredCnt);

			LCA_F[i] = LCA_P[i]= LCA_R[i] = MGIA[i] = 0.0;

			//hierarchy_measure((string(input_file_name)+".hftmp").c_str(), string(input_file_name)+".truetmp", string(input_file_name)+".predtmp", LCA_F[i], LCA_P[i], LCA_R[i], MGIA[i]);
			printf("collecting scores...\n");
			stringstream output;
			#ifdef OS_WINDOWS
				output << exec((string("HEMKit ")+"\""+htmpFilename+"\" \""+truetmpFilename+"\" \""+predtmpFilename+"\"").c_str());
			#else
				output << exec((string("./HEMKit ")+"\""+htmpFilename+"\" \""+truetmpFilename+"\" \""+predtmpFilename+"\"").c_str());
			#endif
			output >> LCA_F[i] >> LCA_P[i] >> LCA_R[i] >> MGIA[i];
			
			fflush(stdout);

			remove(htmpFilename.c_str());
			remove(truetmpFilename.c_str());
			remove(predtmpFilename.c_str());

			stop = clock();
			testTime[i] = (stop-start)/(double)CLOCKS_PER_SEC - inductionTime[i];

			numberOfgrI[i] = N/K + (i<N%K?1:0);
			sumf1micro+=f1micro[i]*numberOfgrI[i];
			sumf1macro+=f1macro[i]*numberOfgrI[i];
			sumaccmacro+=accmacro[i]*numberOfgrI[i];
			sumreclmicro+=reclmicro[i]*numberOfgrI[i];
			sumreclmacro+=reclmacro[i]*numberOfgrI[i];
			sumprecmicro+=precmicro[i]*numberOfgrI[i];
			sumprecmacro+=precmacro[i]*numberOfgrI[i];
			sumInductionTime+=inductionTime[i]*numberOfgrI[i];
			sumTestTime+=testTime[i]*numberOfgrI[i];

			for(j=0;j<class_no;j++)
			{
				sumf1[j]+=f1[j]*numberOfgrI[i];
				sumacc[j]+=acc[j]*numberOfgrI[i];
				sumrecl[j]+=recl[j]*numberOfgrI[i];
				sumprec[j]+=prec[j]*numberOfgrI[i];
			}

			if(hprecmicro[i] + precmicro[i] > epsilon)
				Elbprecmicro[i] = 2*hprecmicro[i]*precmicro[i]/(hprecmicro[i]+precmicro[i]);
			if(hprecmacro[i] + precmacro[i] > epsilon)
				Elbprecmacro[i] = 2*hprecmacro[i]*precmacro[i]/(hprecmacro[i]+precmacro[i]);
			if(hreclmicro[i] + reclmicro[i] > epsilon)
				Elbreclmicro[i] = 2*hreclmicro[i]*reclmicro[i]/(hreclmicro[i]+reclmicro[i]);
			if(hreclmacro[i] + reclmacro[i] > epsilon)
				Elbreclmacro[i] = 2*hreclmacro[i]*reclmacro[i]/(hreclmacro[i]+reclmacro[i]);
			if(hf1micro[i] + f1micro[i] > epsilon)
				ElbF1micro[i] = 2*hf1micro[i]*f1micro[i]/(hf1micro[i]+f1micro[i]);
			if(hf1macro[i] + f1macro[i] > epsilon)
				ElbF1macro[i] = 2*hf1macro[i]*f1macro[i]/(hf1macro[i]+f1macro[i]);

			sumhprecmicro+=hprecmicro[i]*numberOfgrI[i];
			sumhprecmacro+=hprecmacro[i]*numberOfgrI[i];
			sumhreclmicro+=hreclmicro[i]*numberOfgrI[i];
			sumhreclmacro+=hreclmacro[i]*numberOfgrI[i];
			sumhf1micro+=hf1micro[i]*numberOfgrI[i];
			sumhf1macro+=hf1macro[i]*numberOfgrI[i];

			sumElbprecmicro+=Elbprecmicro[i]*numberOfgrI[i];
			sumElbprecmacro+=Elbprecmacro[i]*numberOfgrI[i];
			sumElbreclmicro+=Elbreclmicro[i]*numberOfgrI[i];
			sumElbreclmacro+=Elbreclmacro[i]*numberOfgrI[i];
			sumElbF1micro+=ElbF1micro[i]*numberOfgrI[i];
			sumElbF1macro+=ElbF1macro[i]*numberOfgrI[i];

			sumLCA_F+=LCA_F[i]*numberOfgrI[i];
			sumLCA_P+=LCA_P[i]*numberOfgrI[i];
			sumLCA_R+=LCA_R[i]*numberOfgrI[i];
			sumMGIA+=MGIA[i]*numberOfgrI[i];

			putchar('\n');
		}
	}
	
	if(param.binary_classification)
	{
		fprintf(out, "Summary(Average value of all folds)\n");
		fprintf(out, "Accuracy        : %lg (SD %lg)\n", sumacc[0]/N, getSD(numberOfgrI, accmacro, sumacc[0]/N));
		fprintf(out, "Precision       : %lg (SD %lg)\n", sumprec[0]/N, getSD(numberOfgrI, precmacro, sumprec[0]/N));
		fprintf(out, "Recall          : %lg (SD %lg)\n", sumrecl[0]/N, getSD(numberOfgrI, reclmacro, sumrecl[0]/N));
		fprintf(out, "F1Score         : %lg (SD %lg)\n", sumf1[0]/N, getSD(numberOfgrI, f1macro, sumf1[0]/N));  
		fprintf(out, "Induction time  : %lg sec. (SD %lg)\n", sumInductionTime/N, getSD(numberOfgrI, inductionTime, sumInductionTime/N));
		fprintf(out, "Test time       : %lg sec. (SD %lg)\n", sumTestTime/N, getSD(numberOfgrI, testTime, sumTestTime/N));
		fprintf(out, "\n\n");  

		for(i=0;i<K;i++)
		{
			fprintf(out, "FOLD %d\n", i+1);   
			fprintf(out, "#Removed features : %d\n", max_index+1-numOfSelectedFeature[i][0]);
			fprintf(out, "Accuracy       : %lg\n", accmacro[i]);
			fprintf(out, "Precision      : %lg\n", precmacro[i]);
			fprintf(out, "Recall         : %lg\n", reclmacro[i]);
			fprintf(out, "F1Score        : %lg\n", f1macro[i]);
			fprintf(out, "Induction time : %lg sec.\n", inductionTime[i]);
			fprintf(out, "Test time      : %lg sec.\n", testTime[i]);
			fprintf(out, "\n");  
		}
	}
	else 
	{
		if(param.multilabel_classification || param.multiclass_classification)
		{
			fprintf(out, "Summary(Average value of all folds)\n");
			fprintf(out, "Accuracy        : %lg (SD %lg)\n", sumaccmacro/N, getSD(numberOfgrI, accmacro, sumaccmacro/N));
			fprintf(out, "Micro Precision : %lg (SD %lg)\n", sumprecmicro/N, getSD(numberOfgrI, precmicro, sumprecmicro/N));
			fprintf(out, "Macro Precision : %lg (SD %lg)\n", sumprecmacro/N, getSD(numberOfgrI, precmacro, sumprecmacro/N));
			fprintf(out, "Micro Recall    : %lg (SD %lg)\n", sumreclmicro/N, getSD(numberOfgrI, reclmicro, sumreclmicro/N));
			fprintf(out, "Macro Recall    : %lg (SD %lg)\n", sumreclmacro/N, getSD(numberOfgrI, reclmacro, sumreclmacro/N));
			fprintf(out, "Micro F1Score   : %lg (SD %lg)\n", sumf1micro/N, getSD(numberOfgrI, f1micro, sumf1micro/N));
			fprintf(out, "Macro F1Score   : %lg (SD %lg)\n", sumf1macro/N, getSD(numberOfgrI, f1macro, sumf1macro/N));
			fprintf(out, "Induction time  : %lg sec. (SD %lg)\n", sumInductionTime/N, getSD(numberOfgrI, inductionTime, sumInductionTime/N));
			fprintf(out, "Test time       : %lg sec. (SD %lg)\n", sumTestTime/N, getSD(numberOfgrI, testTime, sumTestTime/N));
			fprintf(out, "\n\n");  

			for(i=0;i<K;i++)
			{ 
				fprintf(out, "FOLD %d\n", i+1);
				fprintf(out, "Accuracy         : %lg\n", accmacro[i]);
				fprintf(out, "Micro Precision  : %lg\n", precmicro[i]);
				fprintf(out, "Macro Precision  : %lg\n", precmacro[i]);
				fprintf(out, "Micro Recall     : %lg\n", reclmicro[i]);
				fprintf(out, "Macro Recall     : %lg\n", reclmacro[i]);
				fprintf(out, "Micro F1Score    : %lg\n", f1micro[i]);
				fprintf(out, "Macro F1Score    : %lg\n", f1macro[i]);
				fprintf(out, "Induction time   : %lg sec.\n", inductionTime[i]);
				fprintf(out, "Test time        : %lg sec.\n", testTime[i]);
				fprintf(out, "\n");  
			}
		}
		else
		{
			fprintf(out, "Summary(Average value of all folds)\n");
			fprintf(out, "Accuracy           : %lg (SD %lg)\n", sumaccmacro/N, getSD(numberOfgrI, accmacro, sumaccmacro/N));
			fprintf(out, "Accuracy(MGIA)     : %lg (SD %lg)\n", sumMGIA/N, getSD(numberOfgrI, MGIA, sumMGIA/N));
			fprintf(out, "Micro label-based Precision  : %lg (SD %lg)\n", sumprecmicro/N, getSD(numberOfgrI, precmicro, sumprecmicro/N));
			fprintf(out, "Macro label-based Precision  : %lg (SD %lg)\n", sumprecmacro/N, getSD(numberOfgrI, precmacro, sumprecmacro/N));
			fprintf(out, "Micro label-based Recall     : %lg (SD %lg)\n", sumreclmicro/N, getSD(numberOfgrI, reclmicro, sumreclmicro/N)); 
			fprintf(out, "Macro label-based Recall     : %lg (SD %lg)\n", sumreclmacro/N, getSD(numberOfgrI, reclmacro, sumreclmacro/N));
			fprintf(out, "Micro label-based F1Score    : %lg (SD %lg)\n", sumf1micro/N, getSD(numberOfgrI, f1micro, sumf1micro/N));
			fprintf(out, "Macro label-based F1Score    : %lg (SD %lg)\n", sumf1macro/N, getSD(numberOfgrI, f1macro, sumf1macro/N));
			fprintf(out, "Micro example-based Precision: %lg (SD %lg)\n", sumhprecmicro/N, getSD(numberOfgrI, hprecmicro, sumhprecmicro/N));
			fprintf(out, "Macro example-based Precision: %lg (SD %lg)\n", sumhprecmacro/N, getSD(numberOfgrI, hprecmacro, sumhprecmacro/N));
			fprintf(out, "Micro example-based Recall   : %lg (SD %lg)\n", sumhreclmicro/N, getSD(numberOfgrI, hreclmicro, sumhreclmicro/N));
			fprintf(out, "Macro example-based Recall   : %lg (SD %lg)\n", sumhreclmacro/N, getSD(numberOfgrI, hreclmacro, sumhreclmacro/N));
			fprintf(out, "Micro example-based F1Score  : %lg (SD %lg)\n", sumhf1micro/N, getSD(numberOfgrI, hf1micro, sumhf1micro/N));
			fprintf(out, "Macro example-based F1Score  : %lg (SD %lg)\n", sumhf1macro/N, getSD(numberOfgrI, hf1macro, sumhf1macro/N));
			fprintf(out, "Micro example-label-based Precision : %lg (SD %lg)\n", sumElbprecmicro/N, getSD(numberOfgrI, Elbprecmicro, sumElbprecmicro/N));
			fprintf(out, "Macro example-label-based Precision : %lg (SD %lg)\n", sumElbprecmacro/N, getSD(numberOfgrI, Elbprecmacro, sumElbprecmacro/N));
			fprintf(out, "Micro example-label-based Recall    : %lg (SD %lg)\n", sumElbreclmicro/N, getSD(numberOfgrI, Elbreclmicro, sumElbreclmicro/N));
			fprintf(out, "Macro example-label-based Recall    : %lg (SD %lg)\n", sumElbreclmacro/N, getSD(numberOfgrI, Elbreclmacro, sumElbreclmacro/N));
			fprintf(out, "Micro example-label-based F1Score   : %lg (SD %lg)\n", sumElbF1micro/N, getSD(numberOfgrI, ElbF1micro, sumElbF1micro/N)); 
			fprintf(out, "Macro example-label-based F1Score   : %lg (SD %lg)\n", sumElbF1macro/N, getSD(numberOfgrI, ElbF1macro, sumElbF1macro/N)); 
			fprintf(out, "LCA Precision      : %lg (SD %lg)\n", sumLCA_P/N, getSD(numberOfgrI, LCA_P, sumLCA_P/N));
			fprintf(out, "LCA Recall         : %lg (SD %lg)\n", sumLCA_R/N, getSD(numberOfgrI, LCA_R, sumLCA_R/N));
			fprintf(out, "LCA F1Score        : %lg (SD %lg)\n", sumLCA_F/N, getSD(numberOfgrI, LCA_F, sumLCA_F/N));
			fprintf(out, "Induction time     : %lg sec. (SD %lg)\n", sumInductionTime/N, getSD(numberOfgrI, inductionTime, sumInductionTime/N));
			fprintf(out, "Test time          : %lg sec. (SD %lg)\n", sumTestTime/N, getSD(numberOfgrI, testTime, sumTestTime/N));
			fprintf(out, "\n");  

			for(i=0;i<K;i++, fflush(stdout), fflush(out))
			{ 
				fprintf(out, "FOLD %d\n", i+1);
				fprintf(out, "Accuracy           : %lg\n", accmacro[i]);
				fprintf(out, "Accuracy(MGIA)     : %lg\n", MGIA[i]);
				fprintf(out, "Micro label-based Precision  : %lg\n", precmicro[i]);
				fprintf(out, "Macro label-based Precision  : %lg\n", precmacro[i]);
				fprintf(out, "Micro label-based Recall     : %lg\n", reclmicro[i]);
				fprintf(out, "Macro label-based Recall     : %lg\n", reclmacro[i]);
				fprintf(out, "Micro label-based F1Score    : %lg\n", f1micro[i]);
				fprintf(out, "Macro label-based F1Score    : %lg\n", f1macro[i]);
				fprintf(out, "Micro example-based Precision: %lg\n", hprecmicro[i]);
				fprintf(out, "Macro example-based Precision: %lg\n", hprecmacro[i]);
				fprintf(out, "Micro example-based Recall   : %lg\n", hreclmicro[i]);
				fprintf(out, "Macro example-based Recall   : %lg\n", hreclmacro[i]);
				fprintf(out, "Micro example-based F1Score  : %lg\n", hf1micro[i]);
				fprintf(out, "Macro example-based F1Score  : %lg\n", hf1macro[i]);
				fprintf(out, "Micro example-label-based Precision : %lg\n", Elbprecmicro[i]);
				fprintf(out, "Macro example-label-based Precision : %lg\n", Elbprecmacro[i]);
				fprintf(out, "Micro example-label-based Recall    : %lg\n", Elbreclmicro[i]);
				fprintf(out, "Macro example-label-based Recall    : %lg\n", Elbreclmacro[i]);
				fprintf(out, "Micro example-label-based F1Score   : %lg\n", ElbF1micro[i]);
				fprintf(out, "Macro example-label-based F1Score   : %lg\n", ElbF1macro[i]);
				fprintf(out, "LCA Precision      : %lg\n", LCA_P[i]);
				fprintf(out, "LCA Recall         : %lg\n", LCA_R[i]);
				fprintf(out, "LCA F1Score        : %lg\n", LCA_F[i]);
				fprintf(out, "Induction time     : %lg sec.\n", inductionTime[i]);
				fprintf(out, "Test time          : %lg sec.\n", testTime[i]);
				fprintf(out, "\n");  
			}
		}

		fprintf(out, "\n");  
		fprintf(out, "Measures of each classes\n");
		vector<int> valid_model(class_no);
		for(j=0;j<class_no;j++)
			valid_model[j] = j;
		sort(valid_model.begin(), valid_model.end(), valid_model_cmp);
				
		for(it=valid_model.begin();it!=valid_model.end();it++)
			if(rare_class.find(*it) == rare_class.end())
			{
				j = *it;
				fprintf(out, "Class %s\n", numMap[j].c_str());
				if(param.hierarchical_classification)
				{
					set<int>::iterator it2;
					fprintf(out, "level          : "); 
					if(!depth[j].empty())
					{
						it2=depth[j].begin();
						fprintf(out, "%d", *it2);
						it2++;
						for(;it2!=depth[j].end();it2++)
							fprintf(out, ",%d", *it2);
					}
					fprintf(out, "\n");
				}
				
				fprintf(out, "#Examples      : %d\n", class_count[j]);
				if(param.hierarchical_classification)
				{
					vector<int>::iterator it;
					int parentCnt=0;

					for(it=parent[j].begin();it!=parent[j].end();it++)
						parentCnt+=class_count[*it];
					fprintf(out, "#Parent Examples      : %d\n", parentCnt);

					fprintf(out, "#Truly Positive Examples (Entered to this classifier) : %d\n", posPredCnt[j]);
					fprintf(out, "#Truly Negative Examples (Entered to this classifier) : %d\n", negPredCnt[j]);
				}

				fprintf(out, "C              : %lg\n", classParam[j].C);    
				fprintf(out, "Kernel type    : %s\n", kernel_type_table[classParam[j].kernel_type]);
				if(classParam[j].kernel_type == 0)
					fprintf(out, "Gamma          : %lg\n", classParam[j].gamma);
				fprintf(out, "#Removed features (each fold) : %d", max_index+1-numOfSelectedFeature[0][j]);
				for(k=1;k<K;k++)
					fprintf(out, ",%d", max_index+1-numOfSelectedFeature[k][j]);
				fprintf(out, "\n");
				fprintf(out, "Accuracy       : %lg\n", sumacc[j]/N);
				fprintf(out, "Precision      : %lg\n", sumprec[j]/N);
				fprintf(out, "Recall         : %lg\n", sumrecl[j]/N);
				fprintf(out, "F1Score        : %lg\n", sumf1[j]/N);
				fprintf(out, "\n");
			} 
	}
	fprintf(out, "\n\n\n\n\n");

	//********************PREDICTION FILE **************************
	if(rare_class.size() != numMap.size())
	{
		FILE *reportf = fopen(predictionfilename.c_str(), "w");
		if(reportf == NULL)
		{
			fprintf(stderr, "error, cannot create file %s\n", predictionfilename.c_str());
			exit(1);
		}

		for(i=0;i<N;i++, fprintf(reportf, "\n"))
		{
			if(prediction[i].empty())
				fprintf(reportf, "none");
			else
			{
				it=prediction[i].begin();
				fprintf(reportf, "%s", numMap[*it].c_str());
				it++;
				for(;it!=prediction[i].end();it++)
					fprintf(reportf, ",%s", numMap[*it].c_str());
			}
		}
		fclose(reportf);
		//********************END PREDICTION FILE **************************

		//********************CSV FILE **************************
		FILE *scoref;

		scoref = fopen(csvfilename.c_str(), "wb");
		if(scoref == NULL)
		{
			fprintf(stderr, "error, cannot create file %s.csv - please make sure that file is not used by other program.\n", csvfilename.c_str());
			exit(1);
		}

		fprintf(scoref, "FOLD");
		for(i=0;i<class_no;i++)
			if(rare_class.find(i) == rare_class.end())
				fprintf(scoref, ",''%s''", numMap[i].c_str());
		fprintf(scoref, "\n");

		for(j=0;j<N;j++, fprintf(scoref, "\n"))
		{
			fprintf(scoref, "%d", m[j]);
			for(i=0;i<class_no;i++)
				if(rare_class.find(i) == rare_class.end())
					if(svm_sc[j].find(i) != svm_sc[j].end())
						fprintf(scoref, ",%lg", svm_sc[j][i]);
					else
						fprintf(scoref, ",");
		}
		fclose(scoref);
		//********************END CSV FILE **************************
	}

	printf("Finished.\n");
	printf("Report file has been created to %s\n", reportfilename.c_str());
	if(rare_class.size() == classMap.size())
		printf("***prediction file cannot not be created because all class is rare class");
	else
		printf("Prediction file has been created to %s\n", predictionfilename.c_str());
	printf("Model file has been created to %s_fold<fold number>\n", model_folder_name);
	
	fclose(out);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name, char *model_folder_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	do_RSVM = 0;
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1.0;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*100);
	param.weight = (double *)realloc(param.weight,sizeof(double)*100);
// param.weight_label = NULL;
// param.weight = NULL;
	param.hierarchical_classification = 0;
	param.multilabel_classification = 0;
	param.multiclass_classification = 0;
	param.binary_classification = 1;
	cross_validation = 0;
	param.do_feature_selection = 0;
	rare_class_threshold = -1;
	randomSeed = -1;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;

		if(++i>=argc)
			exit_with_help();

		switch(argv[i-1][1])
		{
			case 'x':
				randomSeed = atoi(argv[i]);
			break;

			case 'a':
				do_RSVM = atoi(argv[i]);
			break;

			case 'k':
				//convert argv[i] to lowercase.
				transform(argv[i], argv[i]+strlen(argv[i]), argv[i], ::tolower);

				if(strcmp(argv[i], "binary") == 0)
				{
					param.binary_classification = 1;
				}
				else if(strcmp(argv[i], "multiclass") == 0)
				{
					param.binary_classification = 0;
					param.multiclass_classification = 1;
				}
				else if(strcmp(argv[i], "multilabel") == 0)
				{
					param.binary_classification = 0;
					param.multilabel_classification = 1;
				}
				else if(strcmp(argv[i], "hierarchical") == 0)
				{
					param.binary_classification = 0;
					param.hierarchical_classification = 1;
					if(++i>=argc)
						exit_with_help();
					strcpy(hierarchical_classification_path, argv[i]);
				}
				else
				{
					fprintf(stderr,"unkown classification type: %s\n", argv[i]);
					exit_with_help();            
				}
			break;

			case 'l':
				rare_class_threshold = atoi(argv[i]);
			break;

			case 'f':
				param.do_feature_selection = atoi(argv[i]);
			break;

			case 's':
				param.svm_type = atoi(argv[i]);
			break;

			case 't':
				param.kernel_type = atoi(argv[i]);
			break;

			case 'd':
				param.degree = atoi(argv[i]);
			break;

			case 'g':
				param.gamma = atof(argv[i]);
			break;
			
			case 'r':
				param.coef0 = atof(argv[i]);
			break;
			
			case 'n':
				param.nu = atof(argv[i]);
			break;
			
			case 'm':
				param.cache_size = atof(argv[i]);
			break;
			
			case 'c':
				param.C = atof(argv[i]);
			break;
			
			case 'e':
				param.eps = atof(argv[i]);
			break;
			
			case 'p':
				param.p = atof(argv[i]);
			break;
			
			case 'h':
				param.shrinking = atoi(argv[i]);
			break;
			
			case 'b':
				param.probability = atoi(argv[i]);
			break;
			
			case 'q':
				print_func = &print_null;
				i--;
			break;

			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
			break;
			
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
			break;
			
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
			break;
		}
	}

	svm_set_print_string_function(print_func);

	// determine filenames

	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_folder_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_folder_name,"%s.model",p);
	}
	
	char *it;
	for(it=model_folder_name;*it!='\0';it++)
		if(*it == '\\')
			*it = '/';

	char *rr = strrchr(model_folder_name, '/');
	if(rr == NULL)
		rr = model_folder_name;
	else
		rr++;
	strcpy(model_file_name_without_path, rr);
	sprintf(model_file_name, "%s/%s", model_folder_name, rr);
	//printf("%s\n", model_file_name);
}

void set_hierarchy_depth(int i, int level)
{
	vector<int>::iterator it;

	if(level > max_depth)
		max_depth = level;

	if(depth[i].find(level) == depth[i].end())
	{
		depth[i].insert(level);
		for(it=child[i].begin();it!=child[i].end();it++)
			set_hierarchy_depth(*it, level+1);
	}
}

void read_hierarchy_file(char* input_file_name) 
{
	char filename[256];
	char p[4096], c[4096];
	int i;

	strcpy(filename, input_file_name);
	FILE *fp = fopen(filename,"r");

	if(fp == NULL)
	{
		fprintf(stderr,"can't open hierarchy file %s\n",filename);
		exit(1);
	}

	//check if there are new class.
	while(fscanf(fp, "%s", p) != EOF)
	{
		string strlabel = string(p);

		if(strlabel.compare("none") == 0)
		{
			fprintf(stderr, "Wrong input format in hierarchy file. The class label cannot be \"none\".\n");
			exit(1);
		}

		if(classMap.find(strlabel) == classMap.end())
		{
			int num = (int)classMap.size();
			//printf("%s %d\n", strlabel.c_str(), num);
			classMap[strlabel] = num;
			numMap[num] = strlabel;
		}
	}
	rewind(fp);

	class_no = (int)classMap.size();
	parent.resize(class_no);
	child.resize(class_no);
	vector<set<int> > parent_set(class_no);

	while(fscanf(fp, "%s", p) != EOF)
	{
		fscanf(fp, "%s", c);
		int ci = classMap[string(c)], pi = classMap[string(p)];

		if(parent_set[ci].find(pi) == parent_set[ci].end())
		{
			parent[ci].push_back(pi);
			child[pi].push_back(ci);
			parent_set[ci].insert(pi);
		}
	}

	for(i=0;i<class_no;i++)
	{
		sort(parent[i].begin(), parent[i].end());
		sort(child[i].begin(), child[i].end());
	}

	depth.resize(class_no);
	for(i=0;i<class_no;i++)
		if(parent[i].empty())
		{
			root.push_back(i);
			set_hierarchy_depth(i, 1);
		}

	fclose(fp);
	return;
}

void shrink_dag_class_list(vector<int> &l, set<int> &ans)
{
	int i;
	set<int> added;
	set<int>::iterator it;

	for(i=0;i<(int)l.size();i++)
		if(added.find(l[i]) == added.end())
		{   
			set<int> ancestor;
			get_ancestor_list(l[i], ancestor);
			for(it=ancestor.begin();it!=ancestor.end();it++)
			{
				added.insert(*it);

				if(ans.find(*it) != ans.end())
				{
					ans.erase(*it);       
				}
			}

			ans.insert(l[i]);
		}
}

void read_problem(const char *filename)
{
	int inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label, *label_part;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0; // number of training example
	elements = 0; // number of non-zero features

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	label_part = Malloc(char,max_line_len);

	//analyzing data.
	while(readline(fp)!=NULL)
	{
		int label_amount = 1;
		int k=0;
		while(1) 
		{
			if(line[k] == ',')
			{
				label_amount++;
				k++;
				if(line[k] == ',' || line[k] == ' ' || line[k] == '\n' || line[k] == '\t' || line[k] == '\0')
					exit_input_error(prob.l+1);
			}
			else if (line[k] == ' ' || line[k] == '\t' || line[k] == '\0')
				break;
			k++;
		}

		while(line[k] != '\0')
		{
			if(line[k] == ':')
				elements++;
			k++;
		}
		elements++;

		char *p = strtok(line,", \t\n"); 

		if(param.multiclass_classification && label_amount > 1) 
		{
			fprintf(stderr,"Wrong input format at line %d : Multiclass can assign only one label on each example.\n", prob.l+1);
			exit(1);
		}

		// label part
		for(i=0;i<label_amount;i++)
		{
			if(i)
				p=strtok(NULL, ", \t");

			string strlabel = string(p);
			if(strlabel.compare("none") != 0 && classMap.find(strlabel) == classMap.end())
			{
				int num = (int)classMap.size();
				classMap[strlabel] = num;
				numMap[num] = strlabel;
			}
		}
		++prob.l;
	}
	rewind(fp);

	if(param.hierarchical_classification)
		read_hierarchy_file(hierarchical_classification_path);

	class_no = (int)classMap.size();
	class_list = new vector<int>[class_no];

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	tr_class.resize(prob.l);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++) // loop for each training example
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);

		int label_amount = 1;
		int k=0;
		while(1) 
		{
			if (line[k] == ',') label_amount++;
			else if (line[k] == ' ' || line[k] == '\t' || line[k] == '\0') break;
			k++;
		}

		prob.x[i] = &x_space[j];
		label = strtok(line,", \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		string labelstr = string(label);
		int label_int;
		if(labelstr.compare("none") == 0)
			label_int = -1;
		else
			label_int = classMap[labelstr];

		if(label_int >= 0)
		{
			tr_class[i].push_back(label_int);
			if(!param.hierarchical_classification)
				class_list[label_int].push_back(i);
		}
		
		if (label_amount > 1)
			for (k=1;k<label_amount;k++) 
			{
				char* myLabel = strtok(NULL,", ");

				labelstr = string(myLabel);
				if(labelstr.compare("none") == 0)
					label_int = -1;
				else
					label_int = classMap[labelstr];

				if(label_int < 0)
				{
					fprintf(stderr,"Wrong input format at line %d: None label specified with another label\n", i+1);
					exit(1);
				}

				tr_class[i].push_back(label_int);
				if(!param.hierarchical_classification)
					class_list[label_int].push_back(i);
			}

		if(param.hierarchical_classification)
		{
			set<int> shrinked;
			set<int>::iterator it;
			shrink_dag_class_list(tr_class[i], shrinked);
			tr_class[i].clear();
			for(it=shrinked.begin();it!=shrinked.end();it++)
			{
				tr_class[i].push_back(*it);
				class_list[*it].push_back(i);
			}
		}

		int jtmp = j;
		set<int> added;
		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");
			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);   
			if(endptr == idx || errno != 0 || *endptr != '\0')// || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else if(x_space[j].index < 0)
			{
				fprintf(stderr,"Wrong input format at line %d: attribute index must >= 0", i+1);
				exit(1);
			}
			else if(added.find(x_space[j].index) != added.end())
			{
				fprintf(stderr,"Wrong input format at line %d: duplicate attribute number %d", i+1, x_space[j].index);
				exit(1);
			}
			else if(inst_max_index < x_space[j].index)
				inst_max_index = x_space[j].index;
			
			added.insert(x_space[j].index);
			errno = 0;
			x_space[j].value = strtod(val,&endptr);

			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);
			++j;
		}

		sort(x_space+jtmp, x_space+j, x_space_cmp);

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;
	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
	free(label_part);
}

void original_read_problem(const char *filename)
{
	int inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	class_no = 2;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		string strlabel = string(p);
		if(classMap.find(strlabel) == classMap.end())
		{
			int num = ((int)classMap.size())*2-1;
			classMap[strlabel] = num;
			numMap[num] = strlabel;
		}
	
		if(classMap.size() > 2)
		{
			fprintf(stderr, "Wrong input format: Binary classification must have only 2 class.\n");
			exit(1);
		}

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = classMap[string(label)];//strtod(label,&endptr);

		int jtmp = j;
		set<int> added;
		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");
			if(val == NULL)
				break;
			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);   
			if(endptr == idx || errno != 0 || *endptr != '\0')// || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else if(x_space[j].index < 0)
			{
				fprintf(stderr,"Wrong input format at line %d: attribute index must >= 0", i+1);
				exit(1);
			}
			else if(added.find(x_space[j].index) != added.end())
			{
				fprintf(stderr,"Wrong input format at line %d: duplicate attribute number %d", i+1, x_space[j].index);
				exit(1);
			}
			else if(inst_max_index < x_space[j].index)
				inst_max_index = x_space[j].index;
			
			added.insert(x_space[j].index);
			errno = 0;
			x_space[j].value = strtod(val,&endptr);

			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);
			++j;
		}

		sort(x_space+jtmp, x_space+j, x_space_cmp);

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}
