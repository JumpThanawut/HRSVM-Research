#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include<algorithm>
#include<float.h>
#include<map>
#include<set>
#include<string>
#include<time.h>
#include <sstream>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

using namespace std;

int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

struct svm_node *x;
int max_nr_attr = 64;

struct svm_problem prob;
struct svm_parameter param;
struct svm_node *x_space;
vector <int> *class_list;
vector<int> valid_model;
int class_no, do_feature_selection=0, do_RSVM=0;
char hierarchical_classification_path[1050];
bool report_svm_score = true;
char input_file_name[1024];
char output_file_name[1024];
char model_file_name[1024];

vector<vector<int> > parent, child, tr_class;
vector<int> root;
vector<set<int> > depth;
vector<map<int, int> > selected_feature;

int max_depth = 0;

int max_index, elements;

map<string, int> classMap;
map<int, string> numMap;
// Jump

struct svm_model* model;
int predict_probability=0;

static char *line = NULL;
static int max_line_len;

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

bool valid_model_cmp (int i,int j) 
{ 
	return numMap[i].compare(numMap[j]) < 0; 
}
bool x_space_cmp(const svm_node &x, const svm_node &y)
{
	return x.index < y.index;
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

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void exit_with_help()
{
	printf(
	"Usage: svm-predict test_file model_file output_file\n"
	);
	exit(1);
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
	int p, c, i;

	strcpy(filename, input_file_name);
	FILE *fp = fopen(filename,"r");

	if(fp == NULL)
	{
		fprintf(stderr,"can't open hierarchy file %s\n",filename);
		exit(1);
	}
	
	parent.resize(class_no);
	child.resize(class_no);
	vector<set<int> > parent_set(class_no);

	while(fscanf(fp, "%d", &p) != EOF)
	{
		fscanf(fp, "%d", &c);

		if(parent_set[c].find(p) == parent_set[c].end())
		{
			parent[c].push_back(p);
			child[p].push_back(c);
			parent_set[c].insert(p);
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
		char *p = strtok(line," \t"); 

		//feature part
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

		if(labelstr.compare("none") == 0 || classMap.find(labelstr) == classMap.end())
			label_int = -1;
		else
			label_int = classMap[labelstr];

		if(label_int >= 0)
		{
			tr_class[i].push_back(label_int);
			if(!param.hierarchical_classification)
				class_list[label_int].push_back(i);
		}

		if(param.binary_classification)
		{
			prob.y[i] = label_int;
			if(label_amount > 1)
			{
				fprintf(stderr, "Wrong input format at line %d: example is multi-label.\n", i+1);
				exit(1);
			}
		}

		if (label_amount > 1)
			for (k=1;k<label_amount;k++) 
			{
				char* myLabel = strtok(NULL,", ");

				labelstr = string(myLabel);
				if(labelstr.compare("none") == 0 || classMap.find(labelstr) == classMap.end())
					label_int = -1;
				else
					label_int = classMap[labelstr];

				if(label_int >= 0)
				{
					tr_class[i].push_back(label_int);
					if(!param.hierarchical_classification)
						class_list[label_int].push_back(i);
				}
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


void predict_binary(char *model_file_name, char *output_file_name)
{
	int N, i, j;
	vector<svm_node*> test;
	vector<double> testCl;
	double testTime;
	clock_t start, stop;

	start = clock();
	N = prob.l;

	for(j=0;j<N;j++)
	{
		test.push_back(prob.x[j]);
		testCl.push_back(prob.y[j]);
	}

	FILE *scoref;
	if(report_svm_score)
	{
		scoref = fopen((string(output_file_name)+".csv").c_str(), "wb");
		if(scoref == NULL)
		{
			fprintf(stderr, "error, cannot create file %s.csv - please make sure that file is not used by other program.\n", output_file_name);
			exit(1);
		}

		fprintf(scoref, "''%s''\n", numMap[1].c_str());
	}

	char model_file_name_tmp[1050];
	sprintf(model_file_name_tmp, "%s.%s", model_file_name, numMap[1].c_str());
	model = svm_load_model(model_file_name_tmp);
	if(model == 0)
	{
		fprintf(stderr,"can't open model file %s\n",model_file_name_tmp);
		exit(1);
	}

	int tp, tn, fp, fn;
	tp = tn = fp = fn = 0;
	FILE *out = fopen(output_file_name, "wb");
	if(out == NULL)
	{
		fprintf(stderr, "error, cannot create file %s\n", output_file_name);
		exit(1);
	}

	FILE *reportf = fopen((string(output_file_name)+".report").c_str(), "wb");
	if(reportf == NULL)
	{
		fprintf(stderr, "error, cannot create file %s.report\n", output_file_name);
		exit(1);
	}

	for(j=0;j<(int)test.size();j++)
	{
		vector<svm_node> xtmp;
		svm_node *x = test[j];

		if(!selected_feature[0].empty())
		{
			x = test[j];
			map<int, int>::iterator it;

			i=0;
			while(x[i].index != -1)
			{
				it = selected_feature[0].find(x[i].index);
				if(it != selected_feature[0].end())
				{
					svm_node tmp = x[i];
					tmp.index = selected_feature[0][tmp.index];
					xtmp.push_back(tmp);
				}
				i++;
			}

			svm_node end;
			end.index = -1;
			xtmp.push_back(end);

			x = &xtmp[0];
		}

		double p = svm_predict(model, x);
		if(report_svm_score)
			fprintf(scoref, "%g\n", p);
		fprintf(out, "%s\n", p>0.0?numMap[1].c_str():numMap[-1].c_str());

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

	double F = 0.0;
	if(tp != 0 || fp != 0 || fn != 0)
		F = 2*tp/(double)(2*tp+fp+fn);

	stop = clock();
	testTime = (stop-start)/(double)CLOCKS_PER_SEC;

	{
		svm_parameter &param = model->param;
		fprintf(reportf, "Model file         : %s\n", model_file_name);
		fprintf(reportf, "Input file         : %s\n", input_file_name);
		fprintf(reportf, "Kernel type        : %s\n", kernel_type_table[param.kernel_type]);
		if(param.kernel_type == POLY)
			fprintf(reportf, "Degree             : %d\n", param.degree);
		if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
			fprintf(reportf, "Gamma              : %g\n", param.gamma);
		if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
			fprintf(reportf, "Coef0              : %g\n", param.coef0);
		fprintf(reportf, "Feature selection  : %s\n", do_feature_selection?"on":"off");
		fprintf(reportf, "RSVM               : %s\n", do_RSVM?"on":"off");
		fprintf(reportf, "#Class             : %d\n", (int)numMap.size());
		fprintf(reportf, "#Examples          : %d\n", prob.l);
		fprintf(reportf, "\n");
	}

	fprintf(reportf, "Accuracy  : %g\n", prob.l!=0?(1.0-(fp+fn)/(double)prob.l):0.0);
	fprintf(reportf, "Precision : %g\n", (tp+fp)!=0?tp/(double)(tp+fp):0.0);
	fprintf(reportf, "Recall    : %g\n", (tp+fn)!=0?tp/(double)(tp+fn):0.0);
	fprintf(reportf, "F1Score   : %g\n", F);
	fprintf(reportf, "Test time : %g sec.\n", testTime);

	fclose(out);
	fclose(reportf);
	if(report_svm_score) 
		fclose(scoref);
}

void predict_multi(char *model_file_name, char *output_file_name)
{
	double f1micro, f1macro, predmacro, recallmacro, avgacc;
	int N, i, j, k;
	int sumTP, sumFP, sumFN, sumTN;
	vector<svm_node*> test;
	svm_problem probTmp = prob;
	vector<int> testIdx, cl;
	vector<int>::iterator it;
	clock_t start, stop;
	double testTime;
	stringstream buff;

	start = clock();
	N = prob.l;
	f1micro = f1macro = predmacro = recallmacro = avgacc = 0.0;
	sumTP=sumFP=sumFN=sumTN=0;
	
	for(j=0;j<N;j++)
	{
		test.push_back(prob.x[j]);
		testIdx.push_back(j);
	}

	FILE *scoref;
	if(report_svm_score)
	{
		scoref = fopen((string(output_file_name)+".csv").c_str(), "wb");
		if(scoref == NULL)
		{
			fprintf(stderr, "error, cannot create file %s.csv - please make sure that file is not used by other program.\n", output_file_name);
			exit(1);
		}
	}

	FILE *reportf = fopen((string(output_file_name)+".report").c_str(), "wb");
	if(reportf == NULL)
	{
		fprintf(stderr, "error, cannot create file %s.report\n", output_file_name);
		exit(1);
	}

	buff << "Measures of each classes\n";

	cl.resize(testIdx.size());
	vector<double> maxp(testIdx.size(), -DBL_MAX);
	vector<vector<int> > prediction(N);
	vector<vector<double> > prediction_table(test.size());
	for(it=valid_model.begin();it!=valid_model.end();it++)
	{
		i = *it;
		vector<double> y(N);
		vector<double> testCl;

		for(j=0;j<N;j++) 
			y[j] = -1.0;
		for(j=0;j<(int)class_list[i].size();j++)
			y[class_list[i][j]] = 1.0;

		for(j=0;j<N;j++)
		{
			testCl.push_back(y[j]);
		}

		char model_file_name_tmp[1050];
		sprintf(model_file_name_tmp, "%s.%s", model_file_name, numMap[i].c_str());
		model = svm_load_model(model_file_name_tmp);

		if(param.multilabel_classification)
		{
			int tp, tn, fp, fn;
			tp = tn = fp = fn = 0;
			for(j=0;j<(int)test.size();j++)
			{
				vector<svm_node> xtmp;
				svm_node *x = test[j];

				if(!selected_feature[i].empty())
				{
					x = test[j];
					map<int, int>::iterator it;

					k=0;
					while(x[k].index != -1)
					{
						it = selected_feature[i].find(x[k].index);
						if(it != selected_feature[i].end())
						{
							svm_node tmp = x[k];
							tmp.index = selected_feature[i][tmp.index];
							xtmp.push_back(tmp);
						}
						k++;
					}

					svm_node end;
					end.index = -1;
					xtmp.push_back(end);

					x = &xtmp[0];
				}

				double p = svm_predict(model, x);
				prediction_table[j].push_back(p);

				if(p > 0.0)
					prediction[j].push_back(i);

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

			double F = 0.0;
			if(tp != 0 || fp != 0 || fn != 0)
				F = 2*tp/(double)(2*tp+fp+fn);
			double pred = (tp+fp)!=0?tp/(double)(tp+fp):0.0;
			double recall = (tp+fn)!=0?tp/(double)(tp+fn):0.0;
			double acc = prob.l!=0?(1.0-(fp+fn)/(double)prob.l):0.0;

			buff <<  "Class " << numMap[i].c_str() << '\n';
			buff <<  "Accuracy  : " << acc << '\n';
			buff <<  "Precision : " << pred << '\n';
			buff <<  "Recall    : " << recall << '\n';
			buff <<  "F1Score   : " << F << '\n';
			buff <<  "\n";

			predmacro += pred;
			recallmacro += recall;
			avgacc += acc;
			f1macro+=F;
		}
		else //multiclass classification
		{
			for(j=0;j<(int)test.size();j++)
			{
				vector<svm_node> xtmp;
				svm_node *x = test[j];

				if(!selected_feature[i].empty())
				{
					x = test[j];
					map<int, int>::iterator it;

					k=0;
					while(x[k].index != -1)
					{
						it = selected_feature[i].find(x[k].index);
						if(it != selected_feature[i].end())
						{
							svm_node tmp = x[k];
							tmp.index = selected_feature[i][tmp.index];
							xtmp.push_back(tmp);
						}
						k++;
					}

					svm_node end;
					end.index = -1;
					xtmp.push_back(end);

					x = &xtmp[0];
				}

				double p = svm_predict(model, x);
				prediction_table[j].push_back(p);

				if(p > maxp[j])
				{
					maxp[j] = p;
					cl[j] = i;
				}    
			}
		}

		prob = probTmp;
	}

	if(report_svm_score)
	{
		it=valid_model.begin();
		fprintf(scoref, "''%s''", numMap[*it].c_str());
		it++;
		for(;it!=valid_model.end();it++)
			fprintf(scoref, ",''%s''", numMap[*it].c_str());
		fprintf(scoref, "\n");

		for(i=0;i<(int)test.size();i++, fprintf(scoref, "\n"))
		{
			j=0;
			fprintf(scoref, "%g", prediction_table[i][j]);
			j++;
			for(;j<(int)prediction_table[i].size();j++)
				fprintf(scoref, ",%g", prediction_table[i][j]);
		}

		fclose(scoref);
	}

	FILE *out = fopen(output_file_name, "wb");
	if(out == NULL)
	{
		fprintf(stderr, "error, cannot create file %s.\n", output_file_name);
		exit(1);
	}

	if(param.multiclass_classification)
	{
		vector<int> tp(class_no, 0), fp(class_no, 0), fn(class_no, 0);
		
		for(j=0;j<(int)testIdx.size();j++)
		{
			int tr_cl = -1;
			int pr_cl = cl[j];

			if(!tr_class[testIdx[j]].empty())
				tr_cl = tr_class[testIdx[j]][0];

			fprintf(out, "%s\n", numMap[pr_cl].c_str());

			if(tr_cl >= 0 && tr_cl == pr_cl)
				tp[tr_cl]++;
			else
			{
				fp[pr_cl]++;
				if(tr_cl >= 0)
					fn[tr_cl]++;
			}
		}

		for(it=valid_model.begin();it!=valid_model.end();it++)//for(i=0;i<class_no;i++)
		{
			i = *it;
			sumTP+=tp[i];
			sumFP+=fp[i];
			sumFN+=fn[i];

			double F = 0.0;
			if(tp[i] != 0 || fp[i] != 0 || fn[i] != 0)
				F = 2*tp[i]/(double)(2*tp[i]+fp[i]+fn[i]);
			double pred = (tp[i]+fp[i])!=0?tp[i]/(double)(tp[i]+fp[i]):0.0;
			double recall = (tp[i]+fn[i])!=0?tp[i]/(double)(tp[i]+fn[i]):0.0;
			double acc = prob.l!=0?(1.0-(fp[i]+fn[i])/(double)prob.l):0.0;

			buff <<  "Class " << numMap[i].c_str() << '\n';
			buff <<  "Accuracy  : " << acc << '\n';
			buff <<  "Precision : " << pred << '\n';
			buff <<  "Recall    : " << recall << '\n';
			buff <<  "F1Score   : " << F << '\n';
			buff <<  "\n";

			predmacro += pred;
			recallmacro += recall;
			avgacc += acc;
			f1macro+=F;
		}
	}
	else //multi label
		for(j=0;j<(int)testIdx.size();j++)
		{
			vector<int>::iterator it;

			if(prediction[j].empty())
				fprintf(out, "none");
			else
			{
				it=prediction[j].begin();
				fprintf(out, "%s", numMap[*it].c_str());
				it++;
				for(;it!=prediction[j].end();it++)
					fprintf(out, ",%s", numMap[*it].c_str());
			}

			fprintf(out, "\n");
		}

	f1macro/=(double)valid_model.size();
	avgacc/=(double)valid_model.size();
	recallmacro/=(double)valid_model.size();
	predmacro/=(double)valid_model.size();

	f1micro = 0.0;
	if(sumTP != 0 || sumFP != 0 || sumFN != 0)
		f1micro = 2*sumTP/(double)(2*sumTP+sumFP+sumFN);

	stop = clock();
	testTime = (stop-start)/(double)CLOCKS_PER_SEC;

	{
		svm_parameter &param = model->param;
		fprintf(reportf, "Model file         : %s\n", model_file_name);
		fprintf(reportf, "Input file         : %s\n", input_file_name);
		fprintf(reportf, "Kernel type        : %s\n", kernel_type_table[param.kernel_type]);
		if(param.kernel_type == POLY)
			fprintf(reportf, "Degree             : %d\n", param.degree);
		if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
			fprintf(reportf, "Gamma              : %g\n", param.gamma);
		if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
			fprintf(reportf, "Coef0              : %g\n", param.coef0);
		fprintf(reportf, "Feature selection  : %s\n", do_feature_selection?"on":"off");
		fprintf(reportf, "RSVM               : %s\n", do_RSVM?"on":"off");
		fprintf(reportf, "#Class             : %d\n", (int)numMap.size());
		fprintf(reportf, "#Examples          : %d\n", prob.l);
		fprintf(reportf, "#Rare class        : %d\n", (int)(numMap.size()-valid_model.size()));
		fprintf(reportf, "\n");
	}

	fprintf(reportf, "Overall classes\n");
	fprintf(reportf, "Accuracy        : %g\n", avgacc);
	fprintf(reportf, "Micro Precision : %g\n", (sumTP+sumFP)!=0?sumTP/(double)(sumTP+sumFP):0.0);
	fprintf(reportf, "Macro Precision : %g\n", predmacro);
	fprintf(reportf, "Micro Recall    : %g\n", (sumTP+sumFN)!=0?sumTP/(double)(sumTP+sumFN):0.0);
	fprintf(reportf, "Macro Recall    : %g\n", recallmacro);
	fprintf(reportf, "Micro F1Score   : %g\n", f1micro);
	fprintf(reportf, "Macro F1Score   : %g\n", f1macro);
	fprintf(reportf, "Test Time       : %g sec.\n", testTime);
	fprintf(reportf, "\n%s", buff.str().c_str());

	fclose(out);
	fclose(reportf);
}

void predict_hierarchical(char *model_file_name, char *output_file_name)
{
	double f1micro, f1macro, hf1micro, hf1macro;
	int N, i, j;
	int sumTP, sumFP, sumFN, sumTN;
	vector<int> test_data_idx;
	vector<svm_model *> model;
	vector<svm_node> original_x_space;
	set<int>::iterator it;
	double precmacro, precmicro, reclmacro, reclmicro, accmacro;
	double hprecmicro, hprecmacro, hreclmicro, hreclmacro;
	double testTime;
	clock_t start, stop;
	stringstream buff;

	start = clock();

	N = prob.l;
	f1micro=0.0;
	f1macro=0.0;
	hf1micro=0.0;
	hf1macro=0.0;
	precmacro = reclmacro = accmacro = 0.0;
	hprecmicro = hprecmacro = hreclmicro = hreclmacro = 0.0;
	model.resize(class_no);

	vector<svm_node*> test;
	//add testing set.
	for(j=0;j<N;j++)
	{
		test.push_back(prob.x[j]);
		test_data_idx.push_back(j);
	}   

	FILE *out = fopen(output_file_name, "wb");
	for(i=0;i<class_no;i++) 
		model[i] = NULL; 

	FILE *scoref;
	if(report_svm_score)
	{
		vector<int>::iterator it;

		scoref = fopen((string(output_file_name)+".csv").c_str(), "wb");
		if(scoref == NULL)
		{
			fprintf(stderr, "error, cannot create file %s.csv - please make sure that file is not used by other program.\n", output_file_name);
			exit(1);
		}

		it=valid_model.begin();
		fprintf(scoref, "''%s''", numMap[*it].c_str());
		it++;
		for(;it!=valid_model.end();it++)
			fprintf(scoref, ",''%s''", numMap[*it].c_str());
		fprintf(scoref, "\n");
	}

	FILE *reportf = fopen((string(output_file_name)+".report").c_str(), "wb");
	if(reportf == NULL)
	{
		fprintf(stderr, "error, cannot create file %s.report\n", output_file_name);
		exit(1);
	}

	FILE *predtmpf = fopen((string(output_file_name)+".predtmp").c_str(), "wb");
	if(predtmpf == NULL)
	{
		fprintf(stderr, "error, cannot create tmp file %s.predtmp\n", output_file_name);
		exit(1);
	}

	FILE *truetmpf = fopen((string(output_file_name)+".truetmp").c_str(), "wb");
	if(truetmpf == NULL)
	{
		fprintf(stderr, "error, cannot create tmp file %s.truetmp\n", output_file_name);
		exit(1);
	}
	for(i=0;i<(int)test.size();i++)
	{
		int test_idx = test_data_idx[i];
		for(j=0;j<(int)tr_class[test_idx].size();j++)
			fprintf(truetmpf, "%d ", tr_class[test_idx][j]+1);
		fprintf(truetmpf, "\n");
	}
	fclose(truetmpf);

	{
		vector<int>::iterator it;
		for(it=valid_model.begin();it!=valid_model.end();it++)
		{
			char model_file_name_tmp[1050];
			sprintf(model_file_name_tmp, "%s.%s", model_file_name, numMap[*it].c_str());
			model[*it] = svm_load_model(model_file_name_tmp);   
		}
	}

	{
		svm_parameter &param = model[valid_model[0]]->param;
		fprintf(reportf, "Model file         : %s\n", model_file_name);
		fprintf(reportf, "Input file         : %s\n", input_file_name);
		fprintf(reportf, "Feature selection  : %s\n", do_feature_selection?"on":"off");
		fprintf(reportf, "RSVM               : %s\n", do_RSVM?"on":"off");
		fprintf(reportf, "#Class             : %d\n", (int)numMap.size());
		fprintf(reportf, "#Examples          : %d\n", prob.l);
		fprintf(reportf, "#Rare class        : %d\n", (int)(numMap.size()-valid_model.size()));
		fprintf(reportf, "Max depth          : %d\n", max_depth);
		fprintf(reportf, "\n");
	}

	vector<int> tp(class_no, 0), fp(class_no, 0), fn(class_no, 0); 
	vector<int> htp(test.size(), 0), hfp(test.size(), 0), hfn(test.size(), 0); 
	set<int> valid_model_set;

	{
		vector<int>::iterator it;
		for(it=valid_model.begin();it!=valid_model.end();it++)
			valid_model_set.insert(*it);
	}

	for(i=0;i<(int)test.size();i++)
	{
		set<int> pred_cl;
		map<int, double> svm_sc;  
		vector<int>::iterator it2;
		svm_predict_hierarchical(root, model, test[i], selected_feature, child, pred_cl, svm_sc);

		if(report_svm_score)
		{
			it2=valid_model.begin();
			if(svm_sc.find(*it2) != svm_sc.end())
				fprintf(scoref, "%g", svm_sc[*it2]);
			it2++;
			for(;it2!=valid_model.end();it2++)
				if(svm_sc.find(*it2) != svm_sc.end())
					fprintf(scoref, ",%g", svm_sc[*it2]);
				else
					fprintf(scoref, ",");
			fprintf(scoref, "\n");
		}

		int test_idx = test_data_idx[i];
		//variable tr_class contains only leaf. so extend version will contains all node.
		set<int> tr_cl_ext;
	
		for(j=0;j<(int)tr_class[test_idx].size();j++)
		{
			get_ancestor_list(tr_class[test_idx][j], tr_cl_ext);
			tr_cl_ext.insert(tr_class[test_idx][j]);
		}

		for(it=tr_cl_ext.begin();it!=tr_cl_ext.end();)
			if(valid_model_set.find(*it) == valid_model_set.end())
				tr_cl_ext.erase(it++);
			else
				it++;  

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

		if(pred_cl.empty())
			fprintf(out, "none");
		else
		{
			it=pred_cl.begin();
			fprintf(out, "%s", numMap[*it].c_str());
			it++;
			for(;it!=pred_cl.end();it++)
				fprintf(out, ",%s", numMap[*it].c_str());
		}
		fprintf(out, "\n");

		for(it=pred_cl.begin();it!=pred_cl.end();it++)
			fprintf(predtmpf, "%d ", *it+1);
		fprintf(predtmpf, "\n");
	}
	fclose(predtmpf);

	//class based micro macro.
	//fprintf(reportf, "Measures of each classes\n");
	buff << "Measures of each classes\n";

	sumTP = sumTN = sumFP = sumFN = 0; 
	{
		vector<int>::iterator it;
		for(it=valid_model.begin();it!=valid_model.end();it++)
		{
			i = *it;
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

			buff <<  "Class " << numMap[i].c_str() << '\n';

			set<int>::iterator it2;
			buff <<  "level     : ";
			if(!depth[i].empty())
			{
				it2=depth[i].begin();
				buff <<  *it2;
				it2++;
				for(;it2!=depth[i].end();it2++)
					buff << ',' << *it2;
			}
			buff <<  "\n";

			buff <<  "Accuracy  : " << ACC << '\n';
			buff <<  "Precision : " << PREC << '\n';
			buff <<  "Recall    : " << RECL << '\n';
			buff <<  "F1Score   : " << F << '\n';
			buff <<  "\n";
		}
	}
	f1macro/=(double)valid_model.size();
	precmacro/=(double)valid_model.size();
	reclmacro/=(double)valid_model.size();
	accmacro/=(double)valid_model.size();

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
	hf1macro/=(int)test.size(); 
	hprecmacro/=(int)test.size();
	hreclmacro/=(int)test.size();

	hf1micro = hprecmicro = hreclmicro = 0.0;
	if(sumTP != 0 || sumFP != 0 || sumFN != 0)
		hf1micro = 2*sumTP/(double)(2*sumTP+sumFP+sumFN);
	if(sumTP != 0 || sumFP != 0)
		hprecmicro = sumTP/(double)(sumTP+sumFP);
	if(sumTP != 0 || sumFN != 0)
		hreclmicro = sumTP/(double)(sumTP+sumFN);

	double ElbF1micro, ElbF1macro, Elbprecmicro, Elbprecmacro, Elbreclmicro, Elbreclmacro;
	ElbF1micro = ElbF1macro = Elbprecmicro = Elbprecmacro = Elbreclmicro = Elbreclmacro = 0.0;
	if(hprecmicro + precmicro > epsilon)
		Elbprecmicro = 2*hprecmicro*precmicro/(hprecmicro+precmicro);
	if(hprecmacro + precmacro > epsilon)
		Elbprecmacro = 2*hprecmacro*precmacro/(hprecmacro+precmacro);
	if(hreclmicro + reclmicro > epsilon)
		Elbreclmicro = 2*hreclmicro*reclmicro/(hreclmicro+reclmicro);
	if(hreclmacro + reclmacro > epsilon)
		Elbreclmacro = 2*hreclmacro*reclmacro/(hreclmacro+reclmacro);
	if(hf1micro+f1micro > epsilon)
		ElbF1micro = 2*hf1micro*f1micro/(hf1micro+f1micro);
	if(hf1macro+f1macro > epsilon)
		ElbF1macro = 2*hf1macro*f1macro/(hf1macro+f1macro);

	double LCA_F, LCA_P, LCA_R, MGIA;
	stringstream output;
	#ifdef OS_WINDOWS
		output << exec((string("HEMKit ")+"\""+hierarchical_classification_path+"\" \""+string(output_file_name)+".truetmp"+"\" \""+string(output_file_name)+".predtmp"+"\"").c_str());
	#else
		output << exec((string("./HEMKit ")+"\""+hierarchical_classification_path+"\" \""+string(output_file_name)+".truetmp"+"\" \""+string(output_file_name)+".predtmp"+"\"").c_str());
	#endif
	output >> LCA_F >> LCA_P >> LCA_R >> MGIA;

	remove((string(output_file_name)+".truetmp").c_str());
	remove((string(output_file_name)+".predtmp").c_str());

	stop = clock();
	testTime = (stop-start)/(double)CLOCKS_PER_SEC;

	fprintf(reportf, "Summary\n");
	fprintf(reportf, "Accuracy           : %g\n", accmacro);
	fprintf(reportf, "Accuracy(MGIA)     : %g\n", MGIA);
	fprintf(reportf, "Micro label-based Precision  : %g\n", precmicro);
	fprintf(reportf, "Macro label-based Precision  : %g\n", precmacro);
	fprintf(reportf, "Micro label-based Recall     : %g\n", reclmicro);
	fprintf(reportf, "Macro label-based Recall     : %g\n", reclmacro);
	fprintf(reportf, "Micro label-based F1Score    : %g\n", f1micro);
	fprintf(reportf, "Macro label-based F1Score    : %g\n", f1macro);
	fprintf(reportf, "Micro example-based Precision: %g\n", hprecmicro);
	fprintf(reportf, "Macro example-based Precision: %g\n", hprecmacro);
	fprintf(reportf, "Micro example-based Recall   : %g\n", hreclmicro);
	fprintf(reportf, "Macro example-based Recall   : %g\n", hreclmacro);
	fprintf(reportf, "Micro example-based F1Score  : %g\n", hf1micro);
	fprintf(reportf, "Macro example-based F1Score  : %g\n", hf1macro);
	fprintf(reportf, "Micro example-label-based Precision : %g\n", Elbprecmicro);
	fprintf(reportf, "Macro example-label-based Precision : %g\n", Elbprecmacro);
	fprintf(reportf, "Micro example-label-based Recall    : %g\n", Elbreclmicro);
	fprintf(reportf, "Macro example-label-based Recall    : %g\n", Elbreclmacro);
	fprintf(reportf, "Micro example-label-based F1Score   : %g\n", ElbF1micro);
	fprintf(reportf, "Macro example-label-based F1Score   : %g\n", ElbF1macro);
	fprintf(reportf, "LCA Precision      : %g\n", LCA_P);
	fprintf(reportf, "LCA Recall         : %g\n", LCA_R);
	fprintf(reportf, "LCA F1Score        : %g\n", LCA_F);
	fprintf(reportf, "Test time          : %g sec.\n", testTime);
	fprintf(reportf, "\n%s", buff.str().c_str());

	fclose(out);
	fclose(reportf);
	if(report_svm_score)
		fclose(scoref);
}

void predict(char *model_file_name, char *output_file_name)
{
	if(param.binary_classification)
		predict_binary(model_file_name, output_file_name);
	else if(param.hierarchical_classification)
		predict_hierarchical(model_file_name, output_file_name);
	else
			predict_multi(model_file_name, output_file_name);

	printf("Finished.\n");
	printf("Report file has been created to %s\n", (string(output_file_name)+".report").c_str());
	printf("Prediction file has been created to %s\n", output_file_name);
	if(report_svm_score)
		printf("SVM Score prediction file has been created to %s\n", (string(output_file_name)+".csv").c_str());
}

int main(int argc, char **argv)
{
	int i, j;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;

		switch(argv[i-1][1])
		{
			case 'b':
				predict_probability = atoi(argv[i]);
				break;

			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	if(i>=argc-2)
		exit_with_help();
				strcpy(input_file_name,argv[i]);

	strcpy(output_file_name,argv[i+2]);    
	strcpy(model_file_name,argv[i+1]);

	char *rr = strrchr(model_file_name, '/');
	if(rr == NULL)
		rr = model_file_name;
	else
		rr++;

	char str[1500];

	sprintf(str, "%s/%s", model_file_name, rr);
	strcpy(model_file_name, str);

	FILE *fm = fopen(model_file_name, "rb");
	if(fm == NULL)
	{
		fprintf(stderr, "error, file %s not found.\n", model_file_name);
		exit(1);
	}

	fscanf(fm, "%s", str); //get classification_type
	fscanf(fm, "%s", str); //get : character
	fscanf(fm, "%s", str); 

	param.hierarchical_classification = 0;
	param.multilabel_classification = 0;
	param.multiclass_classification = 0;
	param.binary_classification = 0;
	if(strcmp(str, "binary") == 0)
		param.binary_classification = 1;
	else if(strcmp(str, "multiclass") == 0)
		param.multiclass_classification = 1;
	else if(strcmp(str, "multilabel") == 0)
		param.multilabel_classification = 1;
	else if(strcmp(str, "hierarchical") == 0)
		param.hierarchical_classification = 1;
	
	if(param.binary_classification)
	{
		fscanf(fm, "%s", str); //get valid model
		fscanf(fm, "%s", str); //get : charaacter

		fscanf(fm, "%s", str);  
		string strlabel(str);
		classMap[strlabel] = 1;
		numMap[1] = strlabel;

		fscanf(fm, "%s", str);  
		strlabel = str;
		classMap[strlabel] = -1;
		numMap[-1] = strlabel;
		valid_model.push_back(1);
	}
	else if(param.multiclass_classification || param.multilabel_classification)
	{
		fscanf(fm, "%s", str); //get #model
		fscanf(fm, "%s", str); //get : character
		fscanf(fm, "%d", &class_no);
		fscanf(fm, "%s", str); //get model
		fscanf(fm, "%s", str); //get : character

		valid_model.clear();
		for(i=0;i<class_no;i++)
		{
			fscanf(fm, "%s", str);
			string strlabel(str);

			classMap[strlabel] = i;
			numMap[i] = strlabel;
			valid_model.push_back(i);
		}
	}
	else //hierarchical classification
	{
		fscanf(fm, "%s", str); //get #model
		fscanf(fm, "%s", str); //get : character
		fscanf(fm, "%d", &class_no);
		fscanf(fm, "%s", str); //get model
		fscanf(fm, "%s", str); //get : character

		valid_model.clear();
		for(i=0;i<class_no;i++)
		{
			fscanf(fm, "%s", str);
			string strlabel(str);

			classMap[strlabel] = i;
			numMap[i] = strlabel;
		}

		int numc, c;
		fscanf(fm, "%s", str); //get #valid model
		fscanf(fm, "%s", str); //get : character
		fscanf(fm, "%d", &numc);
		fscanf(fm, "%s", str); //get model
		fscanf(fm, "%s", str); //get : character

		for(i=0;i<numc;i++)
		{
			fscanf(fm, "%d", &c);
			valid_model.push_back(c);
		}

		sprintf(hierarchical_classification_path, "%s.hf", model_file_name);
	}

	fscanf(fm, "%s", str); //get RSVM
	fscanf(fm, "%s", str); //get : character
	fscanf(fm, "%d", &do_RSVM); 

	fscanf(fm, "%s", str); //get Feature_selection
	fscanf(fm, "%s", str); //get : character
	fscanf(fm, "%d", &do_feature_selection); 

	if(param.binary_classification)
		class_no = 1;
	selected_feature.resize(class_no);
	for(i=0;i<class_no;i++)
	{
		int size, feat, newfeat;
		fscanf(fm, "%d", &size);
		for(j=0;j<size;j++)
		{
			fscanf(fm, "%X", &feat);
			fscanf(fm, "%X", &newfeat);

			selected_feature[i][feat] = newfeat;
		}
	}

	if((int)valid_model.size() == 0)
		printf("WARNING: This model contains no classifiers. program exit\n");
	else
	{
		sort(valid_model.begin(), valid_model.end(), valid_model_cmp);
		read_problem(input_file_name);
		predict(model_file_name, output_file_name);
	}

	free(line);
	free(prob.y);
	free(prob.x);
	free(x_space);
	
	return 0;
}

