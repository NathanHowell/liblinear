#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "lr.h"

#include "mex.h"
#include "lr_model_matlab.h"

#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void exit_with_help()
{
	mexPrintf(
	"Usage: model = lrtrain(training_label_vector, training_instance_matrix, 'liblr_options');\n"
	);
}

// lr arguments
struct lr_parameter param;		// set by parse_command_line
struct lr_problem lrprob;		// set by read_problem
struct lr_model *model;
struct lr_node *x_space;
struct lr_parameter lrparam;		// set by parse_command_line
int cross_validation;
int nr_fold;
double bias=1.;

double do_cross_validation()
{
	int i;
	int total_correct = 0;
	int *target = Malloc(int,lrprob.l);
	double retval = 0.0;

	lr_cross_validation(&lrprob,&param,nr_fold,target);

	for(i=0;i<lrprob.l;i++)
		if(target[i] == lrprob.y[i])
			++total_correct;
	mexPrintf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/lrprob.l);
	retval = 100.0*total_correct/lrprob.l;

	free(target);
	return retval;
}

// nrhs should be 3
int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name)
{
	int i, argc = 1;
	char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];

	// default values
	param.C = 1;
	param.eps = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation = 0;

	if(nrhs <= 1)
		return 1;
	if(nrhs == 2)
		return 0;

	// put options in argv[]
	mxGetString(prhs[2], cmd,  mxGetN(prhs[2]) + 1);
	if((argv[argc] = strtok(cmd, " ")) == NULL)
		return 0;
	while((argv[++argc] = strtok(NULL, " ")) != NULL)
		;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			return 1;
		switch(argv[i-1][1])
		{
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'B':
				bias = atof(argv[i]);
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					mexPrintf("n-fold cross validation: n must >= 2\n");
					return 1;
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
				mexPrintf("unknown option\n");
				return 1;
		}
	}
	return 0;
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

int read_problem_sparse(const mxArray *label_vec, const mxArray *instance_mat)
{
	int i, j, k, low, high;
	mwIndex *ir, *jc;
	int elements, max_index, num_samples;
	double *samples, *labels;
	mxArray *instance_mat_tr; // transposed instance sparse matrix

	// transpose instance matrix
	{
		mxArray *prhs[1], *plhs[1];
		prhs[0] = mxDuplicateArray(instance_mat);
		if(mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose training instance matrix\n");
			return -1;
		}
		instance_mat_tr = plhs[0];
		mxDestroyArray(prhs[0]);
	}

	// each column is one instance
	labels = mxGetPr(label_vec);
	samples = mxGetPr(instance_mat_tr);
	ir = mxGetIr(instance_mat_tr);
	jc = mxGetJc(instance_mat_tr);

	num_samples = mxGetNzmax(instance_mat_tr);

	// the number of instance
	lrprob.l = mxGetN(instance_mat_tr);
	elements = num_samples + lrprob.l*2;
	max_index = mxGetM(instance_mat_tr);

	lrprob.y = Malloc(int, lrprob.l);
	lrprob.x = Malloc(struct lr_node*, lrprob.l);
	x_space = Malloc(struct lr_node, elements);

	lrprob.bias=bias;

	j = 0;
	for(i=0;i<lrprob.l;i++)
	{
		lrprob.x[i] = &x_space[j];
		lrprob.y[i] = (int)labels[i];
		low = jc[i], high = jc[i+1];
		for(k=low;k<high;k++)
		{
			x_space[j].index = ir[k]+1;
			x_space[j].value = samples[k];
			j++;
	 	}
		if(lrprob.bias>=0)
		{
			x_space[j].index = max_index+1;
			x_space[j].value = lrprob.bias;
			j++;
		}
		x_space[j++].index = -1;
	}

	if(lrprob.bias>=0)
		lrprob.n = max_index+1;

	return 0;
}

// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	const char *error_msg;
	// fix random seed to have same results for each run
	// (for cross validation and probability estimation)
	srand(1);

	// Translate the input Matrix to the format such that lrtrain.exe can recognize it
	if(nrhs > 0 && nrhs < 4)
	{
		if(parse_command_line(nrhs, prhs, NULL))
		{
			exit_with_help();
			lr_destroy_param(&param);
			fake_answer(plhs);
			return;
		}

		if(mxIsSparse(prhs[1]))
			read_problem_sparse(prhs[0], prhs[1]);
		else
			mexPrintf("Training_instance_matrix must be sparse\n");

		// lrtrain's original code
		error_msg = lr_check_parameter(&lrprob, &param);

		if(error_msg)
		{
			if (error_msg != NULL)
				mexPrintf("Error: %s\n", error_msg);
			lr_destroy_param(&param);
			fake_answer(plhs);
			return;
		}

		if(cross_validation)
		{
			double *ptr;
			plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
			ptr = mxGetPr(plhs[0]);
			ptr[0] = do_cross_validation();
		}
		else
		{
			int nr_feat = mxGetM(prhs[1]);
			const char *error_msg;
			model = lr_train(&lrprob, &param);
			error_msg = model_to_matlab_structure(plhs, nr_feat, model);
			if(error_msg)
				mexPrintf("Error: can't convert libsvm model to matrix structure: %s\n", error_msg);
			lr_destroy_model(model);
		}
		lr_destroy_param(&param);
	}
	else
	{
		exit_with_help();
		fake_answer(plhs);
		return;
	}
}

