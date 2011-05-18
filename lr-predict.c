#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "lr.h"

char* line;
int max_line_len = 1024;
struct lr_node *x;
int max_nr_attr = 64;

struct lr_model* model;
int predict_probability=0;

void predict(FILE *input, FILE *output, struct lr_model* model)
{
	int correct = 0;
	int total = 0;

	int nr_class=lr_get_nr_class(model);
	double *prob_estimates=NULL;
	int j, n;
	int nr_feature=lr_get_nr_feature(model);
	if(model->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	if(predict_probability)
	{
		int *labels=(int *) malloc(nr_class*sizeof(int));
		lr_get_labels(model,labels);
		prob_estimates = (double *) malloc(nr_class*sizeof(double));
		fprintf(output,"labels");		
		for(j=0;j<nr_class;j++)
			fprintf(output," %d",labels[j]);
		fprintf(output,"\n");
		free(labels);
	}
	while(1)
	{
		int i = 0;
		int c;
		double target;
		int target_label, predict_label;

		if (fscanf(input,"%lf",&target)==EOF)
			break;
		target_label=(int)target;

		while(1)
		{
			if(i>=max_nr_attr-2)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct lr_node *) realloc(x,max_nr_attr*sizeof(struct lr_node));
			}

			do {
				c = getc(input);
				if(c=='\n' || c==EOF) goto out2;
			} while(isspace(c));
			ungetc(c,input);
			if (fscanf(input,"%d:%lf",&x[i].index,&x[i].value) < 2)
			{
				fprintf(stderr,"Wrong input format at line %d\n", total+1);
				exit(1);
			}
			// feature indices larger than those in training are not used
			if(x[i].index<=nr_feature)
				++i;
		}

out2:
		if(model->bias>=0)
		{
			x[i].index = n;
			x[i].value = model->bias;
			i++;
		}
		x[i].index = -1;

		if(predict_probability)
		{
			int j;
			predict_label = lr_predict_probability(model,x,prob_estimates);
			fprintf(output,"%d ",predict_label);
			for(j=0;j<model->nr_class;j++)
				fprintf(output,"%g ",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else
		{
			predict_label = lr_predict(model,x);
			fprintf(output,"%d\n",predict_label);
		}

		if(predict_label == target_label)
			++correct;
		++total;
	}
	printf("Accuracy = %g%% (%d/%d)\n", (double)correct/total*100,correct,total);
	if(predict_probability)
		free(prob_estimates);
}

void exit_with_help()
{
	printf(
	"Usage: lr-predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0)\n"
	);
	exit(1);
}

int main(int argc, char **argv)
{
	FILE *input, *output;
	int i;

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

			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
				break;
		}
	}
	if(i>=argc)
		exit_with_help();

	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

	if((model=lr_load_model(argv[i+1]))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		exit(1);
	}

	line = (char *) malloc(max_line_len*sizeof(char));
	x = (struct lr_node *) malloc(max_nr_attr*sizeof(struct lr_node));
	predict(input, output, model);
	lr_destroy_model(model);
	free(line);
	free(x);
	fclose(input);
	fclose(output);
	return 0;
}

