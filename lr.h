#ifndef _LIBLR_H
#define _LIBLR_H

#ifdef __cplusplus
extern "C" {
#endif

struct lr_node
{
	int index;
	double value;
};

struct lr_problem
{
	int l, n;
	int *y;
	struct lr_node **x;
	double bias;            /* < 0 if no bias term */  
};

struct lr_parameter
{
	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
};

struct lr_model
{
	struct lr_parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class (label[n]) */
	double bias;
};

struct lr_model* lr_train(const struct lr_problem *lrprob, const struct lr_parameter *param);
void lr_cross_validation(const struct lr_problem *prob, const struct lr_parameter *param, int nr_fold, int *target);

int lr_predict(const struct lr_model *model, const struct lr_node *x);
int lr_predict_probability(const struct lr_model *model, const struct lr_node *x, double* prob_estimates);

int lr_save_model(const char *model_file_name, const struct lr_model *model);
struct lr_model *lr_load_model(const char *model_file_name);

int lr_get_nr_feature(const struct lr_model *model);
int lr_get_nr_class(const struct lr_model *model);
void lr_get_labels(const struct lr_model *model, int* label);

void lr_destroy_model(struct lr_model *model);
void lr_destroy_param(struct lr_parameter *param);

const char *lr_check_parameter(const struct lr_problem *prob, const struct lr_parameter *param);


#ifdef __cplusplus
}
#endif

#endif /* _LIBLR_H */

