#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include "lr.h"

#ifndef min
template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
#endif

template <class T> inline void swap(T& x, T& y) { T t=x; x=y; y=t; }

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#if 1
void info(char *fmt,...)
{
	va_list ap;
	va_start(ap,fmt);
	vprintf(fmt,ap);
	va_end(ap);
}
void info_flush()
{
	fflush(stdout);
}
#else
void info(char *fmt,...) {}
void info_flush() {}
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);

#ifdef __cplusplus
}
#endif

class function
{
public:
	virtual double fun(double *w) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void Hv(double *s, double *Hs) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual ~function(void){}
};

class lr_fun : public function
{
public:
	lr_fun(const lr_problem *lrprob, double Cp, double Cn);
	~lr_fun();
	
	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	const lr_problem *lrprob;
};


lr_fun::lr_fun(const lr_problem *lrprob, double Cp, double Cn)
{
	int i;
	int l=lrprob->l;
	int *y=lrprob->y;

	this->lrprob = lrprob;

	z = new double[l];
	D = new double[l];
	C = new double[l];

	for (i=0; i<l; i++)
	{
		if (y[i] == 1)
			C[i] = Cp;
		else
			C[i] = Cn;
	}
}

lr_fun::~lr_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
}


double lr_fun::fun(double *w)
{
	int i;
	double f=0;
	int *y=lrprob->y;
	int l=lrprob->l;
	int n=lrprob->n;

	Xv(w, z);
	for(i=0;i<l;i++)
	{
	        double yz = y[i]*z[i];
		if (yz >= 0)
		        f += C[i]*log(1 + exp(-yz));
		else
		        f += C[i]*(-yz+log(1 + exp(yz)));
	}
	f = 2*f;
	for(i=0;i<n;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

void lr_fun::grad(double *w, double *g)
{
	int i;
	int *y=lrprob->y;
	int l=lrprob->l;
	int n=lrprob->n;

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

	for(i=0;i<n;i++)
		g[i] = w[i] + g[i];
}

int lr_fun::get_nr_variable(void)
{
	return lrprob->n;
}

void lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=lrprob->l;
	int n=lrprob->n;
	double *wa = new double[l];

	for(i=0;i<l;i++) wa[i]=0;
	Xv(s, wa);
	for(i=0;i<l;i++)
		wa[i] = C[i]*D[i]*wa[i];

	XTv(wa, Hs);
	for(i=0;i<n;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=lrprob->l;
	lr_node **x=lrprob->x;

	for(i=0;i<l;i++)
	{
		lr_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=lrprob->l;
	int n=lrprob->n;
	lr_node **x=lrprob->x;

	for(i=0;i<n;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
	{
		lr_node *s=x[i];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}


class TRON
{
public:
	TRON(const function *fun_obj, double eps = 0.1, int max_iter = 1000);
	~TRON();

	void tron(double *w);

private:
	int trcg(double delta, double *g, double *s, double *r);
	double norm_inf(int n, double *x);

	double eps;
	int max_iter;
	function *fun_obj;
};

TRON::TRON(const function *fun_obj, double eps, int max_iter)
{
	this->fun_obj=const_cast<function *>(fun_obj);
	this->eps=eps;
	this->max_iter=max_iter;
}

TRON::~TRON()
{
}

void TRON::tron(double *w)
{
	// Parameters for updating the iterates.
	double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

	// Parameters for updating the trust region size delta.
	double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	double delta, snorm, one=1.0;
	double alpha, f, fnew, prered, actred, gs;
	int search = 1, iter = 1, inc = 1;
	double *s = new double[n];
	double *r = new double[n];
	double *w_new = new double[n];
	double *g = new double[n];

	for (i=0; i<n; i++)
		w[i] = 0;

        f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	delta = dnrm2_(&n, g, &inc);

	if (norm_inf(n, g) < eps)
		search = 0;

	iter = 1;

	while (iter <= max_iter && search)
	{
		cg_iter = trcg(delta, g, s, r);

		memcpy(w_new, w, sizeof(double)*n);
		daxpy_(&n, &one, s, &inc, w_new, &inc);

		gs = ddot_(&n, g, &inc, s, &inc);
		prered = -0.5*(gs-ddot_(&n, s, &inc, r, &inc));
                fnew = fun_obj->fun(w_new);

		// Compute the actual reduction.
	        actred = f - fnew;

		// On the first iteration, adjust the initial step bound.
		snorm = dnrm2_(&n, s, &inc);
		if (iter == 1)
			delta = min(delta, snorm);

		// Compute prediction alpha*snorm of the step.
		if (fnew - f - gs <= 0)
			alpha = sigma3;
		else
			alpha = max(sigma1, -0.5*(gs/(fnew - f - gs)));

		// Update the trust region bound according to the ratio of actual to predicted reduction.
		if (actred < eta0*prered)
			delta = min(max(alpha, sigma1)*snorm, sigma2*delta);
		else if (actred < eta1*prered)
			delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta));
		else if (actred < eta2*prered)
			delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta));
		else
			delta = max(delta, min(alpha*snorm, sigma3*delta));

		info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, norm_inf(n, g), cg_iter);

		if (actred > eta0*prered)
		{
			iter++;
			memcpy(w, w_new, sizeof(double)*n);
			f = fnew;
		        fun_obj->grad(w, g);

			if (norm_inf(n, g) < eps)
				break;
		}
		if (f < 1.0e-32)
		{
			info("warning: f < 1.0e-32\n");
			break;
		}
		if (fabs(actred) <= 0 && fabs(prered) <= 0)
		{
			info("warning: actred and prered <= 0\n");
			break;
		}
		if (fabs(actred) <= 1.0e-12*fabs(f) &&
		    fabs(prered) <= 1.0e-12*fabs(f))
		{
			info("warning: actred and prered too small\n");
			break;
		}
	}

	delete[] g;
	delete[] r;
	delete[] w_new;
	delete[] s;
}

int TRON::trcg(double delta, double *g, double *s, double *r)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	double one = 1;
	double *d = new double[n];
	double *Hd = new double[n];
	double rTr, rnewTrnew, alpha, beta, cgtol;

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		d[i] = r[i];
	}
	cgtol = 0.1*dnrm2_(&n, g, &inc);

	int cg_iter = 0;
	rTr = ddot_(&n, r, &inc, r, &inc);
	while (1)
	{
		if (dnrm2_(&n, r, &inc) <= cgtol)
			break;
		cg_iter++;
		fun_obj->Hv(d, Hd);

		alpha = rTr/ddot_(&n, d, &inc, Hd, &inc);
		daxpy_(&n, &alpha, d, &inc, s, &inc);
		if (dnrm2_(&n, s, &inc) > delta)
		{
			info("cg reaches trust region boundary\n");
			alpha = -alpha;
			daxpy_(&n, &alpha, d, &inc, s, &inc);

			double std = ddot_(&n, s, &inc, d, &inc);
			double sts = ddot_(&n, s, &inc, s, &inc);
			double dtd = ddot_(&n, d, &inc, d, &inc);
			double dsq = delta*delta;
			double rad = sqrt(std*std + dtd*(dsq-sts));
			if (std >= 0)
				alpha = (dsq - sts)/(std + rad);
			else
				alpha = (rad - std)/dtd;
			daxpy_(&n, &alpha, d, &inc, s, &inc);
			alpha = -alpha;
			daxpy_(&n, &alpha, Hd, &inc, r, &inc);
			break;
		}
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);
		rnewTrnew = ddot_(&n, r, &inc, r, &inc);
		beta = rnewTrnew/rTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, r, &inc, d, &inc);
		rTr = rnewTrnew;
	}

	delete[] d;
	delete[] Hd;

	return(cg_iter);
}

double TRON::norm_inf(int n, double *x)
{
	double dmax = fabs(x[0]);
	for (int i=1; i<n; i++)
		if (fabs(x[i]) >= dmax)
			dmax = fabs(x[i]);
	return(dmax);
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
void lr_group_classes(const lr_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

void lr_train_one(const lr_problem *lrprob, const lr_parameter *param, double *w, double Cp, double Cn)
{
	double eps=param->eps;
	info("eps %f Cp %f Cn %f\n", eps, Cp, Cn);
	lr_fun fun_obj(lrprob, Cp, Cn);
	TRON tron_obj(&fun_obj, eps);

	tron_obj.tron(w);
}

//
// Interface functions
//
lr_model* lr_train(const lr_problem *lrprob, const lr_parameter *param)
{
	int i;
	int l = lrprob->l;
	int n = lrprob->n;
	lr_model *model = Malloc(lr_model,1);

	if(lrprob->bias>=0)
		model->nr_feature=n-1;
	else
		model->nr_feature=n;
	model->param = *param;
	model->bias = lrprob->bias;

	int nr_class;
	int *label = NULL;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int,l);
	double Cp, Cn;

	// group training data of the same class
	lr_group_classes(lrprob,&nr_class,&label,&start,&count,perm);

	model->nr_class=nr_class;
	model->label = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++)
		model->label[i] = label[i];

	// calculate weighted C
	double *weighted_C = Malloc(double, nr_class);
	for(i=0;i<nr_class;i++)
		weighted_C[i] = 1;
	for(i=0;i<param->nr_weight;i++)
	{
		int j;
		for(j=0;j<nr_class;j++)
			if(param->weight_label[i] == label[j])
				break;
		if(j == nr_class)
			fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
		else
			weighted_C[j] *= param->weight[i];
	}

	int nr_classifier;
	if(nr_class==2)
		nr_classifier = 1;
	else
		nr_classifier = nr_class;

	model->w=Malloc(double, n*nr_classifier);
	// constructing the subproblem
	lr_node **x = Malloc(lr_node *,l);
	for(i=0;i<l;i++)
		x[i] = lrprob->x[perm[i]];

	int k;
	lr_problem sub_prob;
	sub_prob.l = l;
	sub_prob.n = n;
	sub_prob.x = Malloc(lr_node *,sub_prob.l);
	sub_prob.y = Malloc(int,sub_prob.l);

	for(k=0; k<sub_prob.l; k++)
		sub_prob.x[k] = x[k];

	for(i=0;i<nr_classifier;i++)
	{
		int si = start[i];
		int ei = si+count[i];

		k=0;
		for(; k<si; k++)
			sub_prob.y[k] = -1;
		for(; k<ei; k++)
			sub_prob.y[k] = +1;
		for(; k<sub_prob.l; k++)
			sub_prob.y[k] = -1;

		Cp = param->C;
		Cn = param->C;

		lr_train_one(&sub_prob, param, &model->w[i*n], Cp*weighted_C[i], Cn*1.);
	}

	free(x);
	free(label);
	free(start);
	free(count);
	free(perm);
	free(sub_prob.x);
	free(sub_prob.y);
	free(weighted_C);
	return model;
}

void lr_destroy_model(struct lr_model *model)
{
	if(model->w != NULL)
		free(model->w);
	if(model->label != NULL)
		free(model->label);
}

int lr_save_model(const char *model_file_name, const struct lr_model *model)
{
	int i;
	int nr_feature=model->nr_feature;
	int n;
	if(model->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	int nr_classifier;
	if(model->nr_class==2)
		nr_classifier=1;
	else
		nr_classifier=model->nr_class;

	fprintf(fp, "nr_class %d\n", model->nr_class);
	fprintf(fp, "label");
	for(i=0; i<model->nr_class; i++)
		fprintf(fp, " %d", model->label[i]);
	fprintf(fp, "\n");

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model->bias);

	fprintf(fp, "w\n");
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_classifier; j++)
			fprintf(fp, "%.16g ", model->w[j*n+i]);
		fprintf(fp, "\n");
	}

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct lr_model *lr_load_model(const char *model_file_name)
{
	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	lr_model *model = Malloc(lr_model,1);
	FILE *fp = fopen(model_file_name,"r");

	model->label = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model->nr_class;
			model->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model);
			return NULL;
		}
	}

	nr_feature=model->nr_feature;
	if(model->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	int nr_classifier;
	if(nr_class==2)
		nr_classifier = 1;
	else
		nr_classifier = nr_class;

	model->w=Malloc(double, n*nr_classifier);
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_classifier; j++)
			fscanf(fp, "%lf ", &model->w[j*n+i]);
		fscanf(fp, "\n");
	}
	return model;
}

int lr_predict(const lr_model *model, const lr_node *x)
{
	double *prob_estimates = Malloc(double, model->nr_class);
	int label=lr_predict_probability(model, x, prob_estimates);
	free(prob_estimates);
	return label;
}

int lr_predict_probability(const struct lr_model *model, const struct lr_node *x, double* prob_estimates)
{
	int idx;
	int n;
	if(model->bias>=0)
		n=model->nr_feature+1;
	else
		n=model->nr_feature;
	double *w=model->w;
	int nr_class=model->nr_class;
	int i;
	int nr_classifier;
	if(nr_class==2)
		nr_classifier = 1;
	else
		nr_classifier = nr_class;
	for(i=0;i<nr_classifier;i++)
	{
		const lr_node *lx=x;
		double wtx=0;
		for(; (idx=lx->index)!=-1; lx++)
		{
			// the dimension of testing data may exceed that of training
			if(idx<=n)
				wtx += w[i*n+idx-1]*lx->value;
		}

		prob_estimates[i] = 1/(1+exp(-wtx));
	}

	if(nr_class==2) // for binary classification
		prob_estimates[1]=1.-prob_estimates[0];
	else
	{
		double sum=0;
		for(i=0; i<nr_class; i++)
			sum+=prob_estimates[i];

		for(i=0; i<nr_class; i++)
			prob_estimates[i]=prob_estimates[i]/sum;
	}

	int dec_max_idx = 0;
	for(i=1;i<nr_class;i++)
	{
		if(prob_estimates[i] > prob_estimates[dec_max_idx])
			dec_max_idx = i;
	}
	return model->label[dec_max_idx];
}

void lr_destroy_param(lr_parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *lr_check_parameter(const lr_problem *prob, const lr_parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	return NULL;
}

void lr_cross_validation(const lr_problem *prob, const lr_parameter *param, int nr_fold, int *target)
{
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int *perm = Malloc(int,l);

	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct lr_problem subprob;

		subprob.n = prob->n;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct lr_node*,subprob.l);
		subprob.y = Malloc(int,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct lr_model *submodel = lr_train(&subprob,param);
		for(j=begin;j<end;j++)
			target[perm[j]] = lr_predict(submodel,prob->x[perm[j]]);
		lr_destroy_model(submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

int lr_get_nr_feature(const lr_model *model)
{
	return model->nr_feature;
}

int lr_get_nr_class(const lr_model *model)
{
	return model->nr_class;
}

void lr_get_labels(const lr_model *model, int* label)
{
	if (model->label != NULL)
		for(int i=0;i<model->nr_class;i++)
			label[i] = model->label[i];
}

