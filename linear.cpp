#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include "linear.h"
#include "tron.h"
typedef signed char schar;
template <class T> inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

#if 1
void info(const char *fmt,...)
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

class l2_lr_fun : public function
{
public:
	l2_lr_fun(const problem *prob, double Cp, double Cn);
	~l2_lr_fun();
	
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
	const problem *prob;
};

l2_lr_fun::l2_lr_fun(const problem *prob, double Cp, double Cn)
{
	int i;
	int l=prob->l;
	int *y=prob->y;

	this->prob = prob;

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

l2_lr_fun::~l2_lr_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
}


double l2_lr_fun::fun(double *w)
{
	int i;
	double f=0;
	int *y=prob->y;
	int l=prob->l;
	int n=prob->n;

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

void l2_lr_fun::grad(double *w, double *g)
{
	int i;
	int *y=prob->y;
	int l=prob->l;
	int n=prob->n;

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

int l2_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int n=prob->n;
	double *wa = new double[l];

	Xv(s, wa);
	for(i=0;i<l;i++)
		wa[i] = C[i]*D[i]*wa[i];

	XTv(wa, Hs);
	for(i=0;i<n;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void l2_lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int n=prob->n;
	feature_node **x=prob->x;

	for(i=0;i<n;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

class l2loss_svm_fun : public function
{
public:
	l2loss_svm_fun(const problem *prob, double Cp, double Cn);
	~l2loss_svm_fun();
	
	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void subXv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const problem *prob;
};

l2loss_svm_fun::l2loss_svm_fun(const problem *prob, double Cp, double Cn)
{
	int i;
	int l=prob->l;
	int *y=prob->y;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	C = new double[l];
	I = new int[l];

	for (i=0; i<l; i++)
	{
		if (y[i] == 1)
			C[i] = Cp;
		else
			C[i] = Cn;
	}
}

l2loss_svm_fun::~l2loss_svm_fun()
{
	delete[] z;
	delete[] D;
	delete[] C;
	delete[] I;
}

double l2loss_svm_fun::fun(double *w)
{
	int i;
	double f=0;
	int *y=prob->y;
	int l=prob->l;
	int n=prob->n;

	Xv(w, z);
	for(i=0;i<l;i++)
	{
	        z[i] = y[i]*z[i];
		double d = z[i]-1;
		if (d < 0)
			f += C[i]*d*d;
	}
	f = 2*f;
	for(i=0;i<n;i++)
		f += w[i]*w[i];
	f /= 2.0;

	return(f);
}

void l2loss_svm_fun::grad(double *w, double *g)
{
	int i;
	int *y=prob->y;
	int l=prob->l;
	int n=prob->n;

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<n;i++)
		g[i] = w[i] + 2*g[i];
}

int l2loss_svm_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2loss_svm_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int n=prob->n;
	double *wa = new double[l];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<n;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2loss_svm_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2loss_svm_fun::subXv(double *v, double *Xv)
{
	int i;
	feature_node **x=prob->x;

	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2loss_svm_fun::subXTv(double *v, double *XTv)
{
	int i;
	int n=prob->n;
	feature_node **x=prob->x;

	for(i=0;i<n;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

// A coordinate descent algorithm for 
// solving L1 loss and L2 loss SVM dual optimization problems
// Solves:
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= alpha_i <= upper_bound_i,
// 
//  where Q is y^TX^TXy and
//  D is a diagonal matrix with D_ii = 1/diag_i
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1		
// 		diag_i = INF
// In L2-Svm case:
// 		upper_bound_i = INF
// 		diag_i = 2*Cp	if y_i = 1
// 		diag_i = 2*Cn	if y_i = -1
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w

static void solve_linear_c_svc(
	const problem *prob, double *w, double eps, 
	double Cp, double Cn, int solver_type)
{
	int l = prob->l;
	int n = prob->n;
	int i, iter = 0;
	double C, d;
	double *QD = new double[l];
	double *G = new double[l];
	int max_iter = 2000;
	int *index = new int[l];
	double error_i, error;
	double *alpha = new double[l];
	schar *y = new schar[l];
	int active_size = l;

	// default solver_type: L2LOSS_SVM_DUAL
	double diag_p = 2*Cp, diag_n = 2*Cn;
	double upper_bound_p = INF, upper_bound_n = INF;
	if(solver_type == L1LOSS_SVM_DUAL)
	{
		diag_p = INF, diag_n = INF;
		upper_bound_p = Cp, upper_bound_n = Cn;
	}
	

	for(i=0; i<n; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		alpha[i] = 0;
		if(prob->y[i] > 0)
		{
			y[i] = +1; 
			QD[i] = 1/diag_p;
		}
		else
		{
			y[i] = -1;
			QD[i] = 1/diag_n;
		}

		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			QD[i] += (xi->value)*(xi->value);
			xi++;
		}
		index[i] = i;
	}

	while (iter < max_iter)
	{
		error = -INF;
		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(int k=0; k<active_size; k++)
		{
			i = index[k];
			G[i] = 0;
			schar yi = y[i];

			feature_node *xi = prob->x[i];
			while(xi->index!= -1)
			{
				G[i] += w[xi->index-1]*(xi->value);
				xi++;
			}
			G[i] = G[i]*yi-1;

			if(yi == 1)
			{
				C = upper_bound_p; 
				G[i] += alpha[i]/diag_p; 
			}
			else 
			{
				C = upper_bound_n;
				G[i] += alpha[i]/diag_n; 
			}
			error_i = fabs(min(max(alpha[i] - G[i], 0.0), C) - alpha[i]);
			error = max(error_i, error);
			if(error_i <= 1.0e-12)
				continue;
			else
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G[i]/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				xi = prob->x[i];
				while (xi->index != -1)
				{
					w[xi->index-1] += d*xi->value;
					xi++;
				}
			}
		}

		if(error <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*"); info_flush();
				continue;
			}
		}


		iter++;

		// shrinking
		if(iter % 5 == 0)
		{
			info("."); info_flush();

			for(int k = 0; k < active_size; k++)
			{
				i = index[k];
				if(y[i] == 1) C = upper_bound_p; else C = upper_bound_n;

				if(alpha[i] == 0 && G[i] > -100*eps)
				{
					active_size--;
					swap(index[k], index[active_size]);
				}
				else if( alpha[i] == C && G[i] < 100*eps)
				{
					active_size--;
					swap(index[k], index[active_size]);
				}
			}
		}
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("Warning: reaching max number of iterations\n");

	// calculate objective value
	
	double v = 0;
	for(int i=0; i<l; i++)
	{
		G[i] = 0;
		schar yi = y[i];
		feature_node *xi = prob->x[i];
		while(xi->index != -1)
		{
			G[i] += w[xi->index-1]*(xi->value);
			xi++;
		}
		G[i] = G[i]*yi;
		if(yi == 1) G[i] += alpha[i]/diag_p; else G[i] += alpha[i]/diag_n;
		v += alpha[i] * (G[i]-2);
	}
	info("Objective value = %lf\n",v/2);
	
	delete [] G;
	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
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

void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn)
{
	double eps=param->eps;
	int pos = 0;
	int neg = 0;
	for (int i=0; i<prob->l;i++)
		if (prob->y[i]==+1)
			pos++;
	neg = prob->l - pos;

	function *fun_obj=NULL;
	switch(param->solver_type)
	{
		case L2_LR:
		{
			fun_obj=new l2_lr_fun(prob, Cp, Cn);
			TRON tron_obj(fun_obj, eps*min(pos,neg)/prob->l);
			tron_obj.tron(w);
			delete fun_obj;
			break;
		}
		case L2LOSS_SVM:
		{
			fun_obj=new l2loss_svm_fun(prob, Cp, Cn);
			TRON tron_obj(fun_obj, eps*min(pos,neg)/prob->l);
			tron_obj.tron(w);
			delete fun_obj;
			break;
		}
		case L2LOSS_SVM_DUAL:
			solve_linear_c_svc(prob, w, eps, Cp, Cn, L2LOSS_SVM_DUAL);
			break;
		case L1LOSS_SVM_DUAL:
			solve_linear_c_svc(prob, w, eps, Cp, Cn, L1LOSS_SVM_DUAL);
			break;
		default:
			fprintf(stderr, "Error: unknown solver_type\n");
			break;
	}
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	int nr_class;
	int *label = NULL;
	int *start = NULL;
	int *count = NULL;
	int *perm = Malloc(int,l);

	// group training data of the same class
	group_classes(prob,&nr_class,&label,&start,&count,perm);

	model_->nr_class=nr_class;
	model_->label = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++)
		model_->label[i] = label[i];

	// calculate weighted C
	double *weighted_C = Malloc(double, nr_class);
	for(i=0;i<nr_class;i++)
		weighted_C[i] = param->C;
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

	// constructing the subproblem
	feature_node **x = Malloc(feature_node *,l);
	for(i=0;i<l;i++)
		x[i] = prob->x[perm[i]];

	int k;
	problem sub_prob;
	sub_prob.l = l;
	sub_prob.n = n;
	sub_prob.x = Malloc(feature_node *,sub_prob.l);
	sub_prob.y = Malloc(int,sub_prob.l);

	for(k=0; k<sub_prob.l; k++)
		sub_prob.x[k] = x[k];

	if(nr_class==2)
	{
		model_->w=Malloc(double, n);

		int e0 = start[0]+count[0];
		k=0;
		for(; k<e0; k++)
			sub_prob.y[k] = +1;
		for(; k<sub_prob.l; k++)
			sub_prob.y[k] = -1;

		train_one(&sub_prob, param, &model_->w[0], weighted_C[0], weighted_C[1]);
	}
	else
	{
		model_->w=Malloc(double, n*nr_class);
		for(i=0;i<nr_class;i++)
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

			train_one(&sub_prob, param, &model_->w[i*n], weighted_C[i], param->C);
		}
	}

	free(x);
	free(label);
	free(start);
	free(count);
	free(perm);
	free(sub_prob.x);
	free(sub_prob.y);
	free(weighted_C);
	return model_;
}

void destroy_model(struct model *model_)
{
	if(model_->w != NULL)
		free(model_->w);
	if(model_->label != NULL)
		free(model_->label);
	free(model_);
}

const char *solver_type_table[]=
{
	"L2_LR", "L2LOSS_SVM_DUAL", "L2LOSS_SVM","L1LOSS_SVM_DUAL", NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	int nr_classifier;
	if(model_->nr_class==2)
		nr_classifier=1;
	else
		nr_classifier=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);
	fprintf(fp, "label");
	for(i=0; i<model_->nr_class; i++)
		fprintf(fp, " %d", model_->label[i]);
	fprintf(fp, "\n");

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_classifier; j++)
			fprintf(fp, "%.16g ", model_->w[j*n+i]);
		fprintf(fp, "\n");
	}

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				free(model_->label);
				free(model_);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free(model_);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	int nr_classifier;
	if(nr_class==2)
		nr_classifier = 1;
	else
		nr_classifier = nr_class;

	model_->w=Malloc(double, n*nr_classifier);
	for(i=0; i<n; i++)
	{
		int j;
		for(j=0; j<nr_classifier; j++)
			fscanf(fp, "%lf ", &model_->w[j*n+i]);
		fscanf(fp, "\n");
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_classifier;
	if(nr_class==2)
		nr_classifier = 1;
	else
		nr_classifier = nr_class;
	for(i=0;i<nr_classifier;i++)
	{
		const feature_node *lx=x;
		double wtx=0;
		for(; (idx=lx->index)!=-1; lx++)
		{
			// the dimension of testing data may exceed that of training
			if(idx<=n)
				wtx += w[i*n+idx-1]*lx->value;
		}

		dec_values[i] = wtx;
	}

	if(nr_class==2)
		return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

int predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	int label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

int predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(model_->param.solver_type==L2_LR)
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_classifier;
		if(nr_class==2)
			nr_classifier = 1;
		else
			nr_classifier = nr_class;

		int label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_classifier;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

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

		return label;		
	}
	else
		return 0;
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->solver_type != L2_LR
	   && param->solver_type != L2LOSS_SVM_DUAL
	   && param->solver_type != L2LOSS_SVM
	   && param->solver_type != L1LOSS_SVM_DUAL)
		return "unknown solver type";

//	if(param->solver_type == L1_LR)
//		return "sorry! sover_type = 1 (L1_LR) is not supported yet";

	return NULL;
}

void cross_validation(const problem *prob, const parameter *param, int nr_fold, int *target)
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
		struct problem subprob;

		subprob.bias = prob->bias;
		subprob.n = prob->n;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
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
		struct model *submodel = train(&subprob,param);
		for(j=begin;j<end;j++)
			target[perm[j]] = predict(submodel,prob->x[perm[j]]);
		destroy_model(submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

