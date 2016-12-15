/***
C++ implementation of the value dynamic programming algorithm and
first passage time probability density computations

Author: Luciano Paz
Year: 2016
***/
#ifndef __DecisionModel
#define __DecisionModel

#include <cmath>
#include <cstddef>
#include <iostream>
#include <cstdio>

#ifdef DEBUG
#ifndef INFO
#define INFO
#endif
#endif

#define SIGN(x) ((x > 0) - (x < 0))

inline double normcdf(double x, double mu, double sigma){
	if (sigma==0.){
		return x>mu ? INFINITY : -INFINITY;
	}
	return 0.5 + 0.5*erf((x-mu)/sigma*0.70710678118654746);
}

inline double erfinv(double y) {
	double x,z;
	if (y<-1. || y>1.){
		// raise ValueError("erfinv(y) argument out of range [-1.,1]")
		return NAN;
	}
	if (y==1. || y==-1.){
		// Precision limit of erf function
		x = y*5.9215871957945083;
	} else if (y<-0.7){
		z = sqrt(-log(0.5*(1.0+y)));
		x = -(((1.641345311*z+3.429567803)*z-1.624906493)*z-1.970840454)/((1.637067800*z+3.543889200)*z+1.0);
	} else {
		if (y<0.7){
			z = y*y;
			x = y*(((-0.140543331*z+0.914624893)*z-1.645349621)*z+0.886226899)/((((0.012229801*z-0.329097515)*z+1.442710462)*z-2.118377725)*z+1.0);
		} else {
			z = sqrt(-log(0.5*(1.0-y)));
			x = (((1.641345311*z+3.429567803)*z-1.624906493)*z-1.970840454)/((1.637067800*z+3.543889200)*z+1.0);
		}
		// Polish to full accuracy
	}
	x-= (erf(x) - y) / (1.128379167 * exp(-x*x));
	x-= (erf(x) - y) / (1.128379167 * exp(-x*x));
	return x;
}

inline double normcdfinv(double y, double mu, double sigma){
	if (sigma==0.){
		return NAN;
	}
	return 1.4142135623730951*sigma*erfinv(2.*(y-0.5))+mu;
}

class DecisionModelDescriptor {
protected:
	bool _owns_cost;
	bool _known_variance;
	bool _conjugate_mu_prior;
public:
	const inline bool owns_cost(){return _owns_cost;};
	const inline bool known_variance(){return _known_variance;};
	const inline bool conjugate_mu_prior(){return _conjugate_mu_prior;};
	
	int n_model_var;
	double* model_var;
	double* prior_var_prob;
	
	double prior_mu_mean;
	double prior_mu_var;
	
	int n;
	double dt;
	int nT;
	double T;
	double reward;
	double penalty;
	double iti;
	double tp;
	double* cost;
	
	int n_prior;
	double *mu_prior;
	double *weight_prior;
	
	DecisionModelDescriptor(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, bool owns_cost);
	DecisionModelDescriptor(double model_var, int n_prior, double* mu_prior, double* weight_prior,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, bool owns_cost);
	DecisionModelDescriptor(int n_model_var, double* model_var,
				   double* prior_var_prob, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, bool owns_cost);
	
	~DecisionModelDescriptor();
	
	void disp();
};

class DecisionModel {
public:
	bool owns_bounds;
	bool known_variance;
	bool conjugate_mu_prior;
	
	int n;
	int nT;
	int bound_strides;
	
	double prior_mu_mean;
	double prior_mu_var;
	double dt;
	double dg;
	double T;
	double* cost;
	double reward;
	double penalty;
	double iti;
	double tp;
	double rho;
	
	double *g;
	double *t;
	double *ub;
	double *lb;
	
	DecisionModel(bool known_variance, bool conjugate_mu_prior,
				   double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost);
	DecisionModel(bool known_variance, bool conjugate_mu_prior,
				   double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, double* ub, double* lb,int bound_strides);
	virtual ~DecisionModel();
	
	static DecisionModel* create(DecisionModelDescriptor& dpc);
	static DecisionModel* create(DecisionModelDescriptor& dpc, double* ub, double* lb, int bound_strides);
	
	virtual void disp();
	
	virtual inline double x2g(const double t, const double x){return NAN;};
	virtual inline double dx2g(const double t, const double x){return NAN;};
	virtual inline double g2x(const double t, const double g){return NAN;};
	virtual inline double x_transition_probability(const double t, const double xt, const double xtdt){return NAN;};
	virtual inline double g_transition_probability(const double t, const double xt, const double xtdt){return NAN;};
	
	double backpropagate_value(){return this->backpropagate_value(this->rho,true);};
	virtual double backpropagate_value(double rho, bool compute_bounds){return NAN;};
	virtual double backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2){return NAN;};
	double value_for_root_finding(double rho);
	double iterate_rho_value(double tolerance);
	double iterate_rho_value(double tolerance, double lower_bound, double upper_bound);
	
	double* x_ubound();
	void x_ubound(double* xb);
	double* x_lbound();
	void x_lbound(double* xb);
	double Psi(double mu, double model_var, double* bound, int itp, double tp, double x0, double t0);
	void rt(double mu, double model_var, double* g1, double* g2, double* xub, double* xlb);
	void fpt_conf_matrix(double* first_passage_time, int* first_passage_time_strides, int n_alternatives, int confidence_partition, double* confidence_response, int* confidence_response_strides, double* out);
};

class DecisionModelConjPrior : public DecisionModel {
public:
	double model_var;
	
	DecisionModelConjPrior(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost):
			DecisionModel(true, true, prior_mu_mean, prior_mu_var, n, dt, T, reward, penalty, iti, tp, cost){
				this->model_var = model_var;
			};
	DecisionModelConjPrior(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, double* ub, double* lb,int bound_strides):
			DecisionModel(true, true, prior_mu_mean, prior_mu_var, n, dt, T, reward, penalty, iti, tp, cost, ub, lb, bound_strides){
				this->model_var = model_var;
			};
	~DecisionModelConjPrior();
	
	void disp();
	
	inline double post_mu_var(const double t){
		return 1./(t/this->model_var + 1./this->prior_mu_var);
	}
	
	inline double post_mu_mean(const double t, const double x){
		return (x/this->model_var+this->prior_mu_mean/this->prior_mu_var)*this->post_mu_var(t);
	}
	
	inline double x2g(const double t, const double x){
		return normcdf(this->post_mu_mean(t,x)/sqrt(this->post_mu_var(t)),0.,1.);
	}
	
	inline double dx2g(const double t, const double x){
		double vt = post_mu_var(t);
		return exp(-0.5*pow(post_mu_mean(t,x),2)/vt) * 0.3989422804014327* sqrt(vt) / this->model_var;
	}
	
	inline double g2x(const double t, const double g){
		return this->model_var*(normcdfinv(g,0.,1.)/sqrt(this->post_mu_var(t))-this->prior_mu_mean/this->prior_mu_var);
	}
	
	inline double x_transition_probability(const double t, const double xt, const double xtdt){
		const double inv_var = 1./((post_mu_var(t)*dt+model_var)*dt);
		return 0.3989422804014327*sqrt(inv_var)*exp(-0.5*inv_var*pow(xtdt-xt-post_mu_mean(t,xt),2));
	}
	
	inline double g_transition_probability(const double t, const double xt, const double xtdt){
		double vt = post_mu_var(t);
		double vtdt = post_mu_var(t+dt);
		return exp(-0.5*pow(xtdt-xt-post_mu_mean(t,xt)*dt,2)/((vt*dt+model_var)*dt)+0.5*vtdt*pow(xtdt/model_var+prior_mu_mean/prior_mu_var,2));
	}
	
	double backpropagate_value(double rho, bool compute_bounds);
	double backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2);
};

class DecisionModelDiscretePrior : public DecisionModel {
protected:
	int n_prior;
	bool is_prior_set;
	double* mu_prior;
	double* mu2_prior;
	double* weight_prior;
	double epsilon;
	double g2x_lower_bound;
	double g2x_upper_bound;
	double g2x_tolerance;
public:
	double model_var;
	
	DecisionModelDiscretePrior(double model_var, int n_prior,double* mu_prior, double* weight_prior,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost):
		DecisionModel(true, false, 0., 0., n, dt, T, reward, penalty, iti, tp, cost)
	{
		/***
		 * Constructor that shares its bound arrays
		***/
		is_prior_set = false;
		this->model_var = model_var;
		this->epsilon = 1e-10;
		this->set_prior(n_prior,mu_prior,weight_prior);
		this->g2x_tolerance = 1e-12;
		#ifdef DEBUG
		std::cout<<"Created DecisionModelDiscretePrior instance at "<<this<<std::endl;
		#endif
	};
	DecisionModelDiscretePrior(double model_var, int n_prior,double* mu_prior, double* weight_prior,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, double* ub, double* lb,int bound_strides):
		DecisionModel(true, false, 0., 0., n, dt, T, reward, penalty, iti, tp, cost, ub, lb, bound_strides)
	{
		/***
		 * Constructor that shares its bound arrays
		***/
		is_prior_set = false;
		this->model_var = model_var;
		this->epsilon = 1e-10;
		this->set_prior(n_prior,mu_prior,weight_prior);
		this->g2x_tolerance = 1e-12;
		#ifdef DEBUG
		std::cout<<"Created DecisionModelDiscretePrior instance at "<<this<<std::endl;
		#endif
	};
	~DecisionModelDiscretePrior();
	
	void set_prior(int n_prior,double* mu_prior, double* weight_prior);
	double get_epsilon(){return this->epsilon;};
	
	void disp();
	
	inline double x2g(const double t, const double x){
		if (t==0){
			double num = 0.;
			double den = 0.;
			const double t_over_model_var = t/model_var;
			const double x_over_model_var = x/model_var;
			for (int i=0; i<n_prior; ++i){
				const double mu_i = mu_prior[i];
				const double w_i = weight_prior[i];
				const double alpha_expmu2t = w_i*exp(-0.5*mu2_prior[i]*t_over_model_var);
				num+= alpha_expmu2t*exp(-mu_i*x_over_model_var);
				den+= alpha_expmu2t*exp(mu_i*x_over_model_var);
			}
			return 1./(1.+num/den);
		} else {
			const double exponent_factor = -0.5/model_var/t;
			double num = 0.;
			double den = 0.;
			for (int i=0; i<n_prior; ++i){
				const double mu_i = mu_prior[i];
				const double w_i = weight_prior[i];
				num+= w_i*exp(exponent_factor*pow(mu_i*t+x,2));
				den+= w_i*exp(exponent_factor*pow(mu_i*t-x,2));
			}
			return 1./(1.+num/den);
		}
	};
	
	inline double dx2g(const double t, const double x){
		if (t==0){
			const double inv_model_var = 1./model_var;
			const double t_over_model_var = t*inv_model_var;
			const double x_over_model_var = x*inv_model_var;
			double plus = 0.;
			double minus = 0.;
			double dplus = 0.;
			double dminus = 0.;
			for (int i=0; i<n_prior; ++i){
				const double mu_i = mu_prior[i];
				const double w_i = weight_prior[i];
				const double alpha_expmu2t = w_i*exp(-0.5*mu2_prior[i]*t_over_model_var);
				const double mu_over_model_var = mu_i*inv_model_var;
				plus+= alpha_expmu2t*exp(mu_i*x_over_model_var);
				minus+= alpha_expmu2t*exp(-mu_i*x_over_model_var);
				dplus+= mu_over_model_var*alpha_expmu2t*exp(mu_i*x_over_model_var);
				dminus+= mu_over_model_var*alpha_expmu2t*exp(-mu_i*x_over_model_var);
			}
			return (dminus*plus+dplus*minus)/pow(plus+minus,2);
		} else {
			const double inv_model_vart = 1./model_var/t;
			const double exponent_factor = -0.5/model_var/t;
			double num_a = 0.;
			double num_b = 0.;
			double num_c = 0.;
			double num_d = 0.;
			double den = 0.;
			for (int i=0; i<n_prior; ++i){
				const double mu_i = mu_prior[i];
				const double w_i = weight_prior[i];
				const double mut_minus_x_exponent = exponent_factor*pow(mu_i*t-x,2);
				const double mut_plus_x_exponent = exponent_factor*pow(mu_i*t+x,2);
				num_a+= w_i*exp(mut_minus_x_exponent);
				num_b+= w_i*inv_model_vart*(mu_i*t+x)*exp(mut_plus_x_exponent);
				num_c+= w_i*exp(mut_plus_x_exponent);
				num_d+= w_i*inv_model_vart*(mu_i*t-x)*exp(mut_minus_x_exponent);
				den+= w_i*(exp(mut_minus_x_exponent)+exp(mut_plus_x_exponent));
			}
			return (num_a*num_b+num_c*num_d)/den/den;
		}
	}
	
	inline double g2x(const double t, const double g){
		// Newton Raphson root finding method
		double next;
		double prev = 0.;
		for (int iter=0; iter<50; ++iter){
			double fprev = this->x2g(t,prev)-g;
			if (fprev==0.) return prev;
			double dfprev = this->dx2g(t,prev);
			if (dfprev==0.) return prev;
			next = prev - fprev/dfprev;
			if (std::abs(next-prev)<1.48e-8) break;
			prev = next;
		}
		return next;
	};
	virtual inline double g2x(const double t, const double g, const double x0){
		// Newton Raphson root finding method
		double next;
		double prev = x0;
		for (int iter=0; iter<50; ++iter){
			double fprev = this->x2g(t,prev)-g;
			if (fprev==0.) return prev;
			double dfprev = this->dx2g(t,prev);
			if (dfprev==0.) return prev;
			next = prev - fprev/dfprev;
			if (std::abs(next-prev)<1.48e-8) break;
			prev = next;
		}
		return next;
	};
	
	inline double x_transition_probability(const double t, const double xt, const double xtdt){
		const double present_exp_factor = -0.5/model_var/t;
		const double future_exp_factor = -0.5/model_var/(t+dt);
		double present_a = 0.;
		double present_b = 0.;
		double future_a = 0.;
		double future_b = 0.;
		for (int i=0; i<n_prior; ++i){
			const double mu_i = mu_prior[i];
			const double w_i = weight_prior[i];
			present_a+= w_i*exp(present_exp_factor*pow(mu_i*t-xt,2));
			present_b+= w_i*exp(present_exp_factor*pow(mu_i*t+xt,2));
			future_a+= w_i*exp(future_exp_factor*pow(mu_i*(t+dt)-xtdt,2));
			future_b+= w_i*exp(future_exp_factor*pow(mu_i*(t+dt)+xtdt,2));
		}
		return exp(future_exp_factor*pow(xt*dt-(xtdt-xt)*t,2)/t/dt)*
				(future_a+future_b)/(present_a+present_b);
	}
	
	inline double g_transition_probability(const double t, const double xt, const double xtdt){
		const double tdt = t+dt;
		const double inv_var = 1./model_var;
		const double inv_var_tdt = inv_var/(tdt);
		const double present_exp_factor = -0.5*inv_var/t;
		const double future_exp_factor = -0.5*inv_var_tdt;
		double present_a = 0.;
		double present_b = 0.;
		double future_a = 0.;
		double future_b = 0.;
		double deriv_future_a = 0.;
		double deriv_future_b = 0.;
		for (int i=0; i<n_prior; ++i){
			const double mu_i = mu_prior[i];
			const double w_i = weight_prior[i];
			present_a+= w_i*exp(present_exp_factor*pow(mu_i*t-xt,2));
			present_b+= w_i*exp(present_exp_factor*pow(mu_i*t+xt,2));
			const double an = w_i*exp(future_exp_factor*pow(mu_i*tdt-xtdt,2));
			const double bn = w_i*exp(future_exp_factor*pow(mu_i*tdt+xtdt,2));
			future_a+= an;
			future_b+= bn;
			deriv_future_a+= (mu_i*tdt-xtdt)*inv_var_tdt*an;
			deriv_future_b+= (mu_i*tdt+xtdt)*inv_var_tdt*bn;
		}
		return exp(future_exp_factor*pow(xt*dt-(xtdt-xt)*t,2)/t/dt)*
				pow(future_a+future_b,3)/(present_a+present_b)/(deriv_future_a*future_b+deriv_future_b*future_a);
	}
	
	double backpropagate_value(double rho, bool compute_bounds);
	double backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2);
};

class DecisionModelUnknownDiscreteVar : public DecisionModel {
public:
	int n_model_var;
	double* model_var;
	double* prior_var_prob;
	
	DecisionModelUnknownDiscreteVar(int n_model_var, double* model_var,
				   double* prior_var_prob, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost):
			DecisionModel(false, true, prior_mu_mean, prior_mu_var, n, dt, T, reward, penalty, iti, tp, cost){
					this->n_model_var = n_model_var;
					this->model_var = new double[n_model_var];
					this->prior_var_prob = new double[n_model_var];
					for (int i=0; i<n_model_var; i++){
						this->model_var[i] = model_var[i];
						this->prior_var_prob[i] = prior_var_prob[i];
					}
				};
	DecisionModelUnknownDiscreteVar(int n_model_var, double* model_var,
				   double* prior_var_prob, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, double* ub, double* lb,int bound_strides):
			DecisionModel(false, true, prior_mu_mean, prior_mu_var, n, dt, T, reward, penalty, iti, tp, cost, ub, lb, bound_strides){
					this->n_model_var = n_model_var;
					this->model_var = new double[n_model_var];
					this->prior_var_prob = new double[n_model_var];
					for (int i=0; i<n_model_var; i++){
						this->model_var[i] = model_var[i];
						this->prior_var_prob[i] = prior_var_prob[i];
					}
				};
	~DecisionModelUnknownDiscreteVar();
	
	void disp();
	
	inline double post_mu_var(const int i, const double t){
		return 1./(t/this->model_var[i] + 1./this->prior_mu_var);
	}
	
	inline double post_mu_mean(const int i, const double t, const double x){
		return (x/this->model_var[i]+this->prior_mu_mean/this->prior_mu_var)*this->post_mu_var(i,t);
	}
	
	inline double x2g(const double t, const double x){
		double num = 0.; double den = 0.;
		for (int i=0; i<n_model_var; ++i){
			const double st = sqrt(this->post_mu_var(i,t));
			const double pst = prior_var_prob[i]*st;
			num+=(pst*normcdf(this->post_mu_mean(i,t,x)/st,0.,1.));
			den+=pst;
		}
		return num/den;
	};
	
	inline double dx2g(const double t, const double x){
		double num = 0.; double den = 0.;
		for (int i=0; i<n_model_var; ++i){
			const double vt = this->post_mu_var(i,t);
			const double st = sqrt(vt);
			const double p = prior_var_prob[i];
			const double v = model_var[i];
			num+=(p*vt/v*exp(-0.5*pow(this->post_mu_mean(i,t,x),2)/vt));
			den+=p*st;
		}
		return 0.3989422804014327*num/den;
	};
	virtual inline double g2x(const double t, const double g){
		// Newton Raphson root finding method
		double next;
		double prev = 0.;
		for (int iter=0; iter<50; ++iter){
			double fprev = this->x2g(t,prev)-g;
			if (fprev==0.) return prev;
			double dfprev = this->dx2g(t,prev);
			if (dfprev==0.) return prev;
			next = prev - fprev/dfprev;
			if (std::abs(next-prev)<1.48e-8) break;
			prev = next;
		}
		return next;
	};
	virtual inline double g2x(const double t, const double g, const double x0){
		// Newton Raphson root finding method
		double next;
		double prev = x0;
		for (int iter=0; iter<50; ++iter){
			double fprev = this->x2g(t,prev)-g;
			if (fprev==0.) return prev;
			double dfprev = this->dx2g(t,prev);
			if (dfprev==0.) return prev;
			next = prev - fprev/dfprev;
			if (std::abs(next-prev)<1.48e-8) break;
			prev = next;
		}
		return next;
	};
	
	inline double x_transition_probability(const double t, const double xt, const double xtdt){
		double num = 0.;
		double den = 0.;
		const double dx = xtdt-xt;
		for (int i=0; i<n_model_var; ++i){
			const double inv_var = 1./((post_mu_var(i,t)*dt+model_var[i])*dt);
			const double w_i = prior_var_prob[i];
			num+= w_i*exp(-0.5*inv_var*pow(dx-post_mu_mean(i,t,xt),2));
			den+= w_i*sqrt(inv_var);
		}
		return 0.3989422804014327*num/den;
	}
	
	inline double g_transition_probability(const double t, const double xt, const double xtdt){
		double num_left = 0.;
		double num_right = 0.;
		double den_left = 0.;
		double den_right = 0.;
		const double dx = xtdt-xt;
		const double tdt = t+dt;
		for (int i=0; i<n_model_var; ++i){
			const double v_i = model_var[i];
			const double future_post_var = post_mu_var(i,tdt);
			const double inv_var = 1./((post_mu_var(i,t)*dt+v_i)*dt);
			const double w_i = prior_var_prob[i];
			num_left+= w_i*exp(-0.5*inv_var*pow(dx-post_mu_mean(i,t,xt),2));
			den_right+= w_i*sqrt(inv_var);
			num_right+= w_i*sqrt(future_post_var);
			den_left+= w_i*future_post_var/v_i*exp(-0.5*post_mu_mean(i,tdt,xtdt)/future_post_var);
		}
		return num_left*num_right/den_left/den_right;
	}
	
	double backpropagate_value(double rho, bool compute_bounds);
	double backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2);
};

#endif
