#include "DecisionPolicy.hpp"
/***
C++ implementation of the value dynamic programming algorithm and
first passage time probability density computations

Author: Luciano Paz
Year: 2016
***/

DecisionPolicyDescriptor::DecisionPolicyDescriptor(double model_var, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, bool owns_cost){
	this->_known_variance = true;
	this->n_model_var = 1;
	this->model_var = new double[1]; this->model_var[0] = model_var;
	this->prior_var_prob = (double*)0;
	
	this->prior_mu_mean = prior_mu_mean;
	this->prior_mu_var = prior_mu_var;
	
	this->n = n;
	this->dt = dt;
	this->nT = (int)(T/dt)+1;
	this->T = T;
	this->reward = reward;
	this->penalty = penalty;
	this->iti = iti;
	this->tp = tp;
	this->cost = cost;
	this->_owns_cost = owns_cost;
	
	this->_conjugate_mu_prior = true;
	this->n_prior = 1;
	this->mu_prior = (double*)0;
	this->weight_prior = (double*)0;
}

DecisionPolicyDescriptor::DecisionPolicyDescriptor(double model_var, int n_prior, double* mu_prior, double* weight_prior,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, bool owns_cost){
	this->_known_variance = true;
	this->n_model_var = 1;
	this->model_var = new double[1]; this->model_var[0] = model_var;
	this->prior_var_prob = (double*)0;
	
	this->n = n;
	this->dt = dt;
	this->nT = (int)(T/dt)+1;
	this->T = T;
	this->reward = reward;
	this->penalty = penalty;
	this->iti = iti;
	this->tp = tp;
	this->cost = cost;
	this->_owns_cost = owns_cost;
	
	this->_conjugate_mu_prior = false;
	this->n_prior = n_prior;
	this->mu_prior = new double[n_prior];
	this->weight_prior = new double[n_prior];
	this->prior_mu_mean = 0.;
	for (int i=0;i<n_prior;++i){
		this->mu_prior[i] = mu_prior[i];
		this->weight_prior[i] = weight_prior[i];
	}
}
DecisionPolicyDescriptor::DecisionPolicyDescriptor(int n_model_var, double* model_var,
				   double* prior_var_prob, double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, bool owns_cost){
	this->_known_variance = false;
	this->n_model_var = n_model_var;
	this->model_var = new double[n_model_var];
	this->prior_var_prob = new double[n_model_var];
	for (int i=0;i<n_model_var;++i){
		this->model_var[i] = model_var[i];
		this->prior_var_prob[i] = prior_var_prob[i];
	}
	
	this->prior_mu_mean = prior_mu_mean;
	this->prior_mu_var = prior_mu_var;
	
	this->n = n;
	this->dt = dt;
	this->nT = (int)(T/dt)+1;
	this->T = T;
	this->reward = reward;
	this->penalty = penalty;
	this->iti = iti;
	this->tp = tp;
	this->cost = cost;
	this->_owns_cost = owns_cost;
	
	this->_conjugate_mu_prior = true;
	this->n_prior = 1;
	this->mu_prior = (double*)0;
	this->weight_prior = (double*)0;
}

DecisionPolicyDescriptor::~DecisionPolicyDescriptor(){
	delete[] this->model_var;
	if (!(_known_variance)){
		delete[] this->prior_var_prob;
	}
	if (!(this->_conjugate_mu_prior)){
		delete[] this->mu_prior;
		delete[] this->weight_prior;
	}
	if (this->_owns_cost){
		delete[] this->cost;
	}
}

void DecisionPolicyDescriptor::disp(){
	/***
	 * Print DecisionPolicyDescriptor instance's information
	***/
	int i;
	std::cout<<"DecisionPolicyDescriptor instance = "<<this<<std::endl;
	std::cout<<"owns_cost = "<<_owns_cost<<std::endl;
	std::cout<<"known_variance = "<<_known_variance<<std::endl;
	std::cout<<"conjugate_mu_prior = "<<_conjugate_mu_prior<<std::endl;
	std::cout<<"n = "<<n<<std::endl;
	std::cout<<"dt = "<<dt<<std::endl;
	std::cout<<"T = "<<T<<std::endl;
	std::cout<<"nT = "<<nT<<std::endl;
	std::cout<<"cost = "<<cost<<std::endl;
	std::cout<<"reward = "<<reward<<std::endl;
	std::cout<<"penalty = "<<penalty<<std::endl;
	std::cout<<"iti = "<<iti<<std::endl;
	std::cout<<"tp = "<<tp<<std::endl;
	
	std::cout<<"n_model_var = "<<n_model_var<<std::endl;
	if (n_model_var==1){
		std::cout<<"model_var = "<<model_var[0]<<std::endl;
		std::cout<<"prior_var_prob = "<<1.<<std::endl;
	} else {
		std::cout<<"model_var = [";
		for (i=0;i<n_model_var-1;++i){
			std::cout<<model_var[i]<<", ";
		}
		std::cout<<model_var[i]<<"]"<<std::endl;
		std::cout<<"prior_var_prob = [";
		for (i=0;i<n_model_var-1;++i){
			std::cout<<prior_var_prob[i]<<", ";
		}
		std::cout<<prior_var_prob[i]<<"]"<<std::endl;
	}
	if (_conjugate_mu_prior){
		std::cout<<"prior_mu_mean = "<<prior_mu_mean<<std::endl;
		std::cout<<"prior_mu_var = "<<prior_mu_var<<std::endl;
		std::cout<<"n_prior = NOT SET"<<std::endl;
		std::cout<<"mu_prior = NOT SET"<<std::endl;
		std::cout<<"weight_prior = NOT SET"<<std::endl;
	} else {
		std::cout<<"prior_mu_mean = NOT SET"<<std::endl;
		std::cout<<"prior_mu_var = NOT SET"<<std::endl;
		std::cout<<"n_prior = "<<n_prior<<std::endl;
		std::cout<<"mu_prior = ["<<std::endl;
		for (i=0;i<n_prior-1;++i){
			std::cout<<mu_prior[i]<<", ";
		}
		std::cout<<mu_prior[i]<<"]"<<std::endl;
		std::cout<<"weight_prior = ["<<std::endl;
		for (i=0;i<n_prior-1;++i){
			std::cout<<weight_prior[i]<<", ";
		}
		std::cout<<weight_prior[i]<<"]"<<std::endl;
	}
}

DecisionPolicy::DecisionPolicy(bool known_variance, bool conjugate_mu_prior,
				   double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost){
	/***
	 * Constructor that creates its own bound arrays
	***/
	int i;
	
	this->owns_bounds = owns_bounds;
	this->known_variance = known_variance;
	this->conjugate_mu_prior = conjugate_mu_prior;
	this->prior_mu_mean = prior_mu_mean;
	this->prior_mu_var = prior_mu_var;
	if (n%2==0){
		this->n = n+1;
	} else {
		this->n = n;
	}
	this->dt = dt;
	this->T = T;
	this->nT = (int)(T/dt)+1;
	this->cost = cost;
	this->reward = reward;
	this->penalty = penalty;
	this->iti = iti;
	this->tp = tp;
	this->rho = 0.;
	this->dg = 1./double(n);
	this->g = new double[n];
	for (i=0;i<n;++i){
		this->g[i] = (0.5+double(i))*this->dg;
	}
	this->t = new double[nT];
	for (i=0;i<nT;++i){
		this->t[i] = double(i)*dt;
	}
	this->owns_bounds = true;
	this->ub = new double[nT];
	this->lb = new double[nT];
	this->bound_strides = 1;
}

DecisionPolicy::DecisionPolicy(bool known_variance, bool conjugate_mu_prior,
				   double prior_mu_mean, double prior_mu_var,
				   int n, double dt, double T, double reward, double penalty,
				   double iti, double tp, double* cost, double* ub, double* lb,int bound_strides){
	/***
	 * Constructor that shares its bound arrays
	***/
	int i;
	
	this->owns_bounds = owns_bounds;
	this->known_variance = known_variance;
	this->conjugate_mu_prior = conjugate_mu_prior;
	this->prior_mu_mean = prior_mu_mean;
	this->prior_mu_var = prior_mu_var;
	if (n%2==0){
		this->n = n+1;
	} else {
		this->n = n;
	}
	this->dt = dt;
	this->T = T;
	this->nT = (int)(T/dt)+1;
	this->cost = cost;
	this->reward = reward;
	this->penalty = penalty;
	this->iti = iti;
	this->tp = tp;
	this->rho = 0.;
	this->dg = 1./double(n);
	this->g = new double[n];
	for (i=0;i<n;++i){
		this->g[i] = (0.5+double(i))*this->dg;
	}
	this->t = new double[nT];
	for (i=0;i<nT;++i){
		this->t[i] = double(i)*dt;
	}
	this->owns_bounds = false;
	this->ub = ub;
	this->lb = lb;
	this->bound_strides = bound_strides;
}

DecisionPolicy::~DecisionPolicy(){
	/***
	 * Destructor
	***/
	delete[] g;
	delete[] t;
	if (owns_bounds){
		delete[] ub;
		delete[] lb;
	}
	#ifdef DEBUG
	std::cout<<"Destroyed DecisionPolicy instance"<<std::endl;
	#endif
}

void DecisionPolicy::disp(){
	/***
	 * Print DecisionPolicy instance's information
	***/
	std::cout<<"DecisionPolicy instance = "<<this<<std::endl;
	std::cout<<"known_variance = "<<known_variance<<std::endl;
	std::cout<<"conjugate_mu_prior = "<<conjugate_mu_prior<<std::endl;
	std::cout<<"prior_mu_mean = "<<prior_mu_mean<<std::endl;
	std::cout<<"prior_mu_var = "<<prior_mu_var<<std::endl;
	std::cout<<"n = "<<n<<std::endl;
	std::cout<<"dt = "<<dt<<std::endl;
	std::cout<<"T = "<<T<<std::endl;
	std::cout<<"nT = "<<nT<<std::endl;
	std::cout<<"dg = "<<dg<<std::endl;
	std::cout<<"cost = "<<cost<<std::endl;
	std::cout<<"reward = "<<reward<<std::endl;
	std::cout<<"penalty = "<<penalty<<std::endl;
	std::cout<<"iti = "<<iti<<std::endl;
	std::cout<<"tp = "<<tp<<std::endl;
	std::cout<<"rho = "<<rho<<std::endl;
	std::cout<<"owns_bounds = "<<owns_bounds<<std::endl;
	std::cout<<"ub = "<<ub<<std::endl;
	std::cout<<"lb = "<<lb<<std::endl;
	std::cout<<"bound_strides = "<<bound_strides<<std::endl;
}

DecisionPolicy* DecisionPolicy::create(DecisionPolicyDescriptor& dpc){
	if (dpc.known_variance()){
		if (dpc.conjugate_mu_prior()){
			return new DecisionPolicyConjPrior(dpc.model_var[0], dpc.prior_mu_mean, dpc.prior_mu_var,
						   dpc.n, dpc.dt, dpc.T, dpc.reward, dpc.penalty,
						   dpc.iti, dpc.tp, dpc.cost);
		} else {
			return new DecisionPolicyDiscretePrior(dpc.model_var[0], dpc.n_prior, dpc.mu_prior, dpc.weight_prior,
						   dpc.n, dpc.dt, dpc.T, dpc.reward, dpc.penalty,
						   dpc.iti, dpc.tp, dpc.cost);
		}
	} else {
		return new DecisionPolicyUnknownDiscreteVar(dpc.n_model_var,dpc.model_var,
						   dpc.prior_var_prob, dpc.prior_mu_mean, dpc.prior_mu_var,
						   dpc.n, dpc.dt, dpc.T, dpc.reward, dpc.penalty,
						   dpc.iti, dpc.tp, dpc.cost);
	}
}

DecisionPolicy* DecisionPolicy::create(DecisionPolicyDescriptor& dpc, double* ub, double* lb, int bound_strides){
	if (dpc.known_variance()){
		if (dpc.conjugate_mu_prior()){
			return new DecisionPolicyConjPrior(dpc.model_var[0], dpc.prior_mu_mean, dpc.prior_mu_var,
						   dpc.n, dpc.dt, dpc.T, dpc.reward, dpc.penalty,
						   dpc.iti, dpc.tp, dpc.cost, ub, lb, bound_strides);
		} else {
			return new DecisionPolicyDiscretePrior(dpc.model_var[0], dpc.n_prior, dpc.mu_prior, dpc.weight_prior,
						   dpc.n, dpc.dt, dpc.T, dpc.reward, dpc.penalty,
						   dpc.iti, dpc.tp, dpc.cost, ub, lb, bound_strides);
		}
	} else {
		return new DecisionPolicyUnknownDiscreteVar(dpc.n_model_var,dpc.model_var,
						   dpc.prior_var_prob, dpc.prior_mu_mean, dpc.prior_mu_var,
						   dpc.n, dpc.dt, dpc.T, dpc.reward, dpc.penalty,
						   dpc.iti, dpc.tp, dpc.cost, ub, lb, bound_strides);
	}
}

double DecisionPolicy::value_for_root_finding(double rho){
	/***
	 * Function that serves as a proxy for the value root finding that
	 * determines rho. Is the same as backpropagate_value(rho,false)
	***/
	return this->backpropagate_value(rho,false);
}

double DecisionPolicy::iterate_rho_value(double tolerance){
	// Use arbitrary default upper and lower bounds
	return this->iterate_rho_value(tolerance,-10.,10.);
}

double DecisionPolicy::iterate_rho_value(double tolerance, double lower_bound, double upper_bound){
	/***
	 * Function that implements Brent's algorithm for root finding.
	 * This function was adapted from brent.cpp written by John Burkardt,
	 * that was based on a fortran 77 implementation by Richard Brent.
	 * It searches for the value of rho that sets the value of g=0.5 at
	 * t=0 equal to 0. It finds a value for rho within a certain tolerance
	 * for the value of g=0.5 at t=0.
	***/
	double low_buffer;
	double func_at_low_bound, func_at_up_bound, func_at_low_buffer;
	double d;
	double interval;
	double m;
	double machine_eps = 2.220446049250313E-016;
	double p;
	double q;
	double r;
	double s;
	double tol;
	func_at_low_bound = this->value_for_root_finding(lower_bound);
	func_at_up_bound = this->value_for_root_finding(upper_bound);
	if (func_at_low_bound==0){
		this->rho = lower_bound;
	} else if (func_at_up_bound==0){
		this->rho = upper_bound;
	} else {
		// Adapt the bounds to get a sign changing interval
		while (SIGN(func_at_low_bound)==SIGN(func_at_up_bound)){
			if ((func_at_low_bound<func_at_up_bound and func_at_low_bound<0) ||
			    (func_at_low_bound>func_at_up_bound and func_at_low_bound>0)){
				lower_bound = upper_bound;
				upper_bound*=10;
				if (upper_bound>0){
					upper_bound*=10;
				} else if (upper_bound<0) {
					upper_bound*=-1;
				} else {
					upper_bound=1e-6;
				}
				func_at_up_bound = this->value_for_root_finding(upper_bound);
			} else if ((func_at_low_bound>func_at_up_bound and func_at_low_bound<0) ||
			    (func_at_low_bound<func_at_up_bound and func_at_low_bound>0)){
				upper_bound = lower_bound;
				if (lower_bound<0){
					lower_bound*=10;
				} else if (lower_bound>0) {
					lower_bound*=-1;
				} else {
					lower_bound=-1e-6;
				}
				func_at_low_bound = this->value_for_root_finding(lower_bound);
			}
		}
		// Brent's Algorithm for root finding
		low_buffer = lower_bound;
		func_at_low_buffer = func_at_low_bound;
		interval = upper_bound - lower_bound;
		d = interval;
		
		for ( ; ; ) {
			if (std::abs(func_at_low_buffer)<std::abs(func_at_up_bound)) {
				lower_bound = upper_bound;
				upper_bound = low_buffer;
				low_buffer = lower_bound;
				func_at_low_bound = func_at_up_bound;
				func_at_up_bound = func_at_low_buffer;
				func_at_low_buffer = func_at_low_bound;
			}
			tol = 2.0*machine_eps*std::abs(upper_bound) + tolerance;
			m = 0.5*(low_buffer-upper_bound);
			if (std::abs(m)<= tol || func_at_up_bound==0.0) {
				break;
			}
			if (std::abs(interval)<tol || std::abs(func_at_low_bound)<=std::abs(func_at_up_bound)) {
				interval = m;
				d = interval;
			} else {
				s = func_at_up_bound / func_at_low_bound;
				if (lower_bound==low_buffer){
					p = 2.0 * m * s;
					q = 1.0 - s;
				} else {
					q = func_at_low_bound / func_at_low_buffer;
					r = func_at_up_bound / func_at_low_buffer;
					p = s * ( 2.0 * m * q * ( q - r ) - ( upper_bound - lower_bound ) * ( r - 1.0 ) );
					q = ( q - 1.0 ) * ( r - 1.0 ) * ( s - 1.0 );
				}
				if ( 0.0 < p ) {
					q = - q;
				} else {
					p = - p;
				}
				
				s = interval;
				interval = d;
				
				if ( 2.0 * p < 3.0 * m * q - std::abs ( tol * q ) &&
					p < std::abs ( 0.5 * s * q ) ) {
					d = p / q;
				} else {
					interval = m;
					d = interval;
				}
			}
			lower_bound = upper_bound;
			func_at_low_bound = func_at_up_bound;

			if ( tol < std::abs ( d ) ){
				upper_bound = upper_bound + d;
			} else if ( 0.0 < m ) {
				upper_bound = upper_bound + tol;
			} else {
				upper_bound = upper_bound - tol;
			}
			
			func_at_up_bound = value_for_root_finding(upper_bound);
			if ( (0.0<func_at_up_bound && 0.0<func_at_low_buffer) || (func_at_up_bound<=0.0 && func_at_low_buffer<=0.0)) {
				low_buffer = lower_bound;
				func_at_low_buffer = func_at_low_bound;
				interval = upper_bound - lower_bound;
				d = interval;
			}
		}
		this->rho = upper_bound;
	}
	return this->rho;
}

double* DecisionPolicy::x_ubound(){
	/***
	 * Compute the x space upper bound as a function of time. This
	 * function creates a new double[] and returns it.
	***/
	#ifdef DEBUG
	std::cout<<"Entered x_ubound = "<<std::endl;
	#endif
	int i;
	double *xb = new double[nT];
	for (i=0;i<nT;++i){
		xb[i] = g2x(t[i],ub[i]);
	}
	return xb;
}
void DecisionPolicy::x_ubound(double* xb){
	/***
	 * Compute the x space upper bound as a function of time. This
	 * function places the values in the provided pointer. Beware of the
	 * size of the allocated memory as no size checks are performed
	***/
	#ifdef DEBUG
	std::cout<<"Entered x_ubound_double* with xb = "<<xb<<std::endl;
	#endif
	int i;
	for (i=0;i<nT;++i){
		xb[i] = g2x(t[i],ub[i]);
	}
}

double* DecisionPolicy::x_lbound(){
	/***
	 * Compute the x space lower bound as a function of time. This
	 * function creates a new double[] and returns it.
	***/
	#ifdef DEBUG
	std::cout<<"Entered x_lbound"<<std::endl;
	#endif
	int i;
	double *xb = new double[nT];
	for (i=0;i<nT;++i){
		xb[i] = g2x(t[i],lb[i]);
	}
	return xb;
}
void DecisionPolicy::x_lbound(double* xb){
	/***
	 * Compute the x space lower bound as a function of time. This
	 * function places the values in the provided pointer. Beware of the
	 * size of the allocated memory as no size checks are performed
	***/
	#ifdef DEBUG
	std::cout<<"Entered x_lbound_double* with xb = "<<xb<<std::endl;
	#endif
	int i;
	for (i=0;i<nT;++i){
		xb[i] = g2x(t[i],lb[i]);
	}
}

double DecisionPolicy::Psi(double mu, double model_var, double* bound, int itp, double tp, double x0, double t0){
	double normpdf = 0.3989422804014327*exp(-0.5*pow(bound[itp]-x0-mu*(tp-t0),2)/model_var/(tp-t0))/sqrt(model_var*(tp-t0));
	double bound_prime;
	if (itp<this->nT-1){
		bound_prime = 0.5*(bound[itp+1]-bound[itp-1])/this->dt;
	} else {
		bound_prime = 0.;
	}
	// double bound_prime = itp<int(sizeof(bound)/sizeof(double)-1) ? (bound[itp+1]-bound[itp])/this->dt : 0.;
	return 0.5*normpdf*(bound_prime-(bound[itp]-x0)/(tp-t0));
}

void DecisionPolicy::rt(double mu, double model_var, double* g1, double* g2, double* xub, double* xlb){
	#ifdef DEBUG
	std::cout<<"Entered rt"<<std::endl;
	#endif
	unsigned int i,j;
	unsigned int tnT = this->nT;
	double t0,tj,ti,normalization;
	bool delete_xub = false;
	bool delete_xlb = false;
	bool bounds_touched = false;
	if (xub==NULL){
		xub = this->x_ubound();
		delete_xub = true;
	}
	if (xlb==NULL){
		xlb = this->x_lbound();
		delete_xlb = true;
	}
	
	g1[0] = 0.;
	g2[0] = 0.;
	t0 = this->t[0];
	
	if (xub[1]<=xlb[1]){
		// If the bounds collapse to 0 instantly, the decision will be taken instantly and randomly
		bounds_touched = true;
		g1[1] = 0.5;
		g2[1] = 0.5;
	} else {
		g1[1] = -2.*this->Psi(mu,model_var,xub,1,this->t[1],this->prior_mu_mean,t0);
		g2[1] = 2.*this->Psi(mu,model_var,xlb,1,this->t[1],this->prior_mu_mean,t0);
	}
	// Because of numerical instabilities, we must take care that g1 and g2 are always positive
	if (g1[1]<0.) g1[1] = 0.;
	if (g2[1]<0.) g2[1] = 0.;
	// If model_var is too small, Psi will always return 0 because the exp
	// will underflow. In these cases, the sign of mu determines choice with
	// absolute certainty.
	if (g1[1]==0 && g2[1]==0){
		if (mu>0.){
			g1[1] = 1.;
		} else {
			g2[1] = 1.;
		}
	}
	normalization = g1[1]+g2[1];
	for (i=2;i<tnT;++i){
		if (bounds_touched){
			g1[i] = 0.;
			g2[i] = 0.;
		} else {
			ti = this->t[i];
			g1[i] = -this->Psi(mu,model_var,xub,i,ti,this->prior_mu_mean,t0);
			g2[i] = this->Psi(mu,model_var,xlb,i,ti,this->prior_mu_mean,t0);
			for (j=1;j<i;++j){
				tj = this->t[j];
				g1[i]+=this->dt*(g1[j]*this->Psi(mu,model_var,xub,i,ti,xub[j],tj)+
								 g2[j]*this->Psi(mu,model_var,xub,i,ti,xlb[j],tj));
				g2[i]-=this->dt*(g1[j]*this->Psi(mu,model_var,xlb,i,ti,xub[j],tj)+
								 g2[j]*this->Psi(mu,model_var,xlb,i,ti,xlb[j],tj));
			}
			g1[i]*=2.;
			g2[i]*=2.;
			// Because of numerical instabilities, we must take care that g1 and g2 are always positive
			if (g1[i]<0.) g1[i] = 0.;
			if (g2[i]<0.) g2[i] = 0.;
			normalization+= g1[i]+g2[i];
		}
		if (xub[i]<=xlb[i]){
			bounds_touched = true;
		}
	}
	normalization*=this->dt;
	for (i=0;i<tnT;++i){
		g1[i]/=normalization;
		g2[i]/=normalization;
	}
	
	if (delete_xub) delete[] xub;
	if (delete_xlb) delete[] xlb;
}

void DecisionPolicy::fpt_conf_matrix(double* first_passage_time, int* first_passage_time_strides, int n_alternatives, int confidence_partition, double* confidence_response, int* confidence_response_strides, double* out){
	int decision_ind, t_ind, i, out_base_ind;
	double prior_c_ind, c_ind, next_c_ind;
	double ftp;
	bool instant_rise, instant_fall;
	double rise_start, rise_end, fall_end, rise_slope, fall_slope, h;
	const double ind_scaling = (double)(confidence_partition-1);
	const int fpt_dec_stride = first_passage_time_strides[0];
	const int fpt_t_stride = first_passage_time_strides[1];
	const int conf_dec_stride = confidence_response_strides[0];
	const int conf_t_stride = confidence_response_strides[1];
	
	for (decision_ind=0; decision_ind<n_alternatives; ++decision_ind){
		prior_c_ind = c_ind = ind_scaling*confidence_response[decision_ind*conf_dec_stride];
		next_c_ind = ind_scaling*confidence_response[decision_ind*conf_dec_stride+conf_t_stride];
		ftp = first_passage_time[decision_ind*fpt_dec_stride];
		instant_rise = std::abs(next_c_ind-c_ind)<=1.;
		
		out[decision_ind*nT*confidence_partition+int(c_ind)] = ftp*(1.+int(c_ind)-c_ind);
		out[decision_ind*nT*confidence_partition+int(ceil(c_ind))]+= ftp*(c_ind-int(c_ind));
		c_ind = next_c_ind;
		for (t_ind=1; t_ind<nT-1; ++t_ind){
			out_base_ind = decision_ind*nT*confidence_partition + t_ind*confidence_partition;
			next_c_ind = ind_scaling*confidence_response[(t_ind+1)*conf_t_stride+decision_ind*conf_dec_stride];
			ftp = first_passage_time[t_ind*fpt_t_stride+decision_ind*fpt_dec_stride];
			
			instant_fall = std::abs(next_c_ind-c_ind)<=1.;
			if (instant_rise && instant_fall){
				// Instant rise and fall
				int floor_c_ind = (int) c_ind;
				int ceil_c_ind = (int) ceil(c_ind);
				out[out_base_ind+floor_c_ind] = ftp*(1.+floor_c_ind-c_ind);
				out[out_base_ind+ceil_c_ind]+= ftp*(c_ind-floor_c_ind);
			} else if (instant_rise){
				// Instant rise and linear fall
				rise_end = (int)round(c_ind);
				if (c_ind<next_c_ind){
					fall_end = (int)ceil(next_c_ind);
				} else {
					fall_end = (int)next_c_ind;
				}
				h = 2.*ftp/std::abs(double(fall_end-rise_end-1));
				fall_slope = -h/(fall_end-rise_end);
				if (fall_end>rise_end){
					for (i=rise_end; i<fall_end; ++i){
						out[out_base_ind+i] = fall_slope*(i-fall_end);
					}
				} else {
					for (i=rise_end; i>fall_end; --i){
						out[out_base_ind+i] = fall_slope*(i-fall_end);
					}
				}
			} else if (instant_fall){
				// Instant fall and linear rise
				rise_end = (int)round(c_ind);
				if (prior_c_ind<c_ind){
					rise_start = (int)prior_c_ind;
				} else {
					rise_start = (int)ceil(prior_c_ind);
				}
				h = 2.*ftp/std::abs(double(rise_end-rise_start-1));
				rise_slope = h/(rise_end-rise_start);
				if (rise_end>rise_start){
					for (i=rise_start+1; i<=rise_end; ++i){
						out[out_base_ind+i] = rise_slope*(i-rise_start);
					}
				} else {
					for (i=rise_start-1; i>=rise_end; --i){
						out[out_base_ind+i] = rise_slope*(i-rise_start);
					}
				}
			} else {
				// Linear rise and fall
				rise_end = (int)round(c_ind);
				if (prior_c_ind<c_ind){
					rise_start = (int)prior_c_ind;
				} else {
					rise_start = (int)ceil(prior_c_ind);
				}
				if (c_ind<next_c_ind){
					fall_end = (int)ceil(next_c_ind);
				} else {
					fall_end = (int)next_c_ind;
				}
				h = 2.*ftp/(std::abs(double(rise_end-rise_start))+std::abs(double(fall_end-rise_end)));
				rise_slope = h/(rise_end-rise_start);
				fall_slope = -h/(fall_end-rise_end);
				if (rise_end>rise_start){
					for (i=rise_start+1; i<=rise_end; ++i){
						out[out_base_ind+i] = rise_slope*(i-rise_start);
					}
				} else {
					for (i=rise_start-1; i>=rise_end; --i){
						out[out_base_ind+i] = rise_slope*(i-rise_start);
					}
				}
				if (fall_end>rise_end){
					for (i=rise_end+1; i<fall_end; ++i){
						out[out_base_ind+i] = fall_slope*(i-fall_end);
					}
				} else {
					for (i=rise_end-1; i>fall_end; --i){
						out[out_base_ind+i] = fall_slope*(i-fall_end);
					}
				}
			}
			prior_c_ind = c_ind;
			c_ind = next_c_ind;
			instant_rise = instant_fall;
		}
		ftp = first_passage_time[t_ind*fpt_t_stride+decision_ind*fpt_dec_stride];
		out_base_ind = decision_ind*nT*confidence_partition + t_ind*confidence_partition;
		if (instant_rise){
			// Instant rise and fall
			int floor_c_ind = (int) c_ind;
			int ceil_c_ind = (int) ceil(c_ind);
			out[out_base_ind+floor_c_ind] = ftp*(1.+floor_c_ind-c_ind);
			out[out_base_ind+ceil_c_ind]+= ftp*(c_ind-floor_c_ind);
		} else {
			// Instant fall and linear rise
			rise_end = (int)round(c_ind);
			if (prior_c_ind<c_ind){
				rise_start = (int)prior_c_ind;
			} else {
				rise_start = (int)ceil(prior_c_ind);
			}
			h = 2.*ftp/std::abs(double(rise_end-rise_start-1));
			rise_slope = h/(rise_end-rise_start);
			if (rise_end>rise_start){
				for (i=rise_start+1; i<=rise_end; ++i){
					out[out_base_ind+i] = rise_slope*(i-rise_start);
				}
			} else {
				for (i=rise_start-1; i>=rise_end; --i){
					out[out_base_ind+i] = rise_slope*(i-rise_start);
				}
			}
		}
	}
}

/***
 * DecisionPolicyConjPrior is a class that implements the dynamic programing method
 * that computes the value of a given belief state as a function of time.
 * 
 * This class implements the method given in Drugowitsch et al 2012 but
 * is limited to constant cost values.
***/

DecisionPolicyConjPrior::~DecisionPolicyConjPrior(){
	/***
	 * Destructor
	***/
	#ifdef DEBUG
	std::cout<<"Destroyed DecisionPolicyConjPrior instance"<<std::endl;
	#endif
}

void DecisionPolicyConjPrior::disp(){
	/***
	 * Print DecisionPolicyConjPrior instance's information
	***/
	std::cout<<"DecisionPolicyConjPrior instance = "<<this<<std::endl;
	std::cout<<"model_var = "<<model_var<<std::endl;
	std::cout<<"prior_mu_mean = "<<prior_mu_mean<<std::endl;
	std::cout<<"prior_mu_var = "<<prior_mu_var<<std::endl;
	std::cout<<"dt = "<<dt<<std::endl;
	std::cout<<"dg = "<<dg<<std::endl;
	std::cout<<"T = "<<T<<std::endl;
	std::cout<<"cost = "<<cost<<std::endl;
	std::cout<<"reward = "<<reward<<std::endl;
	std::cout<<"iti = "<<iti<<std::endl;
	std::cout<<"tp = "<<tp<<std::endl;
	std::cout<<"rho = "<<rho<<std::endl;
	std::cout<<"n = "<<n<<std::endl;
	std::cout<<"nT = "<<nT<<std::endl;
	std::cout<<"t = "<<t<<std::endl;
	
	std::cout<<"owns_bounds = "<<owns_bounds<<std::endl;
	std::cout<<"bound_strides = "<<bound_strides<<std::endl;
	std::cout<<"ub = "<<ub<<std::endl;
	std::cout<<"lb = "<<lb<<std::endl;
}

double DecisionPolicyConjPrior::backpropagate_value(double rho, bool compute_bounds){
	/***
	 * Main function:
	 * backpropagate_value(double rho, bool compute_bounds)
	 * 
	 * This function applies dynamic programing to determine the value
	 * of holding belief g at time t. It should be used under two different
	 * circumstances.
	 * 1) Iterate rho value until the value of g=0.5 at t=0 is 0
	 * 2) To compute the decision bounds in g space, once rho has been computed
	 * 
	 * This means that the value of belief g at time t is not stored
	 * during the execution. This is done to improve memory usage and
	 * execution time.
	 * 
	 * This function returns the value of g=0.5 at t=0.
	 * If compute_bounds=true it also sets the values of the bound arrays
	 * ub and lb.
	***/
	#ifdef DEBUG
	std::cout<<"Entered backpropagate_value with rho = "<<rho<<std::endl;
	#endif
	bool setted_ub = false;
	int previous_value_zone;
	int current_value_zone;
	int i, j, k, bound_ind, curr_invg, fut_invg;
	double post_var_t1, post_var_t, norm_p, maxp;
	double value[n], v1[n], v2[n], v_explore[n], p[n];
	double invg[2][n];
	
	this->rho = rho;
	curr_invg = 0;
	fut_invg = 1;
	#ifdef DEBUG
	FILE *details_file = fopen("details.txt","w");
	FILE *prob_file = fopen("prob.txt","w");
	FILE *value_file = fopen("value.txt","w");
	FILE *v_explore_file = fopen("v_explore.txt","w");
	#endif
	// Compute the value at the time limit T, where the subject must decide
	for (i=0;i<n;++i){
		// Value of deciding option 1
		v1[i] = reward*g[i]-penalty*(1.-g[i]) - (iti+(1.-g[i])*tp)*rho;
		// Value of deciding option 2
		v2[i] = reward*(1.-g[i])-penalty*g[i] - (iti+g[i]*tp)*rho;
		// Value of the belief g[i]
		value[i] = v1[i]>=v2[i] ? v1[i] : v2[i];
		// We compute invg that is the x(t) that corresponds to having g[i] at time T
		// and store it to save computations
		invg[fut_invg][i] = g2x(t[nT-1],g[i]);
		#ifdef DEBUG
		if (i<n-1){
			fprintf(value_file,"%f\t",value[i]);
		} else {
			fprintf(value_file,"%f\n",value[i]);
		}
		#endif
	}
	if (compute_bounds){
		this->lb[bound_strides*(nT-1)] = 0.5;
		this->ub[bound_strides*(nT-1)] = 0.5;
	}
	
	post_var_t1 = post_mu_var(this->t[nT-1]);
	// Dynamic programing loop that goes backward in time from T->0
	// Speed increase by precalculating values
	const double prior_div = prior_mu_mean/prior_mu_var;
	const double inv_model_var = 1./model_var;
	for (i=nT-2;i>=0;i--){
		#ifdef INFO
		if (i%100==0) std::cout<<i<<std::endl;
		#endif
		setted_ub = false;
		bound_ind = bound_strides*i;
		ub[bound_ind] = g[n-1];
		lb[bound_ind] = g[0];
		
		//Speed increase by reducing array access
		const double t_i = t[i];
		const double cost_rho_dt = (cost[i]+rho)*dt;
		post_var_t = post_mu_var(t_i);
		for (j=0;j<n;++j){
			v_explore[j] = 0.;
			invg[curr_invg][j] = g2x(t_i,g[j]);
			norm_p = 0.;
			maxp = -INFINITY;
			// Speed increase by reducing array access and precalculating values
			const double mu_n_dt = post_mu_mean(t_i,invg[curr_invg][j])*dt;
			const double mean_1 = invg[curr_invg][j]+mu_n_dt;
			const double var_1 = 1./((post_var_t*dt+model_var)*dt);
			const double* future_x = invg[fut_invg];
			
			// Compute P(g(t+dt)|g(t)) in two steps. First compute the exponent
			for (k=0;k<n;++k){
				p[k] = -0.5*pow(future_x[k]-mean_1,2)*var_1+
						0.5*pow(future_x[k]*inv_model_var+prior_div,2)*post_var_t1;
				maxp = p[k]>maxp ? p[k] : maxp;
				#ifdef DEBUG
				fprintf(details_file,"%f\t%f\t%f\t%f\t%f\n",invg[fut_invg][k],invg[curr_invg][j],mu_n_dt,post_var_t,post_var_t1);
				#endif
			}
			// Then exponentiate and compute the value of exploring
			for (k=0;k<n;++k){
				p[k] = exp(p[k]-maxp);
				norm_p+=p[k];
				v_explore[j]+= p[k]*value[k];
			}
			// Divide the value of exploring by the normalization factor and discount the cost and rho
			v_explore[j] = v_explore[j]/norm_p - cost_rho_dt;
			
			#ifdef DEBUG
			for (k=0;k<n-1;++k){
				fprintf(prob_file,"%f\t",p[k]/norm_p);
			}
			fprintf(prob_file,"%f\n",p[k]/norm_p);
			if (j<n-1){
				fprintf(v_explore_file,"%f\t",v_explore[j]);
			} else {
				fprintf(v_explore_file,"%f\n",v_explore[j]);
			}
			#endif
		}
		// Update temporal values
		post_var_t1 = post_var_t;
		curr_invg = (curr_invg+1)%2;
		fut_invg = (fut_invg+1)%2;
		// Value computation
		previous_value_zone = -1;
		current_value_zone = -1;
		for (j=0;j<n;++j){
			if (v1[j]>=v2[j] && v1[j]>=v_explore[j]){
				value[j] = v1[j];
				current_value_zone = 1;
			} else if (v2[j]>v1[j] && v2[j]>=v_explore[j]){
				value[j] = v2[j];
				current_value_zone = 2;
			} else if (v_explore[j]>v1[j] && v_explore[j]>v2[j]){
				value[j] = v_explore[j];
				current_value_zone = 0;
			}
			#ifdef DEBUG
			if (j<n-1){
				fprintf(value_file,"%f\t",value[j]);
			} else {
				fprintf(value_file,"%f\n",value[j]);
			}
			#endif
			// Bound computation
			if (compute_bounds){
				if (j>0 && j<n){
					if (std::abs(v1[j]-v_explore[j])<1e-8){
						if (!setted_ub){
							ub[bound_ind] = g[j];
							setted_ub = true;
						}
					} else if (std::abs(v2[j]-v_explore[j])<1e-8){
						lb[bound_ind] = g[j];
					} else if (current_value_zone!=previous_value_zone){
						if (current_value_zone==1 && previous_value_zone==0 && !setted_ub){
							ub[bound_ind] = (g[j-1]*(v1[j]-v_explore[j]) - g[j]*(v1[j-1]-v_explore[j-1])) / (v_explore[j-1]-v_explore[j]+v1[j]-v1[j-1]);
						} else if (current_value_zone==1 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v1[j]-v2[j]) - g[j]*(v1[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v1[j]-v1[j-1]);
							if (!setted_ub){
								ub[bound_ind] = lb[bound_ind];
							}
						} else if (current_value_zone==0 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v_explore[j]-v2[j]) - g[j]*(v_explore[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v_explore[j]-v_explore[j-1]);
						}
					}
				}
			}
			previous_value_zone = current_value_zone;
		}
	}
	
	#ifdef DEBUG
	fclose(prob_file);
	fclose(value_file);
	fclose(v_explore_file);
	std::cout<<"Exited backpropagate_value "<<std::endl;
	#endif
	return value[int(0.5*n)];
}

double DecisionPolicyConjPrior::backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2){
	/***
	 * Main function:
	 * backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2)
	 * 
	 * This function applies dynamic programing to determine the value
	 * of holding belief g at time t. It should be used under two different
	 * circumstances.
	 * 1) Iterate rho value until the value of g=0.5 at t=0 is 0
	 * 2) To compute the decision bounds in g space, once rho has been computed
	 * 
	 * This means that the value of belief g at time t is not stored
	 * during the execution. This is done to improve memory usage and
	 * execution time.
	 * 
	 * This function returns the value of g=0.5 at t=0.
	 * If compute_bounds=true it also sets the values of the bound arrays
	 * ub and lb.
	***/
	#ifdef DEBUG
	std::cout<<"Entered backpropagate_value with rho = "<<rho<<std::endl;
	#endif
	bool setted_ub = false;
	int previous_value_zone;
	int current_value_zone;
	int i, j, k, bound_ind, curr_invg, fut_invg;
	double post_var_t1, post_var_t, norm_p, maxp;
	double p[n];
	double invg[2][n];
	
	this->rho = rho;
	curr_invg = 0;
	fut_invg = 1;
	#ifdef DEBUG
	FILE *details_file = fopen("details.txt","w");
	FILE *prob_file = fopen("prob.txt","w");
	FILE *value_file = fopen("value.txt","w");
	FILE *v_explore_file = fopen("v_explore.txt","w");
	#endif
	// Compute the value at the time limit T, where the subject must decide
	for (i=0;i<n;++i){
		// Value of deciding option 1
		v1[i] = reward*g[i]-penalty*(1.-g[i]) - (iti+(1.-g[i])*tp)*rho;
		// Value of deciding option 2
		v2[i] = reward*(1.-g[i])-penalty*g[i] - (iti+g[i]*tp)*rho;
		// Value of the belief g[i]
		value[i+(nT-1)*n] = v1[i]>=v2[i] ? v1[i] : v2[i];
		// We compute invg that is the x(t) that corresponds to having g[i] at time T
		// and store it to save computations
		invg[fut_invg][i] = g2x(t[nT-1],g[i]);
		#ifdef DEBUG
		if (i<n-1){
			fprintf(value_file,"%f\t",value[i+(nT-1)*n]);
		} else {
			fprintf(value_file,"%f\n",value[i+(nT-1)*n]);
		}
		#endif
	}
	if (compute_bounds){
		this->lb[bound_strides*(nT-1)] = 0.5;
		this->ub[bound_strides*(nT-1)] = 0.5;
	}
	
	post_var_t1 = post_mu_var(this->t[nT-1]);
	// Dynamic programing loop that goes backward in time from T->0
	// Speed increase by precalculating values
	const double prior_div = prior_mu_mean/prior_mu_var;
	const double inv_model_var = 1./model_var;
	for (i=nT-2;i>=0;i--){
		#ifdef INFO
		if (i%100==0) std::cout<<i<<std::endl;
		#endif
		setted_ub = false;
		bound_ind = bound_strides*i;
		ub[bound_ind] = g[n-1];
		lb[bound_ind] = g[0];
		
		//Speed increase by reducing array access
		const double t_i = t[i];
		const double cost_rho_dt = (cost[i]+rho)*dt;
		post_var_t = post_mu_var(t_i);
		for (j=0;j<n;++j){
			v_explore[j+i*n] = 0.;
			invg[curr_invg][j] = g2x(t_i,g[j]);
			norm_p = 0.;
			maxp = -INFINITY;
			// Speed increase by reducing array access and precalculating values
			const double mu_n_dt = post_mu_mean(t_i,invg[curr_invg][j])*dt;
			const double mean_1 = invg[curr_invg][j]+mu_n_dt;
			const double var_1 = 1./((post_var_t*dt+model_var)*dt);
			const double* future_x = invg[fut_invg];
			
			// Compute P(g(t+dt)|g(t)) in two steps. First compute the exponent
			for (k=0;k<n;++k){
				p[k] = -0.5*pow(future_x[k]-mean_1,2)*var_1+
						0.5*pow(future_x[k]*inv_model_var+prior_div,2)*post_var_t1;
				maxp = p[k]>maxp ? p[k] : maxp;
				#ifdef DEBUG
				fprintf(details_file,"%f\t%f\t%f\t%f\t%f\n",invg[fut_invg][k],invg[curr_invg][j],mu_n_dt,post_var_t,post_var_t1);
				#endif
			}
			// Then exponentiate and compute the value of exploring
			for (k=0;k<n;++k){
				p[k] = exp(p[k]-maxp);
				norm_p+=p[k];
				v_explore[j+i*n]+= p[k]*value[k+(i+1)*n];
			}
			// Divide the value of exploring by the normalization factor and discount the cost and rho
			v_explore[j+i*n] = v_explore[j+i*n]/norm_p - cost_rho_dt;
			
			#ifdef DEBUG
			for (k=0;k<n-1;++k){
				fprintf(prob_file,"%f\t",p[k]/norm_p);
			}
			fprintf(prob_file,"%f\n",p[k]/norm_p);
			if (j<n-1){
				fprintf(v_explore_file,"%f\t",v_explore[j+i*n]);
			} else {
				fprintf(v_explore_file,"%f\n",v_explore[j+i*n]);
			}
			#endif
		}
		// Update temporal values
		post_var_t1 = post_var_t;
		curr_invg = (curr_invg+1)%2;
		fut_invg = (fut_invg+1)%2;
		// Value computation
		previous_value_zone = -1;
		current_value_zone = -1;
		for (j=0;j<n;++j){
			if (v1[j]>=v2[j] && v1[j]>=v_explore[j+i*n]){
				value[j+i*n] = v1[j];
				current_value_zone = 1;
			} else if (v2[j]>v1[j] && v2[j]>=v_explore[j+i*n]){
				value[j+i*n] = v2[j];
				current_value_zone = 2;
			} else if (v_explore[j+i*n]>v1[j] && v_explore[j+i*n]>v2[j]){
				value[j+i*n] = v_explore[j+i*n];
				current_value_zone = 0;
			}
			#ifdef DEBUG
			if (j<n-1){
				fprintf(value_file,"%f\t",value[j+i*n]);
			} else {
				fprintf(value_file,"%f\n",value[j+i*n]);
			}
			#endif
			// Bound computation
			if (compute_bounds){
				if (j>0 && j<n){
					if (std::abs(v1[j]-v_explore[j+i*n])<1e-8){
						if (!setted_ub){
							ub[bound_ind] = g[j];
							setted_ub = true;
						}
					} else if (std::abs(v2[j]-v_explore[j+i*n])<1e-8){
						lb[bound_ind] = g[j];
					} else if (current_value_zone!=previous_value_zone){
						if (current_value_zone==1 && previous_value_zone==0 && !setted_ub){
							ub[bound_ind] = (g[j-1]*(v1[j]-v_explore[j+i*n]) - g[j]*(v1[j-1]-v_explore[j-1+i*n])) / (v_explore[j-1+i*n]-v_explore[j+i*n]+v1[j]-v1[j-1]);
						} else if (current_value_zone==1 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v1[j]-v2[j]) - g[j]*(v1[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v1[j]-v1[j-1]);
							if (!setted_ub){
								ub[bound_ind] = lb[bound_ind];
							}
						} else if (current_value_zone==0 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v_explore[j+i*n]-v2[j]) - g[j]*(v_explore[j-1+i*n]-v2[j-1])) / (v2[j-1]-v2[j]+v_explore[j+i*n]-v_explore[j-1+i*n]);
						}
					}
				}
			}
			previous_value_zone = current_value_zone;
		}
	}
	
	#ifdef DEBUG
	fclose(prob_file);
	fclose(value_file);
	fclose(v_explore_file);
	std::cout<<"Exited backpropagate_value "<<std::endl;
	#endif
	return value[int(0.5*n)];
}


DecisionPolicyDiscretePrior::~DecisionPolicyDiscretePrior(){
	/***
	 * Destructor
	***/
	if (is_prior_set){
		delete[] mu_prior;
		delete[] mu2_prior;
		delete[] weight_prior;
	}
	#ifdef DEBUG
	std::cout<<"Destroyed DecisionPolicyDiscretePrior instance"<<std::endl;
	#endif
}

void DecisionPolicyDiscretePrior::set_prior(int n_prior,double* mu_prior, double* weight_prior){
	int i;
	double normalization = 0.;
	this->n_prior = n_prior;
	if (this->is_prior_set){
		delete[] this->mu_prior;
		delete[] this->mu2_prior;
		delete[] this->weight_prior;
	}
	this->mu_prior = new double[n_prior];
	this->mu2_prior = new double[n_prior];
	this->weight_prior = new double[n_prior];
	this->prior_mu_mean = 0.;
	
	for (i=0;i<n_prior;++i){
		if (mu_prior[i]==0.){
			this->mu_prior[i] = this->epsilon;
		} else {
			this->mu_prior[i] = mu_prior[i];
		}
		this->mu2_prior[i] = this->mu_prior[i]*this->mu_prior[i];
		this->weight_prior[i] = weight_prior[i];
		normalization+= this->weight_prior[i];
	}
	this->prior_mu_var = 0.;
	for (i=0;i<n_prior;++i){
		this->weight_prior[i]*=0.5/normalization;
		this->prior_mu_var+= 2*this->weight_prior[i]*this->mu2_prior[i];
	}
	this->is_prior_set = true;
}

void DecisionPolicyDiscretePrior::disp(){
	/***
	 * Print DecisionPolicyDiscretePrior instance's information
	***/
	std::cout<<"DecisionPolicyDiscretePrior instance = "<<this<<std::endl;
	std::cout<<"model_var = "<<model_var<<std::endl;
	std::cout<<"mu_prior = "<<mu_prior<<std::endl;
	std::cout<<"mu2_prior = "<<mu2_prior<<std::endl;
	std::cout<<"weight_prior = "<<weight_prior<<std::endl;
	std::cout<<"dt = "<<dt<<std::endl;
	std::cout<<"dg = "<<dg<<std::endl;
	std::cout<<"T = "<<T<<std::endl;
	std::cout<<"cost = "<<cost<<std::endl;
	std::cout<<"reward = "<<reward<<std::endl;
	std::cout<<"iti = "<<iti<<std::endl;
	std::cout<<"tp = "<<tp<<std::endl;
	std::cout<<"rho = "<<rho<<std::endl;
	std::cout<<"n = "<<n<<std::endl;
	std::cout<<"nT = "<<nT<<std::endl;
	std::cout<<"t = "<<t<<std::endl;
	
	std::cout<<"owns_bounds = "<<owns_bounds<<std::endl;
	std::cout<<"bound_strides = "<<bound_strides<<std::endl;
	std::cout<<"ub = "<<ub<<std::endl;
	std::cout<<"lb = "<<lb<<std::endl;
}

double DecisionPolicyDiscretePrior::backpropagate_value(double rho, bool compute_bounds){
	/***
	 * Main function:
	 * backpropagate_value(double rho, bool compute_bounds)
	 * 
	 * This function applies dynamic programing to determine the value
	 * of holding belief g at time t. It should be used under two different
	 * circumstances.
	 * 1) Iterate rho value until the value of g=0.5 at t=0 is 0
	 * 2) To compute the decision bounds in g space, once rho has been computed
	 * 
	 * This means that the value of belief g at time t is not stored
	 * during the execution. This is done to improve memory usage and
	 * execution time.
	 * 
	 * This function returns the value of g=0.5 at t=0.
	 * If compute_bounds=true it also sets the values of the bound arrays
	 * ub and lb.
	***/
	#ifdef DEBUG
	std::cout<<"Entered backpropagate_value with rho = "<<rho<<std::endl;
	#endif
	bool setted_ub = false;
	int previous_value_zone;
	int current_value_zone;
	int i, j, k, bound_ind, curr_invg, fut_invg, ind_prior;
	double norm_p;
	double value[n], v1[n], v2[n], v_explore[n], p[n];
	double invg[2][n];
	
	this->rho = rho;
	curr_invg = 0;
	fut_invg = 1;
	#ifdef DEBUG
	FILE *details_file = fopen("details.txt","w");
	FILE *prob_file = fopen("prob.txt","w");
	FILE *value_file = fopen("value.txt","w");
	FILE *v_explore_file = fopen("v_explore.txt","w");
	#endif
	// Compute the value at the time limit T, where the subject must decide
	for (i=0;i<n;++i){
		// Value of deciding option 1
		v1[i] = reward*g[i]-penalty*(1.-g[i]) - (iti+(1.-g[i])*tp)*rho;
		// Value of deciding option 2
		v2[i] = reward*(1.-g[i])-penalty*g[i] - (iti+g[i]*tp)*rho;
		// Value of the belief g[i]
		value[i] = v1[i]>=v2[i] ? v1[i] : v2[i];
		// We compute invg that is the x(t) that corresponds to having g[i] at time T
		// and store it to save computations
		if (i==0){
			invg[fut_invg][i] = g2x(t[nT-1],g[i]);
		} else {
			invg[fut_invg][i] = g2x(t[nT-1],g[i],invg[fut_invg][i-1]);
		}
		#ifdef DEBUG
		if (i<n-1){
			fprintf(value_file,"%f\t",value[i]);
		} else {
			fprintf(value_file,"%f\n",value[i]);
		}
		#endif
	}
	if (compute_bounds){
		this->lb[bound_strides*(nT-1)] = 0.5;
		this->ub[bound_strides*(nT-1)] = 0.5;
	}
	
	// Dynamic programing loop that goes backward in time from T->0
	// Speed increase by precalculating values
	for (i=nT-2;i>=0;i--){
		#ifdef INFO
		if (i%100==0) std::cout<<i<<std::endl;
		#endif
		setted_ub = false;
		bound_ind = bound_strides*i;
		ub[bound_ind] = g[n-1];
		lb[bound_ind] = g[0];
		
		//Speed increase by reducing array access
		const double t_i = t[i];
		const double t_i1 = t[i+1];
		const double cost_rho_dt = (cost[i]+rho)*dt;
		const double inv_model_var_ti = 1./(model_var*t_i);
		const double inv_model_var_ti1 = 1./(model_var*t_i1);
		const double present_exp_factor = -0.5*inv_model_var_ti;
		const double future_exp_factor = -0.5*inv_model_var_ti1;
		for (j=0;j<n;++j){
			v_explore[j] = 0.;
			invg[curr_invg][j] = g2x(t_i,g[j],invg[fut_invg][j]);
			if (isnan(invg[curr_invg][j])){
				invg[curr_invg][j] = g2x(t_i,g[j],0.);
			}
			norm_p = 0.;
			double maxp = -INFINITY;
			// Speed increase by reducing array access and precalculating values
			const double present_x = invg[curr_invg][j];
			// Compute P(g(t+dt)|g(t))
			for (k=0;k<n;++k){
				const double future_x = invg[fut_invg][k];
				double present_a = 0.;
				double present_b = 0.;
				double future_a = 0.;
				double future_b = 0.;
				double deriv_future_a = 0.;
				double deriv_future_b = 0.;
				for (ind_prior=0; ind_prior<n_prior; ++ind_prior){
					const double mu_i = mu_prior[ind_prior];
					const double w_i = weight_prior[ind_prior];
					present_a+= w_i*exp(present_exp_factor*pow(mu_i*t_i-present_x,2));
					present_b+= w_i*exp(present_exp_factor*pow(mu_i*t_i+present_x,2));
					const double an = w_i*exp(future_exp_factor*pow(mu_i*t_i1-future_x,2));
					const double bn = w_i*exp(future_exp_factor*pow(mu_i*t_i1+future_x,2));
					future_a+= an;
					future_b+= bn;
					deriv_future_a+= (mu_i*t_i1-future_x)*inv_model_var_ti1*an;
					deriv_future_b+= (mu_i*t_i1+future_x)*inv_model_var_ti1*bn;
				}
				//~ p[k] = exp(future_exp_factor*pow(present_x*dt-(future_x-present_x)*t_i,2)/t_i/dt)*
						//~ pow(future_a+future_b,3)/(present_a+present_b)/(deriv_future_a*future_b+deriv_future_b*future_a);
				p[k] = future_exp_factor*pow(present_x*dt-(future_x-present_x)*t_i,2)/t_i/dt +
						3*log(future_a+future_b)-log(present_a+present_b)-log(deriv_future_a*future_b+deriv_future_b*future_a);
				maxp = p[k]>maxp ? p[k] : maxp;
				#ifdef DEBUG
				fprintf(details_file,"%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",k,t_i,p[k],future_x,present_x,future_a,future_b,deriv_future_a,deriv_future_b,present_a,present_b,future_exp_factor*pow(present_x*dt-(future_x-present_x)*t_i,2)/t_i/dt,3*log(future_a+future_b),log(present_a+present_b),log(deriv_future_a*future_b+deriv_future_b*future_a));
				#endif
			}
			// Then normalize and compute the value of exploring
			if (isinf(maxp)){
				for (k=0;k<n;++k){
					p[k] = (isinf(p[k]) && p[k]>0) ? 1. : 0;
					norm_p+= p[k];
					v_explore[j]+= p[k]*value[k];
				}
			} else {
				for (k=0;k<n;++k){
					p[k] = exp(p[k]-maxp);
					norm_p+= p[k];
					v_explore[j]+= p[k]*value[k];
				}
			}
			// Divide the value of exploring by the normalization factor and discount the cost and rho
			v_explore[j] = v_explore[j]/norm_p - cost_rho_dt;
			#ifdef DEBUG
			for (k=0;k<n-1;++k){
				fprintf(prob_file,"%f\t",p[k]/norm_p);
			}
			fprintf(prob_file,"%f\n",p[k]/norm_p);
			if (j<n-1){
				fprintf(v_explore_file,"%f\t",v_explore[j]);
			} else {
				fprintf(v_explore_file,"%f\n",v_explore[j]);
			}
			#endif
		}
		// Update temporal values
		curr_invg = (curr_invg+1)%2;
		fut_invg = (fut_invg+1)%2;
		// Value computation
		previous_value_zone = -1;
		current_value_zone = -1;
		for (j=0;j<n;++j){
			if (v1[j]>=v2[j] && v1[j]>=v_explore[j]){
				value[j] = v1[j];
				current_value_zone = 1;
			} else if (v2[j]>v1[j] && v2[j]>=v_explore[j]){
				value[j] = v2[j];
				current_value_zone = 2;
			} else if (v_explore[j]>v1[j] && v_explore[j]>v2[j]){
				value[j] = v_explore[j];
				current_value_zone = 0;
			}
			#ifdef DEBUG
			if (j<n-1){
				fprintf(value_file,"%f\t",value[j]);
			} else {
				fprintf(value_file,"%f\n",value[j]);
			}
			#endif
			// Bound computation
			if (compute_bounds){
				if (j>0 && j<n){
					if (std::abs(v1[j]-v_explore[j])<1e-8){
						if (!setted_ub){
							ub[bound_ind] = g[j];
							setted_ub = true;
						}
					} else if (std::abs(v2[j]-v_explore[j])<1e-8){
						lb[bound_ind] = g[j];
					} else if (current_value_zone!=previous_value_zone){
						if (current_value_zone==1 && previous_value_zone==0 && !setted_ub){
							ub[bound_ind] = (g[j-1]*(v1[j]-v_explore[j]) - g[j]*(v1[j-1]-v_explore[j-1])) / (v_explore[j-1]-v_explore[j]+v1[j]-v1[j-1]);
						} else if (current_value_zone==1 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v1[j]-v2[j]) - g[j]*(v1[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v1[j]-v1[j-1]);
							if (!setted_ub){
								ub[bound_ind] = lb[bound_ind];
							}
						} else if (current_value_zone==0 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v_explore[j]-v2[j]) - g[j]*(v_explore[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v_explore[j]-v_explore[j-1]);
						}
					}
				}
			}
			previous_value_zone = current_value_zone;
		}
	}
	
	#ifdef DEBUG
	fclose(prob_file);
	fclose(value_file);
	fclose(v_explore_file);
	std::cout<<"Exited backpropagate_value "<<std::endl;
	#endif
	return value[int(0.5*n)];
}

double DecisionPolicyDiscretePrior::backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2){
	/***
	 * Main function:
	 * backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2)
	 * 
	 * This function applies dynamic programing to determine the value
	 * of holding belief g at time t. It should be used under two different
	 * circumstances.
	 * 1) Iterate rho value until the value of g=0.5 at t=0 is 0
	 * 2) To compute the decision bounds in g space, once rho has been computed
	 * 
	 * This means that the value of belief g at time t is not stored
	 * during the execution. This is done to improve memory usage and
	 * execution time.
	 * 
	 * This function returns the value of g=0.5 at t=0.
	 * If compute_bounds=true it also sets the values of the bound arrays
	 * ub and lb.
	***/
	#ifdef DEBUG
	std::cout<<"Entered backpropagate_value with rho = "<<rho<<std::endl;
	#endif
	bool setted_ub = false;
	int previous_value_zone;
	int current_value_zone;
	int i, j, k, bound_ind, curr_invg, fut_invg, ind_prior;
	double norm_p;
	double p[n];
	double invg[2][n];
	
	this->rho = rho;
	curr_invg = 0;
	fut_invg = 1;
	#ifdef DEBUG
	FILE *details_file = fopen("details.txt","w");
	FILE *prob_file = fopen("prob.txt","w");
	FILE *value_file = fopen("value.txt","w");
	FILE *v_explore_file = fopen("v_explore.txt","w");
	#endif
	// Compute the value at the time limit T, where the subject must decide
	for (i=0;i<n;++i){
		// Value of deciding option 1
		v1[i] = reward*g[i]-penalty*(1.-g[i]) - (iti+(1.-g[i])*tp)*rho;
		// Value of deciding option 2
		v2[i] = reward*(1.-g[i])-penalty*g[i] - (iti+g[i]*tp)*rho;
		// Value of the belief g[i]
		value[i+(nT-1)*n] = v1[i]>=v2[i] ? v1[i] : v2[i];
		// We compute invg that is the x(t) that corresponds to having g[i] at time T
		// and store it to save computations
		invg[fut_invg][i] = g2x(t[nT-1],g[i]);
		#ifdef DEBUG
		if (i<n-1){
			fprintf(value_file,"%f\t",value[i+(nT-1)*n]);
		} else {
			fprintf(value_file,"%f\n",value[i+(nT-1)*n]);
		}
		#endif
	}
	if (compute_bounds){
		this->lb[bound_strides*(nT-1)] = 0.5;
		this->ub[bound_strides*(nT-1)] = 0.5;
	}
	
	// Dynamic programing loop that goes backward in time from T->0
	// Speed increase by precalculating values
	for (i=nT-2;i>=0;i--){
		#ifdef INFO
		if (i%100==0) std::cout<<i<<std::endl;
		#endif
		setted_ub = false;
		bound_ind = bound_strides*i;
		ub[bound_ind] = g[n-1];
		lb[bound_ind] = g[0];
		
		//Speed increase by reducing array access
		const double t_i = t[i];
		const double t_i1 = t[i+1];
		const double cost_rho_dt = (cost[i]+rho)*dt;
		const double inv_model_var_ti = 1./(model_var*t_i);
		const double inv_model_var_ti1 = 1./(model_var*t_i1);
		const double present_exp_factor = -0.5*inv_model_var_ti;
		const double future_exp_factor = -0.5*inv_model_var_ti1;
		for (j=0;j<n;++j){
			v_explore[j+i*n] = 0.;
			invg[curr_invg][j] = g2x(t_i,g[j],invg[fut_invg][j]);
			if (isnan(invg[curr_invg][j])){
				invg[curr_invg][j] = g2x(t_i,g[j],0.);
			}
			norm_p = 0.;
			double maxp = -INFINITY;
			// Speed increase by reducing array access and precalculating values
			const double present_x = invg[curr_invg][j];
			// Compute P(g(t+dt)|g(t))
			for (k=0;k<n;++k){
				const double future_x = invg[fut_invg][k];
				double present_a = 0.;
				double present_b = 0.;
				double future_a = 0.;
				double future_b = 0.;
				double deriv_future_a = 0.;
				double deriv_future_b = 0.;
				for (ind_prior=0; ind_prior<n_prior; ++ind_prior){
					const double mu_i = mu_prior[ind_prior];
					const double w_i = weight_prior[ind_prior];
					present_a+= w_i*exp(present_exp_factor*pow(mu_i*t_i-present_x,2));
					present_b+= w_i*exp(present_exp_factor*pow(mu_i*t_i+present_x,2));
					const double an = w_i*exp(future_exp_factor*pow(mu_i*t_i1-future_x,2));
					const double bn = w_i*exp(future_exp_factor*pow(mu_i*t_i1+future_x,2));
					future_a+= an;
					future_b+= bn;
					deriv_future_a+= (mu_i*t_i1-future_x)*inv_model_var_ti1*an;
					deriv_future_b+= (mu_i*t_i1+future_x)*inv_model_var_ti1*bn;
				}
				//~ p[k] = exp(future_exp_factor*pow(present_x*dt-(future_x-present_x)*t_i,2)/t_i/dt)*
						//~ pow(future_a+future_b,3)/(present_a+present_b)/(deriv_future_a*future_b+deriv_future_b*future_a);
				p[k] = future_exp_factor*pow(present_x*dt-(future_x-present_x)*t_i,2)/t_i/dt +
						3*log(future_a+future_b)-log(present_a+present_b)-log(deriv_future_a*future_b+deriv_future_b*future_a);
				maxp = p[k]>maxp ? p[k] : maxp;
				#ifdef DEBUG
				fprintf(details_file,"%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",k,t_i,p[k],future_x,present_x,future_a,future_b,deriv_future_a,deriv_future_b,present_a,present_b,future_exp_factor*pow(present_x*dt-(future_x-present_x)*t_i,2)/t_i/dt,3*log(future_a+future_b),log(present_a+present_b),log(deriv_future_a*future_b+deriv_future_b*future_a));
				#endif
			}
			// Then normalize and compute the value of exploring
			if (isinf(maxp)){
				for (k=0;k<n;++k){
					p[k] = (isinf(p[k]) && p[k]>0) ? 1. : 0;
					norm_p+= p[k];
					v_explore[j+i*n]+= p[k]*value[j+(i+1)*n];
				}
			} else {
				for (k=0;k<n;++k){
					p[k] = exp(p[k]-maxp);
					norm_p+= p[k];
					v_explore[j+i*n]+= p[k]*value[j+(i+1)*n];
				}
			}
			// Divide the value of exploring by the normalization factor and discount the cost and rho
			v_explore[j+i*n] = v_explore[j+i*n]/norm_p - cost_rho_dt;
			
			#ifdef DEBUG
			for (k=0;k<n-1;++k){
				fprintf(prob_file,"%f\t",p[k]/norm_p);
			}
			fprintf(prob_file,"%f\n",p[k]/norm_p);
			if (j<n-1){
				fprintf(v_explore_file,"%f\t",v_explore[j+i*n]);
			} else {
				fprintf(v_explore_file,"%f\n",v_explore[j+i*n]);
			}
			#endif
		}
		// Update temporal values
		curr_invg = (curr_invg+1)%2;
		fut_invg = (fut_invg+1)%2;
		// Value computation
		previous_value_zone = -1;
		current_value_zone = -1;
		for (j=0;j<n;++j){
			if (v1[j]>=v2[j] && v1[j]>=v_explore[j+i*n]){
				value[j+i*n] = v1[j];
				current_value_zone = 1;
			} else if (v2[j]>v1[j] && v2[j]>=v_explore[j+i*n]){
				value[j+i*n] = v2[j];
				current_value_zone = 2;
			} else if (v_explore[j+i*n]>v1[j] && v_explore[j+i*n]>v2[j]){
				value[j+i*n] = v_explore[j+i*n];
				current_value_zone = 0;
			}
			#ifdef DEBUG
			if (j<n-1){
				fprintf(value_file,"%f\t",value[j+i*n]);
			} else {
				fprintf(value_file,"%f\n",value[j+i*n]);
			}
			#endif
			// Bound computation
			if (compute_bounds){
				if (j>0 && j<n){
					if (std::abs(v1[j]-v_explore[j+i*n])<1e-8){
						if (!setted_ub){
							ub[bound_ind] = g[j];
							setted_ub = true;
						}
					} else if (std::abs(v2[j]-v_explore[j+i*n])<1e-8){
						lb[bound_ind] = g[j];
					} else if (current_value_zone!=previous_value_zone){
						if (current_value_zone==1 && previous_value_zone==0 && !setted_ub){
							ub[bound_ind] = (g[j-1]*(v1[j]-v_explore[j+i*n]) - g[j]*(v1[j-1]-v_explore[j-1+i*n])) / (v_explore[j-1+i*n]-v_explore[j+i*n]+v1[j]-v1[j-1]);
						} else if (current_value_zone==1 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v1[j]-v2[j]) - g[j]*(v1[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v1[j]-v1[j-1]);
							if (!setted_ub){
								ub[bound_ind] = lb[bound_ind];
							}
						} else if (current_value_zone==0 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v_explore[j+i*n]-v2[j]) - g[j]*(v_explore[j-1+i*n]-v2[j-1])) / (v2[j-1]-v2[j]+v_explore[j+i*n]-v_explore[j-1+i*n]);
						}
					}
				}
			}
			previous_value_zone = current_value_zone;
		}
	}
	
	#ifdef DEBUG
	fclose(prob_file);
	fclose(value_file);
	fclose(v_explore_file);
	std::cout<<"Exited backpropagate_value "<<std::endl;
	#endif
	return value[int(0.5*n)];
}

DecisionPolicyUnknownDiscreteVar::~DecisionPolicyUnknownDiscreteVar(){
	/***
	 * Destructor
	***/
	delete[] this->model_var;
	delete[] this->prior_var_prob;
	#ifdef DEBUG
	std::cout<<"Destroyed DecisionPolicyUnknownDiscreteVar instance"<<std::endl;
	#endif
}

void DecisionPolicyUnknownDiscreteVar::disp(){
	/***
	 * Print DecisionPolicyConjPrior instance's information
	***/
	std::cout<<"DecisionPolicyUnknownDiscreteVar instance = "<<this<<std::endl;
	std::cout<<"n_model_var = "<<n_model_var<<std::endl;
	int i;
	std::cout<<"model_var = [";
	for (i=0;i<n_model_var-1;++i){
		std::cout<<model_var[i]<<", ";
	}
	std::cout<<model_var[i]<<"]"<<std::endl;
	std::cout<<"prior_var_prob = [";
	for (i=0;i<n_model_var-1;++i){
		std::cout<<prior_var_prob[i]<<", ";
	}
	std::cout<<prior_var_prob[i]<<"]"<<std::endl;
	std::cout<<"prior_mu_mean = "<<prior_mu_mean<<std::endl;
	std::cout<<"prior_mu_var = "<<prior_mu_var<<std::endl;
	std::cout<<"dt = "<<dt<<std::endl;
	std::cout<<"dg = "<<dg<<std::endl;
	std::cout<<"T = "<<T<<std::endl;
	std::cout<<"cost = "<<cost<<std::endl;
	std::cout<<"reward = "<<reward<<std::endl;
	std::cout<<"iti = "<<iti<<std::endl;
	std::cout<<"tp = "<<tp<<std::endl;
	std::cout<<"rho = "<<rho<<std::endl;
	std::cout<<"n = "<<n<<std::endl;
	std::cout<<"nT = "<<nT<<std::endl;
	std::cout<<"t = "<<t<<std::endl;
	
	std::cout<<"owns_bounds = "<<owns_bounds<<std::endl;
	std::cout<<"bound_strides = "<<bound_strides<<std::endl;
	std::cout<<"ub = "<<ub<<std::endl;
	std::cout<<"lb = "<<lb<<std::endl;
}

double DecisionPolicyUnknownDiscreteVar::backpropagate_value(double rho, bool compute_bounds){
	/***
	 * Main function:
	 * backpropagate_value(double rho, bool compute_bounds)
	 * 
	 * This function applies dynamic programing to determine the value
	 * of holding belief g at time t. It should be used under two different
	 * circumstances.
	 * 1) Iterate rho value until the value of g=0.5 at t=0 is 0
	 * 2) To compute the decision bounds in g space, once rho has been computed
	 * 
	 * This means that the value of belief g at time t is not stored
	 * during the execution. This is done to improve memory usage and
	 * execution time.
	 * 
	 * This function returns the value of g=0.5 at t=0.
	 * If compute_bounds=true it also sets the values of the bound arrays
	 * ub and lb.
	***/
	#ifdef DEBUG
	std::cout<<"Entered backpropagate_value with rho = "<<rho<<std::endl;
	#endif
	bool setted_ub = false;
	int previous_value_zone, prior_ind, current_value_zone;
	int i, j, k, bound_ind, curr_invg, fut_invg;
	double post_var_t1[n_model_var], post_var_t[n_model_var], norm_p;
	double value[n], v1[n], v2[n], v_explore[n], p[n];
	double invg[2][n];
	
	this->rho = rho;
	curr_invg = 0;
	fut_invg = 1;
	#ifdef DEBUG
	FILE *details_file = fopen("uv_details.txt","w");
	FILE *prob_file = fopen("uv_prob.txt","w");
	FILE *value_file = fopen("uv_value.txt","w");
	FILE *v_explore_file = fopen("uv_v_explore.txt","w");
	#endif
	// Compute the value at the time limit T, where the subject must decide
	for (i=0;i<n;++i){
		// Value of deciding option 1
		v1[i] = reward*g[i]-penalty*(1.-g[i]) - (iti+(1.-g[i])*tp)*rho;
		// Value of deciding option 2
		v2[i] = reward*(1.-g[i])-penalty*g[i] - (iti+g[i]*tp)*rho;
		// Value of the belief g[i]
		value[i] = v1[i]>=v2[i] ? v1[i] : v2[i];
		// We compute invg that is the x(t) that corresponds to having g[i] at time T
		// and store it to save computations
		if (i==0){
			invg[fut_invg][i] = g2x(t[nT-1],g[i]);
		} else {
			invg[fut_invg][i] = g2x(t[nT-1],g[i],invg[fut_invg][i-1]);
		}
		#ifdef DEBUG
		if (i<n-1){
			fprintf(value_file,"%f\t",value[i]);
		} else {
			fprintf(value_file,"%f\n",value[i]);
		}
		#endif
	}
	if (compute_bounds){
		this->lb[bound_strides*(nT-1)] = 0.5;
		this->ub[bound_strides*(nT-1)] = 0.5;
	}
	
	
	for (j=0;j<n_model_var;++j){
		post_var_t1[j] = post_mu_var(j,this->t[nT-1]);
	}
	// Dynamic programing loop that goes backward in time from T->0
	// Speed increase by precalculating values
	for (i=nT-2;i>=0;i--){
		#ifdef INFO
		if (i%100==0) std::cout<<i<<std::endl;
		#endif
		setted_ub = false;
		bound_ind = bound_strides*i;
		ub[bound_ind] = g[n-1];
		lb[bound_ind] = g[0];
		
		//Speed increase by reducing array access
		const double t_i = t[i];
		const double cost_rho_dt = (cost[i]+rho)*dt;
		for (j=0;j<n_model_var;++j){
			post_var_t[j] = post_mu_var(j,t_i);
		}
		// Loop over g(t)
		for (j=0;j<n;++j){
			v_explore[j] = 0.;
			norm_p = 0.;
			invg[curr_invg][j] = g2x(t_i,g[j],invg[fut_invg][j]);
			// Speed increase by reducing array access and precalculating values
			const double* future_x = invg[fut_invg];
			
			// Compute P(g(t+dt)|g(t)) in two steps. First compute the exponents
			for (k=0;k<n;++k){
				double a = 0.;
				double b = 0.;
				double c = 0.;
				double d = 0.;
				for (prior_ind=0;prior_ind<n_model_var;++prior_ind){
					const double mut = post_mu_mean(prior_ind,t_i,invg[curr_invg][j]);
					const double mutdt = post_mu_mean(prior_ind,t[i+1],future_x[k]);
					const double vtdt = post_var_t1[prior_ind];
					const double vt_plus_v = (post_var_t[prior_ind]*dt + model_var[prior_ind])*dt;
					a+= prior_var_prob[prior_ind]*exp(-0.5*pow(future_x[k]-invg[curr_invg][j]-mut*dt,2)/(vt_plus_v));
					b+= prior_var_prob[prior_ind]*sqrt(vtdt);
					c+= prior_var_prob[prior_ind]*vtdt/model_var[prior_ind]*exp(-0.5*pow(mutdt,2)/vtdt);
					d+= prior_var_prob[prior_ind]*sqrt(vt_plus_v);
				}
				p[k] = a*b/(c*d);
				norm_p+= p[k];
				v_explore[j]+= p[k]*value[k];
			}
			// Divide the value of exploring by the normalization factor and discount the cost and rho
			v_explore[j] = v_explore[j]/norm_p - cost_rho_dt;
			
			#ifdef DEBUG
			for (k=0;k<n-1;++k){
				fprintf(prob_file,"%f\t",p[k]/norm_p);
			}
			fprintf(prob_file,"%f\n",p[k]/norm_p);
			if (j<n-1){
				fprintf(v_explore_file,"%f\t",v_explore[j]);
			} else {
				fprintf(v_explore_file,"%f\n",v_explore[j]);
			}
			#endif
		}
		// Update temporal values
		for (j=0;j<n_model_var;++j){
			post_var_t1[j] = post_var_t[j];
		}
		curr_invg = (curr_invg+1)%2;
		fut_invg = (fut_invg+1)%2;
		// Value computation
		previous_value_zone = -1;
		current_value_zone = -1;
		for (j=0;j<n;++j){
			if (v1[j]>=v2[j] && v1[j]>=v_explore[j]){
				value[j] = v1[j];
				current_value_zone = 1;
			} else if (v2[j]>v1[j] && v2[j]>=v_explore[j]){
				value[j] = v2[j];
				current_value_zone = 2;
			} else if (v_explore[j]>v1[j] && v_explore[j]>v2[j]){
				value[j] = v_explore[j];
				current_value_zone = 0;
			}
			#ifdef DEBUG
			if (j<n-1){
				fprintf(value_file,"%f\t",value[j]);
			} else {
				fprintf(value_file,"%f\n",value[j]);
			}
			#endif
			// Bound computation
			if (compute_bounds){
				if (j>0 && j<n){
					if (std::abs(v1[j]-v_explore[j])<1e-8){
						if (!setted_ub){
							ub[bound_ind] = g[j];
							setted_ub = true;
						}
					} else if (std::abs(v2[j]-v_explore[j])<1e-8){
						lb[bound_ind] = g[j];
					} else if (current_value_zone!=previous_value_zone){
						if (current_value_zone==1 && previous_value_zone==0 && !setted_ub){
							ub[bound_ind] = (g[j-1]*(v1[j]-v_explore[j]) - g[j]*(v1[j-1]-v_explore[j-1])) / (v_explore[j-1]-v_explore[j]+v1[j]-v1[j-1]);
						} else if (current_value_zone==1 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v1[j]-v2[j]) - g[j]*(v1[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v1[j]-v1[j-1]);
							if (!setted_ub){
								ub[bound_ind] = lb[bound_ind];
							}
						} else if (current_value_zone==0 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v_explore[j]-v2[j]) - g[j]*(v_explore[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v_explore[j]-v_explore[j-1]);
						}
					}
				}
			}
			previous_value_zone = current_value_zone;
		}
	}
	
	#ifdef DEBUG
	fclose(prob_file);
	fclose(value_file);
	fclose(v_explore_file);
	std::cout<<"Exited backpropagate_value "<<std::endl;
	#endif
	return value[int(0.5*n)];
}

double DecisionPolicyUnknownDiscreteVar::backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2){
	/***
	 * Main function:
	 * backpropagate_value(double rho, bool compute_bounds, double* value, double* v_explore, double* v1, double* v2)
	 * 
	 * This function applies dynamic programing to determine the value
	 * of holding belief g at time t. It should be used under two different
	 * circumstances.
	 * 1) Iterate rho value until the value of g=0.5 at t=0 is 0
	 * 2) To compute the decision bounds in g space, once rho has been computed
	 * 
	 * This means that the value of belief g at time t is not stored
	 * during the execution. This is done to improve memory usage and
	 * execution time.
	 * 
	 * This function returns the value of g=0.5 at t=0.
	 * If compute_bounds=true it also sets the values of the bound arrays
	 * ub and lb.
	***/
	#ifdef DEBUG
	std::cout<<"Entered backpropagate_value with rho = "<<rho<<std::endl;
	#endif
	bool setted_ub = false;
	int previous_value_zone, prior_ind, current_value_zone;
	int i, j, k, bound_ind, curr_invg, fut_invg;
	double post_var_t1[n_model_var], post_var_t[n_model_var], norm_p;
	double p[n];
	double invg[2][n];
	
	this->rho = rho;
	curr_invg = 0;
	fut_invg = 1;
	#ifdef DEBUG
	FILE *details_file = fopen("details.txt","w");
	FILE *prob_file = fopen("prob.txt","w");
	FILE *value_file = fopen("value.txt","w");
	FILE *v_explore_file = fopen("v_explore.txt","w");
	#endif
	// Compute the value at the time limit T, where the subject must decide
	for (i=0;i<n;++i){
		// Value of deciding option 1
		v1[i] = reward*g[i]-penalty*(1.-g[i]) - (iti+(1.-g[i])*tp)*rho;
		// Value of deciding option 2
		v2[i] = reward*(1.-g[i])-penalty*g[i] - (iti+g[i]*tp)*rho;
		// Value of the belief g[i]
		value[i+(nT-1)*n] = v1[i]>=v2[i] ? v1[i] : v2[i];
		// We compute invg that is the x(t) that corresponds to having g[i] at time T
		// and store it to save computations
		invg[fut_invg][i] = g2x(t[nT-1],g[i]);
		#ifdef DEBUG
		if (i<n-1){
			fprintf(value_file,"%f\t",value[i+(nT-1)*n]);
		} else {
			fprintf(value_file,"%f\n",value[i+(nT-1)*n]);
		}
		#endif
	}
	if (compute_bounds){
		this->lb[bound_strides*(nT-1)] = 0.5;
		this->ub[bound_strides*(nT-1)] = 0.5;
	}
	
	for (j=0;j<n_model_var;++j){
		post_var_t1[j] = post_mu_var(j,this->t[nT-1]);
	}
	// Dynamic programing loop that goes backward in time from T->0
	for (i=nT-2;i>=0;i--){
		#ifdef INFO
		if (i%100==0) std::cout<<i<<std::endl;
		#endif
		setted_ub = false;
		bound_ind = bound_strides*i;
		ub[bound_ind] = g[n-1];
		lb[bound_ind] = g[0];
		
		//Speed increase by reducing array access
		const double t_i = t[i];
		const double cost_rho_dt = (cost[i]+rho)*dt;
		for (j=0;j<n_model_var;++j){
			post_var_t[j] = post_mu_var(j,t_i);
		}
		// Loop over g(t)
		for (j=0;j<n;++j){
			v_explore[j+i*n] = 0.;
			norm_p = 0.;
			invg[curr_invg][j] = g2x(t_i,g[j],invg[fut_invg][j]);
			// Speed increase by reducing array access and precalculating values
			const double* future_x = invg[fut_invg];
			
			// Compute P(g(t+dt)|g(t)) in two steps. First compute the exponent
			for (k=0;k<n;++k){
				double a = 0.;
				double b = 0.;
				double c = 0.;
				double d = 0.;
				for (prior_ind=0;prior_ind<n_model_var;++prior_ind){
					const double mut = post_mu_mean(prior_ind,t_i,invg[curr_invg][j]);
					const double mutdt = post_mu_mean(prior_ind,t[i+1],future_x[k]);
					const double vtdt = post_var_t1[prior_ind];
					const double vt_plus_v = (post_var_t[prior_ind]*dt + model_var[prior_ind])*dt;
					a+= prior_var_prob[prior_ind]*exp(-0.5*pow(future_x[k]-invg[curr_invg][j]-mut*dt,2)/(vt_plus_v));
					b+= prior_var_prob[prior_ind]*sqrt(vtdt);
					c+= prior_var_prob[prior_ind]*vtdt/model_var[prior_ind]*exp(-0.5*pow(mutdt,2)/vtdt);
					d+= prior_var_prob[prior_ind]*sqrt(vt_plus_v);
				}
				p[k] = a*b/(c*d);
				norm_p+= p[k];
				v_explore[j+i*n]+= p[k]*value[k+(i+1)*n];
			}
			// Divide the value of exploring by the normalization factor and discount the cost and rho
			v_explore[j+i*n] = v_explore[j+i*n]/norm_p - cost_rho_dt;
			
			#ifdef DEBUG
			for (k=0;k<n-1;++k){
				fprintf(prob_file,"%f\t",p[k]/norm_p);
			}
			fprintf(prob_file,"%f\n",p[k]/norm_p);
			if (j<n-1){
				fprintf(v_explore_file,"%f\t",v_explore[j+i*n]);
			} else {
				fprintf(v_explore_file,"%f\n",v_explore[j+i*n]);
			}
			#endif
		}
		// Update temporal values
		for (j=0;j<n_model_var;++j){
			post_var_t1[j] = post_var_t[j];
		}
		curr_invg = (curr_invg+1)%2;
		fut_invg = (fut_invg+1)%2;
		// Value computation
		previous_value_zone = -1;
		current_value_zone = -1;
		for (j=0;j<n;++j){
			if (v1[j]>=v2[j] && v1[j]>=v_explore[j+i*n]){
				value[j+i*n] = v1[j];
				current_value_zone = 1;
			} else if (v2[j]>v1[j] && v2[j]>=v_explore[j+i*n]){
				value[j+i*n] = v2[j];
				current_value_zone = 2;
			} else if (v_explore[j+i*n]>v1[j] && v_explore[j+i*n]>v2[j]){
				value[j+i*n] = v_explore[j+i*n];
				current_value_zone = 0;
			}
			#ifdef DEBUG
			if (j<n-1){
				fprintf(value_file,"%f\t",value[j+i*n]);
			} else {
				fprintf(value_file,"%f\n",value[j+i*n]);
			}
			#endif
			// Bound computation
			if (compute_bounds){
				if (j>0 && j<n){
					if (std::abs(v1[j]-v_explore[j+i*n])<1e-8){
						if (!setted_ub){
							ub[bound_ind] = g[j];
							setted_ub = true;
						}
					} else if (std::abs(v2[j]-v_explore[j+i*n])<1e-8){
						lb[bound_ind] = g[j];
					} else if (current_value_zone!=previous_value_zone){
						if (current_value_zone==1 && previous_value_zone==0 && !setted_ub){
							ub[bound_ind] = (g[j-1]*(v1[j]-v_explore[j+i*n]) - g[j]*(v1[j-1]-v_explore[j-1+i*n])) / (v_explore[j-1+i*n]-v_explore[j+i*n]+v1[j]-v1[j-1]);
						} else if (current_value_zone==1 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v1[j]-v2[j]) - g[j]*(v1[j-1]-v2[j-1])) / (v2[j-1]-v2[j]+v1[j]-v1[j-1]);
							if (!setted_ub){
								ub[bound_ind] = lb[bound_ind];
							}
						} else if (current_value_zone==0 && previous_value_zone==2){
							lb[bound_ind] = (g[j-1]*(v_explore[j+i*n]-v2[j]) - g[j]*(v_explore[j-1+i*n]-v2[j-1])) / (v2[j-1]-v2[j]+v_explore[j+i*n]-v_explore[j-1+i*n]);
						}
					}
				}
			}
			previous_value_zone = current_value_zone;
		}
	}
	
	#ifdef DEBUG
	fclose(prob_file);
	fclose(value_file);
	fclose(v_explore_file);
	std::cout<<"Exited backpropagate_value "<<std::endl;
	#endif
	return value[int(0.5*n)];
}


