/***
Python C++ extension that provides an interface between the c++
decision_model.cpp and decision_model.py package

Author: Luciano Paz
Year: 2016
***/
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#define CUSTOM_NPY_1_7_API_VERSION 7
#if NPY_API_VERSION<CUSTOM_NPY_1_7_API_VERSION
#define PyArray_SHAPE PyArray_DIMS
#endif
#undef CUSTOM_NPY_1_7_API_VERSION
#include "decision_model.hpp"

#include <cmath>
#include <cstdio>

DecisionModelDescriptor* get_descriptor(PyObject* py_dp){
	int prior_type;
	int n_model_var = 1;
	bool known_variance;
	PyObject* py_prior_mu_mean = NULL;
	PyObject* py_prior_mu_var = NULL;
	PyObject* py_mu_prior = NULL;
	PyObject* py_weight_prior = NULL;
	PyObject* py_model_var = PyObject_GetAttrString(py_dp,"model_var");
	PyObject* py_prior_var_prob = NULL;
	double* unknown_model_var_array = NULL;
	double* prior_var_prob = NULL;
	double known_model_var_value = -1.;
	if (!PyArray_Check(py_model_var)){
		known_variance = true;
		known_model_var_value = PyFloat_AsDouble(py_model_var);
		if (PyErr_Occurred()!=NULL){
			Py_XDECREF(py_model_var);
			return NULL;
		}
		Py_XDECREF(py_model_var);
	} else {
		if (PyArray_NDIM((PyArrayObject*)py_model_var)!=1){
			PyErr_SetString(PyExc_ValueError,"DecisionModel instance's attribute 'model_var' must be a float or a 1D numpy array.");
			Py_XDECREF(py_model_var);
			return NULL;
		} else {
			n_model_var = (int)PyArray_SHAPE((PyArrayObject*)py_model_var)[0];
			if (n_model_var==1){
				known_variance = true;
				known_model_var_value = ((double*)PyArray_DATA((PyArrayObject*)py_model_var))[0];
				Py_XDECREF(py_model_var);
			} else {
				known_variance = false;
				py_prior_var_prob = PyObject_GetAttrString(py_dp,"prior_var_prob");
				if (py_prior_var_prob==NULL){
					Py_XDECREF(py_model_var);
					PyErr_SetString(PyExc_AttributeError,"DecisionModel with unknown variance must have attribute 'prior_var_prob'. No such attribute was found.");
					return NULL;
				} else if (!PyArray_Check(py_prior_var_prob)){
					Py_XDECREF(py_model_var);
					Py_XDECREF(py_prior_var_prob);
					PyErr_SetString(PyExc_ValueError,"DecisionModel instance's attribute 'prior_var_prob' must be a 1D numpy array.");
					return NULL;
				} else if (PyArray_NDIM((PyArrayObject*)py_prior_var_prob)!=1){
					Py_XDECREF(py_model_var);
					Py_XDECREF(py_prior_var_prob);
					PyErr_SetString(PyExc_ValueError,"DecisionModel instance's attribute 'prior_var_prob' must be a 1D numpy array.");
					return NULL;
				} else if (PyArray_SHAPE((PyArrayObject*)py_prior_var_prob)[0]!=n_model_var){
					Py_XDECREF(py_model_var);
					Py_XDECREF(py_prior_var_prob);
					PyErr_SetString(PyExc_ValueError,"DecisionModel instance's attribute 'prior_var_prob' must have the same shape as attribute 'model_var'.");
					return NULL;
				}
				unknown_model_var_array = (double*)PyArray_DATA((PyArrayObject*)py_model_var);
				prior_var_prob = (double*)PyArray_DATA((PyArrayObject*)py_prior_var_prob);
				Py_XDECREF(py_model_var);
				Py_XDECREF(py_prior_var_prob);
			}
		}
	}
	PyObject* py_prior_type = PyObject_GetAttrString(py_dp,"prior_type");
	if (py_prior_type==NULL){
		prior_type = 1;
	} else {
		prior_type = int(PyInt_AS_LONG(py_prior_type));
		Py_XDECREF(py_prior_type);
	}
	if (prior_type==1){
		py_prior_mu_mean = PyObject_GetAttrString(py_dp,"prior_mu_mean");
		py_prior_mu_var = PyObject_GetAttrString(py_dp,"prior_mu_var");
	} else {
		if (!known_variance){
			PyErr_SetString(PyExc_ValueError,"Unknown variance with discrete symmetric prior is not yet supported.");
			return NULL;
		}
		py_mu_prior = PyObject_GetAttrString(py_dp,"mu_prior");
		py_weight_prior = PyObject_GetAttrString(py_dp,"weight_prior");
	}
	PyObject* py_n = PyObject_GetAttrString(py_dp,"n");
	PyObject* py_dt = PyObject_GetAttrString(py_dp,"dt");
	PyObject* py_T = PyObject_GetAttrString(py_dp,"T");
	PyObject* py_reward = PyObject_GetAttrString(py_dp,"reward");
	PyObject* py_penalty = PyObject_GetAttrString(py_dp,"penalty");
	PyObject* py_iti = PyObject_GetAttrString(py_dp,"iti");
	PyObject* py_tp = PyObject_GetAttrString(py_dp,"tp");
	PyObject* py_cost = PyObject_GetAttrString(py_dp,"cost");
	
	if (py_n==NULL || py_dt==NULL || py_T==NULL ||
		py_reward==NULL || py_penalty==NULL || py_iti==NULL || py_tp==NULL ||
		py_cost==NULL ||
		(prior_type==1 && (py_prior_mu_mean==NULL || py_prior_mu_var==NULL)) ||
		(prior_type==2 && (py_mu_prior==NULL || py_weight_prior==NULL))){
		PyErr_SetString(PyExc_ValueError, "Could not parse all DecisionModel property values");
		Py_XDECREF(py_prior_mu_mean);
		Py_XDECREF(py_prior_mu_var);
		Py_XDECREF(py_mu_prior);
		Py_XDECREF(py_weight_prior);
		Py_XDECREF(py_n);
		Py_XDECREF(py_dt);
		Py_XDECREF(py_T);
		Py_XDECREF(py_reward);
		Py_XDECREF(py_penalty);
		Py_XDECREF(py_iti);
		Py_XDECREF(py_tp);
		Py_XDECREF(py_cost);
		return NULL;
	}
	int n = int(PyInt_AS_LONG(py_n));
	double dt = PyFloat_AsDouble(py_dt);
	double T = PyFloat_AsDouble(py_T);
	double reward = PyFloat_AsDouble(py_reward);
	double penalty = PyFloat_AsDouble(py_penalty);
	double iti = PyFloat_AsDouble(py_iti);
	double tp = PyFloat_AsDouble(py_tp);
	double prior_mu_mean = NAN;
	double prior_mu_var = NAN;
	int n_prior = 0;
	int nT = (int)(T/dt)+1;
	double* mu_prior = NULL;
	double* weight_prior = NULL;
	if (prior_type==1){
		prior_mu_mean = PyFloat_AsDouble(py_prior_mu_mean);
		prior_mu_var = PyFloat_AsDouble(py_prior_mu_var);
	} else {
		if (!PyArray_Check(py_mu_prior) || !PyArray_Check(py_weight_prior)){
			PyErr_SetString(PyExc_TypeError,"Supplied mu_prior and weight_prior must be numpy arrays.");
		} else if (PyArray_NDIM((PyArrayObject*)py_mu_prior)!=1 || PyArray_NDIM((PyArrayObject*)py_weight_prior)!=1){
			PyErr_SetString(PyExc_ValueError,"Supplied mu_prior and weight_prior must be one dimensional numpy arrays.");
		} else if (PyArray_SHAPE((PyArrayObject*)py_mu_prior)[0]!=PyArray_SHAPE((PyArrayObject*)py_weight_prior)[0]){
			PyErr_SetString(PyExc_ValueError,"Supplied mu_prior and weight_prior must have the same number of elements.");
		} else if ((!PyArray_ISFLOAT((PyArrayObject*)py_mu_prior)) || (!PyArray_ISFLOAT((PyArrayObject*)py_weight_prior))){
			PyErr_SetString(PyExc_TypeError,"Supplied mu_prior and weight_prior must be of type np.float. Re-cast using astype.");
		} else {
			n_prior = (int) PyArray_SHAPE((PyArrayObject*)py_mu_prior)[0];
			mu_prior = (double*) PyArray_DATA((PyArrayObject*)py_mu_prior);
			weight_prior = (double*) PyArray_DATA((PyArrayObject*)py_weight_prior);
		}
	}
	double* cost = NULL;
	bool owns_cost = false;
	if (PyObject_IsInstance(py_cost,(PyObject*)(&PyArray_Type))){
		if (!PyArray_IsAnyScalar((PyArrayObject*)py_cost)){
			if (PyArray_NDIM((PyArrayObject*)py_cost)!=1){
				PyErr_SetString(PyExc_ValueError,"Supplied cost must be a scalar or a one dimensional numpy ndarray.");
			} else if ((int)PyArray_SHAPE((PyArrayObject*)py_cost)[0]!=nT-1){
				PyErr_SetString(PyExc_ValueError,"Supplied cost must have the length = len(t)-1.");
			} else if (!PyArray_ISFLOAT((PyArrayObject*)py_cost)){
				PyErr_SetString(PyExc_TypeError,"Supplied cost must be a floating point number or numpy array that can be casted to double.");
			} else {
				cost = ((double*)PyArray_DATA((PyArrayObject*)py_cost));
			}
		} else if (!PyArray_ISFLOAT((PyArrayObject*)py_cost)){
			PyErr_SetString(PyExc_TypeError,"Supplied cost must be a floating point number that can be casted to double.");
		} else {
			double _constant_cost = ((double*)PyArray_DATA((PyArrayObject*)py_cost))[0];
			owns_cost = true;
			cost = new double[nT-1];
			for (int i=0;i<nT-1;++i){
				cost[i] = _constant_cost;
			}
		}
	} else {
		double _constant_cost = PyFloat_AsDouble(py_cost);
		owns_cost = true;
		cost = new double[nT-1];
		for (int i=0;i<nT-1;++i){
			cost[i] = _constant_cost;
		}
	}
	// Check if an error occured while getting the c typed values from the python objects
	if (PyErr_Occurred()!=NULL){
		Py_XDECREF(py_prior_mu_mean);
		Py_XDECREF(py_prior_mu_var);
		Py_XDECREF(py_mu_prior);
		Py_XDECREF(py_weight_prior);
		Py_XDECREF(py_n);
		Py_XDECREF(py_dt);
		Py_XDECREF(py_T);
		Py_XDECREF(py_reward);
		Py_XDECREF(py_penalty);
		Py_XDECREF(py_iti);
		Py_XDECREF(py_tp);
		Py_XDECREF(py_cost);
		return NULL;
	}
	
	
	Py_XDECREF(py_prior_mu_mean);
	Py_XDECREF(py_prior_mu_var);
	Py_XDECREF(py_mu_prior);
	Py_XDECREF(py_weight_prior);
	Py_DECREF(py_n);
	Py_DECREF(py_dt);
	Py_DECREF(py_T);
	Py_DECREF(py_reward);
	Py_DECREF(py_penalty);
	Py_DECREF(py_iti);
	Py_DECREF(py_tp);
	Py_DECREF(py_cost);
	
	DecisionModelDescriptor* out;
	if (prior_type==1){
		if (known_variance){
			out = new DecisionModelDescriptor(known_model_var_value, prior_mu_mean, prior_mu_var,
							n, dt, T, reward, penalty, iti, tp, cost, owns_cost);
		} else {
			out = new DecisionModelDescriptor(n_model_var, unknown_model_var_array,
							prior_var_prob, prior_mu_mean, prior_mu_var, n, dt, T, reward,
							penalty, iti, tp, cost, owns_cost);
		}
	} else {
		out = new DecisionModelDescriptor(known_model_var_value, n_prior, mu_prior, weight_prior,
						n, dt, T, reward, penalty, iti, tp, cost, owns_cost);
	}
	return out;
}

/* method xbounds(DecisionModel, tolerance=1e-12, set_rho=True, set_bounds=True, return_values=False, root_bounds=None) */
static PyObject* dpmod_xbounds(PyObject* self, PyObject* args, PyObject* keywds){
	double tolerance = 1e-12;
	PyObject* py_dp;
	PyObject* py_set_rho_in_py_dp = Py_True;
	PyObject* py_touch_py_bounds = Py_True;
	PyObject* py_ret_values = Py_False;
	PyObject* py_root_bounds = Py_None;
	int set_rho_in_py_dp = 0;
	int touch_py_bounds = 0;
	int ret_values = 0;
	int must_dec_ref_py_bounds = 1;
	bool must_create_py_bounds = false;
	bool use_root_bounds = false;
	double lower_bound, upper_bound;
	PyObject* py_bounds = NULL;
	PyObject* py_out = NULL;
	PyObject* py_xub = NULL;
	PyObject* py_xlb = NULL;
	PyObject* py_value = NULL;
	PyObject* py_v_explore = NULL;
	PyObject* py_v1 = NULL;
	PyObject* py_v2 = NULL;
	DecisionModel* dp;
	DecisionModelDescriptor* dpd;
	
	
	static char* kwlist[] = {"decPol", "tolerance","set_rho","set_bounds","return_values","root_bounds", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|dOOOO", kwlist, &py_dp, &tolerance, &py_set_rho_in_py_dp, &py_touch_py_bounds, &py_ret_values, &py_root_bounds))
		return NULL;
	
	if (tolerance <= 0.0) {
		PyErr_SetString(PyExc_ValueError, "tolerance needs to be larger than 0");
		return NULL;
	}
	set_rho_in_py_dp = PyObject_IsTrue(py_set_rho_in_py_dp);
	if (set_rho_in_py_dp==-1) { // Failed to evaluate truth statement
		PyErr_SetString(PyExc_ValueError, "set_rho needs to evaluate to a valid truth statemente");
		return NULL;
	}
	touch_py_bounds = PyObject_IsTrue(py_touch_py_bounds);
	if (touch_py_bounds==-1) { // Failed to evaluate truth statement
		PyErr_SetString(PyExc_ValueError, "set_bounds needs to evaluate to a valid truth statemente");
		return NULL;
	}
	ret_values = PyObject_IsTrue(py_ret_values);
	if (ret_values==-1) { // Failed to evaluate truth statement
		PyErr_SetString(PyExc_ValueError, "return_values needs to evaluate to a valid truth statemente");
		return NULL;
	}
	if (py_root_bounds!=Py_None){
		use_root_bounds = true;
		if (!PyArg_ParseTuple(py_root_bounds,"dd",&lower_bound,&upper_bound)){
			if (PyErr_ExceptionMatches(PyExc_SystemError)){
				PyErr_SetString(PyExc_TypeError,"Supplied parameter 'root_bounds' must be None or a tuple with two elements. Both elements must be floats");
			} else if (PyErr_ExceptionMatches(PyExc_TypeError)){
				PyErr_SetString(PyExc_TypeError,"Supplied parameter 'root_bounds' must be None or a tuple with two elements. Both elements must be floats");
			}
			return NULL;
		}
	}
	
	dpd = get_descriptor(py_dp);
	if (dpd==NULL){
		// An error occurred while getting the descriptor and the error message was set within get_descriptor
		return NULL;
	}
	int nT = (int)(dpd->T/dpd->dt)+1;
	npy_intp py_nT[1] = {nT};
	
	if (!touch_py_bounds){
		dp = DecisionModel::create(*dpd);
	} else {
		npy_intp bounds_shape[2] = {2,nT};
		py_bounds = PyObject_GetAttrString(py_dp,"bounds");
		if (py_bounds==NULL){ // If the attribute bounds does not exist, create it
			PyErr_Clear(); // As we are handling the exception that py_dp has no attribute "bounds", we clear the exception state.
			must_create_py_bounds = true;
		} else if (!PyArray_Check((PyArrayObject*)py_bounds)){
			// Attribute 'bounds' in DecisionModel instance is not a numpy array. We must re create py_bounds
			Py_DECREF(py_bounds);
			must_create_py_bounds = true;
		} else if (PyArray_NDIM((PyArrayObject*)py_bounds)!=2){
			// Attribute 'bounds' in DecisionModel instance does not have the correct shape. We must re create py_bounds
			Py_DECREF(py_bounds);
			must_create_py_bounds = true;
		} else {
			for (int i=0;i<2;i++){
				if (bounds_shape[i]!=PyArray_SHAPE((PyArrayObject*)py_bounds)[i]){
					// Attribute 'bounds' in DecisionModel instance does not have the correct shape. We must re create py_bounds
					Py_DECREF(py_bounds);
					must_create_py_bounds = true;
					break;
				}
			}
		}
		if (must_create_py_bounds){
			py_bounds = PyArray_SimpleNew(2,bounds_shape,NPY_DOUBLE);
			if (py_bounds==NULL){
				PyErr_SetString(PyExc_MemoryError,"An error occured attempting to create the numpy array that would stores the DecisionModel instance's bounds attribute. Out of memory.");
				goto error_cleanup;
			}
			if (PyObject_SetAttrString(py_dp,"bounds", py_bounds)==-1){
				PyErr_SetString(PyExc_AttributeError,"Could not create and assign attribute bounds for the Decision policy instance");
				goto error_cleanup;
			}
			must_dec_ref_py_bounds = 0;
		}
		dp = DecisionModel::create(*dpd,
								(double*)PyArray_GETPTR2((PyArrayObject*)py_bounds,(npy_intp)0,(npy_intp)0),
								(double*)PyArray_GETPTR2((PyArrayObject*)py_bounds,(npy_intp)1,(npy_intp)0),
								((int) PyArray_STRIDE((PyArrayObject*)py_bounds,1))/sizeof(double)); // We divide by sizeof(double) because strides determines the number of bytes to stride in each dimension. As we cast the supplied void pointer to double*, each element has sizeof(double) bytes instead of 1 byte.
	}
	
	if (!ret_values){
		py_out = PyTuple_New(2);
		py_xub = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		py_xlb = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		if (py_out==NULL || py_xub==NULL || py_xlb==NULL){
			PyErr_SetString(PyExc_MemoryError, "Out of memory");
			delete(dp);
			goto error_cleanup;
		}
		PyTuple_SET_ITEM(py_out, 0, py_xub); // Steals a reference to py_xub so no dec_ref must be called on py_xub on cleanup
		PyTuple_SET_ITEM(py_out, 1, py_xlb); // Steals a reference to py_xlb so no dec_ref must be called on py_xlb on cleanup
	} else {
		npy_intp py_value_shape[2] = {dp->nT,dp->n};
		npy_intp py_v_explore_shape[2] = {dp->nT-1,dp->n};
		npy_intp py_v12_shape[1] = {dp->n};
		
		py_out = PyTuple_New(6);
		py_xub = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		py_xlb = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		py_value = PyArray_SimpleNew(2, py_value_shape, NPY_DOUBLE);
		py_v_explore = PyArray_SimpleNew(2, py_v_explore_shape, NPY_DOUBLE);
		py_v1 = PyArray_SimpleNew(1, py_v12_shape, NPY_DOUBLE);
		py_v2 = PyArray_SimpleNew(1, py_v12_shape, NPY_DOUBLE);
		if (py_out==NULL || py_xub==NULL || py_xlb==NULL || py_value==NULL || py_v_explore==NULL || py_v1==NULL || py_v2==NULL){
			PyErr_SetString(PyExc_MemoryError, "Out of memory");
			delete(dp);
			goto error_cleanup;
		}
		PyTuple_SET_ITEM(py_out, 0, py_xub); // Steals a reference to py_xub so no dec_ref must be called on py_xub on cleanup
		PyTuple_SET_ITEM(py_out, 1, py_xlb); // Steals a reference to py_xlb so no dec_ref must be called on py_xlb on cleanup
		PyTuple_SET_ITEM(py_out, 2, py_value); // Steals a reference to py_value so no dec_ref must be called on py_value on cleanup
		PyTuple_SET_ITEM(py_out, 3, py_v_explore); // Steals a reference to py_v_explore so no dec_ref must be called on py_v_explore on cleanup
		PyTuple_SET_ITEM(py_out, 4, py_v1); // Steals a reference to py_v1 so no dec_ref must be called on py_v1 on cleanup
		PyTuple_SET_ITEM(py_out, 5, py_v2); // Steals a reference to py_v2 so no dec_ref must be called on py_v2 on cleanup
	}
	
	if (use_root_bounds){
		dp->iterate_rho_value(tolerance,lower_bound,upper_bound);
	} else {
		dp->iterate_rho_value(tolerance);
	}
	if (set_rho_in_py_dp){
		if (PyObject_SetAttrString(py_dp,"rho",Py_BuildValue("d",dp->rho))==-1){
			PyErr_SetString(PyExc_ValueError, "Could not set DecisionModel property rho");
			delete(dp);
			goto error_cleanup;
		}
	}
	// Backpropagate and compute bounds in the diffusion space
	if (!ret_values) {
		dp->backpropagate_value();
	} else {
		dp->backpropagate_value(dp->rho,true,
				(double*) PyArray_DATA((PyArrayObject*) py_value),
				(double*) PyArray_DATA((PyArrayObject*) py_v_explore),
				(double*) PyArray_DATA((PyArrayObject*) py_v1),
				(double*) PyArray_DATA((PyArrayObject*) py_v2));
	}
	dp->x_ubound((double*) PyArray_DATA((PyArrayObject*) py_xub));
	dp->x_lbound((double*) PyArray_DATA((PyArrayObject*) py_xlb));
	
	// normal_cleanup
	delete(dpd);
	delete(dp);
	if (must_dec_ref_py_bounds) Py_XDECREF(py_bounds);
	return py_out;

error_cleanup:
	delete(dpd);
	if (must_dec_ref_py_bounds) Py_XDECREF(py_bounds);
	Py_XDECREF(py_xub);
	Py_XDECREF(py_xlb);
	Py_XDECREF(py_value);
	Py_XDECREF(py_v_explore);
	Py_XDECREF(py_v1);
	Py_XDECREF(py_v2);
	Py_XDECREF(py_out);
	return NULL;
}

/* method xbounds_fixed_rho(DecisionModel, rho=None, set_bounds=False, return_values=False) */
static PyObject* dpmod_xbounds_fixed_rho(PyObject* self, PyObject* args, PyObject* keywds){
	PyObject* py_dp;
	PyObject* py_touch_py_bounds = Py_False;
	PyObject* py_ret_values = Py_False;
	PyObject* py_rho = Py_None;
	double rho;
	int touch_py_bounds = 0;
	int ret_values = 0;
	int must_dec_ref_py_bounds = 1;
	bool must_create_py_bounds = false;
	PyObject* py_bounds = NULL;
	PyObject* py_out = NULL;
	PyObject* py_xub = NULL;
	PyObject* py_xlb = NULL;
	PyObject* py_value = NULL;
	PyObject* py_v_explore = NULL;
	PyObject* py_v1 = NULL;
	PyObject* py_v2 = NULL;
	DecisionModel* dp;
	DecisionModelDescriptor* dpd;
	
	
	static char* kwlist[] = {"decPol","rho","set_bounds","return_values", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|OOO", kwlist, &py_dp, &py_rho, &py_touch_py_bounds, &py_ret_values))
		return NULL;
	
	if (py_rho==Py_None) { // Use dp rho value
		py_rho = PyObject_GetAttrString(py_dp,"rho");
		if (py_rho==NULL){
			PyErr_SetString(PyExc_ValueError, "DecisionModel instance has no rho attribute. You should set it or pass rho as the second parameter to the 'value' function");
			return NULL;
		}
		rho = PyFloat_AsDouble(py_rho);
		if (PyErr_Occurred()!=NULL){
			return NULL;
		}
	} else {
		rho = PyFloat_AsDouble(py_rho);
		if (PyErr_Occurred()!=NULL){
			return NULL;
		}
	}
	touch_py_bounds = PyObject_IsTrue(py_touch_py_bounds);
	if (touch_py_bounds==-1) { // Failed to evaluate truth statement
		PyErr_SetString(PyExc_ValueError, "set_bounds needs to evaluate to a valid truth statemente");
		return NULL;
	}
	ret_values = PyObject_IsTrue(py_ret_values);
	if (ret_values==-1) { // Failed to evaluate truth statement
		PyErr_SetString(PyExc_ValueError, "return_values needs to evaluate to a valid truth statemente");
		return NULL;
	}
	
	dpd = get_descriptor(py_dp);
	if (dpd==NULL){
		// An error occurred while getting the descriptor and the error message was set within get_descriptor
		return NULL;
	}
	int nT = (int)(dpd->T/dpd->dt)+1;
	npy_intp py_nT[1] = {nT};
	
	if (!touch_py_bounds){
		dp = DecisionModel::create(*dpd);
	} else {
		npy_intp bounds_shape[2] = {2,nT};
		py_bounds = PyObject_GetAttrString(py_dp,"bounds");
		if (py_bounds==NULL){ // If the attribute bounds does not exist, create it
			PyErr_Clear(); // As we are handling the exception that py_dp has no attribute "bounds", we clear the exception state.
			must_create_py_bounds = true;
		} else if (!PyArray_Check((PyArrayObject*)py_bounds)){
			// Attribute 'bounds' in DecisionModel instance is not a numpy array. We must re create py_bounds
			Py_DECREF(py_bounds);
			must_create_py_bounds = true;
		} else if (PyArray_NDIM((PyArrayObject*)py_bounds)!=2){
			// Attribute 'bounds' in DecisionModel instance does not have the correct shape. We must re create py_bounds
			Py_DECREF(py_bounds);
			must_create_py_bounds = true;
		} else {
			for (int i=0;i<2;i++){
				if (bounds_shape[i]!=PyArray_SHAPE((PyArrayObject*)py_bounds)[i]){
					// Attribute 'bounds' in DecisionModel instance does not have the correct shape. We must re create py_bounds
					Py_DECREF(py_bounds);
					must_create_py_bounds = true;
					break;
				}
			}
		}
		if (must_create_py_bounds){
			py_bounds = PyArray_SimpleNew(2,bounds_shape,NPY_DOUBLE);
			if (py_bounds==NULL){
				PyErr_SetString(PyExc_MemoryError,"An error occured attempting to create the numpy array that would stores the DecisionModel instance's bounds attribute. Out of memory.");
				goto error_cleanup;
			}
			if (PyObject_SetAttrString(py_dp,"bounds", py_bounds)==-1){
				PyErr_SetString(PyExc_AttributeError,"Could not create and assign attribute bounds for the Decision policy instance");
				goto error_cleanup;
			}
			must_dec_ref_py_bounds = 0;
		}
		dp = DecisionModel::create(*dpd,
								(double*)PyArray_GETPTR2((PyArrayObject*)py_bounds,(npy_intp)0,(npy_intp)0),
								(double*)PyArray_GETPTR2((PyArrayObject*)py_bounds,(npy_intp)1,(npy_intp)0),
								((int) PyArray_STRIDE((PyArrayObject*)py_bounds,1))/sizeof(double)); // We divide by sizeof(double) because strides determines the number of bytes to stride in each dimension. As we cast the supplied void pointer to double*, each element has sizeof(double) bytes instead of 1 byte.
	}
	dp->rho = rho;
	
	if (!ret_values){
		py_out = PyTuple_New(2);
		py_xub = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		py_xlb = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		if (py_out==NULL || py_xub==NULL || py_xlb==NULL){
			PyErr_SetString(PyExc_MemoryError, "Out of memory");
			delete(dp);
			goto error_cleanup;
		}
		PyTuple_SET_ITEM(py_out, 0, py_xub); // Steals a reference to py_xub so no dec_ref must be called on py_xub on cleanup
		PyTuple_SET_ITEM(py_out, 1, py_xlb); // Steals a reference to py_xlb so no dec_ref must be called on py_xlb on cleanup
	} else {
		npy_intp py_value_shape[2] = {dp->nT,dp->n};
		npy_intp py_v_explore_shape[2] = {dp->nT-1,dp->n};
		npy_intp py_v12_shape[1] = {dp->n};
		
		py_out = PyTuple_New(6);
		py_xub = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		py_xlb = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
		py_value = PyArray_SimpleNew(2, py_value_shape, NPY_DOUBLE);
		py_v_explore = PyArray_SimpleNew(2, py_v_explore_shape, NPY_DOUBLE);
		py_v1 = PyArray_SimpleNew(1, py_v12_shape, NPY_DOUBLE);
		py_v2 = PyArray_SimpleNew(1, py_v12_shape, NPY_DOUBLE);
		if (py_out==NULL || py_xub==NULL || py_xlb==NULL || py_value==NULL || py_v_explore==NULL || py_v1==NULL || py_v2==NULL){
			PyErr_SetString(PyExc_MemoryError, "Out of memory");
			delete(dp);
			goto error_cleanup;
		}
		PyTuple_SET_ITEM(py_out, 0, py_xub); // Steals a reference to py_xub so no dec_ref must be called on py_xub on cleanup
		PyTuple_SET_ITEM(py_out, 1, py_xlb); // Steals a reference to py_xlb so no dec_ref must be called on py_xlb on cleanup
		PyTuple_SET_ITEM(py_out, 2, py_value); // Steals a reference to py_value so no dec_ref must be called on py_value on cleanup
		PyTuple_SET_ITEM(py_out, 3, py_v_explore); // Steals a reference to py_v_explore so no dec_ref must be called on py_v_explore on cleanup
		PyTuple_SET_ITEM(py_out, 4, py_v1); // Steals a reference to py_v1 so no dec_ref must be called on py_v1 on cleanup
		PyTuple_SET_ITEM(py_out, 5, py_v2); // Steals a reference to py_v2 so no dec_ref must be called on py_v2 on cleanup
	}
	
	// Backpropagate and compute bounds in the diffusion space
	if (!ret_values) {
		dp->backpropagate_value(dp->rho,true);
	} else {
		dp->backpropagate_value(dp->rho,true,
				(double*) PyArray_DATA((PyArrayObject*) py_value),
				(double*) PyArray_DATA((PyArrayObject*) py_v_explore),
				(double*) PyArray_DATA((PyArrayObject*) py_v1),
				(double*) PyArray_DATA((PyArrayObject*) py_v2));
	}
	dp->x_ubound((double*) PyArray_DATA((PyArrayObject*) py_xub));
	dp->x_lbound((double*) PyArray_DATA((PyArrayObject*) py_xlb));
	
	// normal_cleanup
	delete(dpd);
	delete(dp);
	if (must_dec_ref_py_bounds) Py_XDECREF(py_bounds);
	return py_out;

error_cleanup:
	delete(dpd);
	if (must_dec_ref_py_bounds) Py_XDECREF(py_bounds);
	Py_XDECREF(py_xub);
	Py_XDECREF(py_xlb);
	Py_XDECREF(py_value);
	Py_XDECREF(py_v_explore);
	Py_XDECREF(py_v1);
	Py_XDECREF(py_v2);
	Py_XDECREF(py_out);
	return NULL;
}

/* method values(DecisionModel, rho=None) */
static PyObject* dpmod_values(PyObject* self, PyObject* args, PyObject* keywds){
	PyObject* py_dp;
	PyObject* py_rho = Py_None;
	double rho;
	PyObject* py_out = NULL;
	PyObject* py_value = NULL;
	PyObject* py_v_explore = NULL;
	PyObject* py_v1 = NULL;
	PyObject* py_v2 = NULL;
	DecisionModel* dp;
	DecisionModelDescriptor* dpd;
	
	static char* kwlist[] = {"decPol", "rho", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|O", kwlist, &py_dp, &py_rho))
		return NULL;
	
	if (py_rho==Py_None) { // Use dp rho value
		py_rho = PyObject_GetAttrString(py_dp,"rho");
		if (py_rho==NULL){
			PyErr_SetString(PyExc_ValueError, "DecisionModel instance has no rho attribute. You should set it or pass rho as the second parameter to the 'value' function");
			return NULL;
		}
		rho = PyFloat_AsDouble(py_rho);
		if (PyErr_Occurred()!=NULL){
			return NULL;
		}
	} else {
		rho = PyFloat_AsDouble(py_rho);
		if (PyErr_Occurred()!=NULL){
			return NULL;
		}
	}
	
	dpd = get_descriptor(py_dp);
	if (dpd==NULL){
		// An error occurred while getting the descriptor and the error message was set within get_descriptor
		return NULL;
	}
	dp = DecisionModel::create(*dpd);
	dp->rho = rho;
	
	npy_intp py_value_shape[2] = {dp->nT,dp->n};
	npy_intp py_v_explore_shape[2] = {dp->nT-1,dp->n};
	npy_intp py_v12_shape[1] = {dp->n};
	
	py_out = PyTuple_New(4);
	py_value = PyArray_SimpleNew(2, py_value_shape, NPY_DOUBLE);
	py_v_explore = PyArray_SimpleNew(2, py_v_explore_shape, NPY_DOUBLE);
	py_v1 = PyArray_SimpleNew(1, py_v12_shape, NPY_DOUBLE);
	py_v2 = PyArray_SimpleNew(1, py_v12_shape, NPY_DOUBLE);
	
	if (py_out==NULL || py_value==NULL || py_v_explore==NULL || py_v1==NULL || py_v2==NULL){
		PyErr_SetString(PyExc_MemoryError, "Out of memory");
		goto error_cleanup;
	}
	PyTuple_SET_ITEM(py_out, 0, py_value); // Steals a reference to py_value so no dec_ref must be called on py_value on cleanup
	PyTuple_SET_ITEM(py_out, 1, py_v_explore); // Steals a reference to py_v_explore so no dec_ref must be called on py_v_explore on cleanup
	PyTuple_SET_ITEM(py_out, 2, py_v1); // Steals a reference to py_v1 so no dec_ref must be called on py_v1 on cleanup
	PyTuple_SET_ITEM(py_out, 3, py_v2); // Steals a reference to py_v2 so no dec_ref must be called on py_v2 on cleanup
		
	dp->backpropagate_value(dp->rho,false,
			(double*) PyArray_DATA((PyArrayObject*) py_value),
			(double*) PyArray_DATA((PyArrayObject*) py_v_explore),
			(double*) PyArray_DATA((PyArrayObject*) py_v1),
			(double*) PyArray_DATA((PyArrayObject*) py_v2));
	
	delete(dp);
	delete(dpd);
	return py_out;

error_cleanup:
	delete(dp);
	delete(dpd);
	Py_XDECREF(py_value);
	Py_XDECREF(py_v_explore);
	Py_XDECREF(py_v1);
	Py_XDECREF(py_v2);
	Py_XDECREF(py_out);
	return NULL;
}

/* method rt(DecisionModel, mu, bounds=None) */
static PyObject* dpmod_rt(PyObject* self, PyObject* args, PyObject* keywds){
	PyObject* py_dp;
	PyObject* py_bounds = Py_None;
	PyObject* py_model_var = Py_None;
	PyObject* py_rho;
	double mu, rho, model_var;
	bool must_decref_py_bounds = false;
	PyArrayObject* py_xub, *py_xlb;
	
	PyObject* py_out = NULL;
	PyObject* py_g1 = NULL;
	PyObject* py_g2 = NULL;
	DecisionModel* dp;
	DecisionModelDescriptor* dpd;
	
	static char* kwlist[] = {"decPol", "mu", "model_var", "bounds", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "Od|OO", kwlist, &py_dp, &mu, &py_model_var, &py_bounds))
		return NULL;
	
	dpd = get_descriptor(py_dp);
	if (dpd==NULL){
		// An error occurred while getting the descriptor and the error message was set within get_descriptor
		return NULL;
	}
	npy_intp py_nT[1] = {dpd->nT};
	if (py_model_var==Py_None){
		if (dpd->known_variance()){
			model_var = dpd->model_var[0];
		} else {
			PyErr_SetString(PyExc_ValueError, "When the DecisionModel instance has unknown variance (many discrete prior variances), this method's 'model_var' input becomes mandatory");
			goto early_error_cleanup;
		}
	} else {
		model_var = PyFloat_AsDouble(py_model_var);
	}
	
	py_rho = PyObject_GetAttrString(py_dp,"rho");
	if (py_rho==NULL){
		PyErr_SetString(PyExc_ValueError, "DecisionModel instance has no rho value set");
		goto early_error_cleanup;
	}
	rho = PyFloat_AsDouble(py_rho);
	if (PyErr_Occurred()!=NULL){
		Py_XDECREF(py_rho);
		goto early_error_cleanup;
	}
	Py_DECREF(py_rho);
	
	dp = DecisionModel::create(*dpd);
	dp->rho = rho;
	
	if (py_bounds==Py_None) { // Compute xbounds if they are not provided
		PyObject* args2 = PyTuple_Pack(1,py_dp);
		py_bounds = dpmod_xbounds(self,args2,NULL);
		Py_DECREF(args2);
		if (py_bounds==NULL){
			return NULL;
		}
		must_decref_py_bounds = true;
	}
	
	if (!PyArg_ParseTuple(py_bounds,"O!O!", &PyArray_Type, &py_xub, &PyArray_Type, &py_xlb))
		goto error_cleanup;
	if (PyArray_NDIM(py_xub)!=1 || PyArray_NDIM(py_xlb)!=1){
		// Attribute 'bounds' in DecisionModel instance does not have the correct shape. We must re create py_bounds
		PyErr_SetString(PyExc_ValueError,"Supplied bounds must be numpy arrays with one dimension");
		goto error_cleanup;
	} else if (PyArray_SHAPE(py_xub)[0]!=py_nT[0] || PyArray_SHAPE(py_xlb)[0]!=py_nT[0]) {
		PyErr_Format(PyExc_ValueError,"Supplied bounds must be numpy arrays of shape (%d)",dpd->nT);
		goto error_cleanup;
	}
	
	py_out = PyTuple_New(2);
	py_g1 = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
	py_g2 = PyArray_SimpleNew(1, py_nT, NPY_DOUBLE);
	
	if (py_out==NULL || py_g1==NULL || py_g2==NULL){
		PyErr_SetString(PyExc_MemoryError, "Out of memory");
		goto error_cleanup;
	}
	PyTuple_SET_ITEM(py_out, 0, py_g1); // Steals a reference
	PyTuple_SET_ITEM(py_out, 1, py_g2); // Steals a reference
		
	dp->rt(mu,model_var,(double*) PyArray_DATA((PyArrayObject*) py_g1),
			 (double*) PyArray_DATA((PyArrayObject*) py_g2),
			 (double*) PyArray_DATA((PyArrayObject*) py_xub),
			 (double*) PyArray_DATA((PyArrayObject*) py_xlb));
	
	delete(dp);
	delete(dpd);
	if (must_decref_py_bounds){
		Py_XDECREF(py_bounds);
		// The elements in py_bounds are also decref'ed so only calling
		// decref on the py_bounds instance is sufficient
	}
	return py_out;

error_cleanup:
	delete(dp);
	if (must_decref_py_bounds){
		Py_XDECREF(py_bounds);
		// The elements in py_bounds are also decref'ed so only calling
		// decref on the py_bounds instance is sufficient
	}
	Py_XDECREF(py_g1);
	Py_XDECREF(py_g2);
	Py_XDECREF(py_out);
	goto early_error_cleanup;

early_error_cleanup:
	delete(dpd);
	return NULL;
}

/* method rt(DecisionModel, mu, bounds=None) */
static PyObject* dpmod_fpt_conf_matrix(PyObject* self, PyObject* args, PyObject* keywds){
	/***
	* This method takes the confidence report as a function of time (CR)
	* and converts it to a matrix. This matrix is filled with zeroes
	* except for the entries that are touched by the plot of CR. The
	* value of each entry is given by the first passage time probability
	* density (FPT).
	* 
	* fpt_conf_matrix(self,first_passage_time, confidence_response, confidence_partition=100)
	* 
	* Input:
	* 	first_passage_time: A 2D numpy array of doubles with the FPT.
	* 		Axis=0 represents different responses and axis=1 time. The
	* 		shape of axis=1 must be equal to self.nT.
	* 	confidence_response: A 2D numpy array of doubles with the CR.
	* 		It must have the same shape as the first_passage_time input.
	* 	confidence_partition: An int that determines the number of
	* 		discrete confidence report values, uniformly distributed in
	* 		the interval [0,1], that will be used to construct the
	* 		output array.
	* 
	* Output: A 3D numpy array of shape:
	* 	(first_passage_time.shape[0],confidence_partition,first_passage_time.shape[1])
	* 	The values are such that np.sum(output,axis=1)==first_passage_time
	* 	and calling imshow(output[0]>0,origin='lower') and
	* 	plot(confidence_partition) will show almost overlapping curves.
	* 
	***/
	
	PyObject* py_dp;
	PyArrayObject* py_first_passage_time;
	PyArrayObject* py_confidence_response;
	int confidence_partition = 100;
	int first_passage_time_strides[2], confidence_response_strides[2];
	
	PyObject* py_out = NULL;
	
	static char* kwlist[] = {"decPol", "first_passage_time", "confidence_response", "confidence_partition", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OO!O!|i", kwlist,
									&py_dp, &PyArray_Type, &py_first_passage_time, &PyArray_Type, &py_confidence_response, &confidence_partition))
		return NULL;
	
	if (confidence_partition<=0){
		PyErr_SetString(PyExc_ValueError, "confidence_partition must be greater than 0");
		return NULL;
	}
	
	if (PyArray_NDIM(py_first_passage_time)!=2 || PyArray_NDIM(py_confidence_response)!=2){
		PyErr_SetString(PyExc_ValueError, "first_passage_time and confidence_response must be 2D numpy arrays");
		return NULL;
	}
	if (PyArray_DESCR(py_first_passage_time)->type_num != NPY_DOUBLE || PyArray_DESCR(py_confidence_response)->type_num != NPY_DOUBLE){
		PyErr_SetString(PyExc_ValueError, "first_passage_time and confidence_response must contain double precision floating point values");
		return NULL;
	}
	int n_alternatives = (int) PyArray_SHAPE(py_first_passage_time)[0];
	int array_nT = (int) PyArray_SHAPE(py_first_passage_time)[1];
	if ((int) PyArray_SHAPE(py_confidence_response)[0]!=n_alternatives || (int) PyArray_SHAPE(py_confidence_response)[1]!=array_nT){
		PyErr_SetString(PyExc_ValueError, "first_passage_time and confidence_response must have the same shape");
		return NULL;
	}
	for (int i=0; i<2; ++i){
		first_passage_time_strides[i] = ((int) PyArray_STRIDES(py_first_passage_time)[i])/sizeof(double);
		confidence_response_strides[i] = ((int) PyArray_STRIDES(py_confidence_response)[i])/sizeof(double);
	}
	double* first_passage_time = (double*) PyArray_DATA(py_first_passage_time);
	double* confidence_response = (double*) PyArray_DATA(py_confidence_response);
	
	npy_intp out_shape[3] = {(npy_intp)n_alternatives, (npy_intp)array_nT, (npy_intp)confidence_partition};
	py_out = PyArray_SimpleNew(3, out_shape, NPY_DOUBLE);
	if (py_out==NULL){
		PyErr_SetString(PyExc_MemoryError, "Cannot allocate output to memory");
		return NULL;
	}
	double* out = (double*) PyArray_DATA((PyArrayObject*)py_out);
	for (int i=0;i<n_alternatives*array_nT*confidence_partition;++i){
		out[i] = 0.;
	}
	
	DecisionModelDescriptor* dpd = get_descriptor(py_dp);
	if (dpd==NULL){
		// An error occurred while getting the descriptor and the error message was set within get_descriptor
		Py_DECREF(py_out);
		return NULL;
	}
	if (dpd->nT!=array_nT){
		PyErr_Format(PyExc_ValueError, "first_passage_time and confidence_response have an inconsistent shape in axis=1. The shape should be equal to the DecisionModel instance's nT = %d",dpd->nT);
		delete dpd;
		Py_DECREF(py_out);
		return NULL;
	}
	
	DecisionModel* dp = DecisionModel::create(*dpd);
	
	dp->fpt_conf_matrix(first_passage_time, first_passage_time_strides, n_alternatives, confidence_partition, confidence_response, confidence_response_strides, out);
	
	delete dp;
	delete dpd;
	return PyArray_SwapAxes((PyArrayObject*) py_out, 1, 2);
}

static PyObject* dpmod_testsuite(PyObject* self, PyObject* args, PyObject* keywds){
	if (!PyArg_ParseTuple(args, ""))
		return NULL;
	
	int i;
	PyObject* dict1 = PyDict_New();
	PyObject* dict2 = PyDict_New();
	PyObject* dict3 = PyDict_New();
	
	double internal_var = 10;
	double external_var = 100;
	double model_var = internal_var+external_var;
	double prior_mu_mean = 0.;
	double prior_mu_var = 1000.;
	int n = 101;
	double dt = 1e-2;
	double T = 1.;
	int nT = (int)(T/dt)+1;
	double reward = 1.;
	double penalty = 0.;
	double iti = 3.;
	double tp = 0.;
	double cost = 0.;
	double cost_pointer[nT];
	for (i=0;i<nT;++i){
		cost_pointer[i] = cost;
	}
	int n_prior = 10;
	npy_intp discrete_prior_dims[] = {n_prior};
	PyObject* py_mu_prior = PyArray_SimpleNew(1,discrete_prior_dims,NPY_DOUBLE);
	PyObject* py_weight_prior = PyArray_SimpleNew(1,discrete_prior_dims,NPY_DOUBLE);
	double* mu_prior = (double*) PyArray_DATA((PyArrayObject*)py_mu_prior);
	double* weight_prior = (double*) PyArray_DATA((PyArrayObject*)py_weight_prior);
	double norm = 0.;
	for (i=0;i<n_prior;++i){
		mu_prior[i] = 20./((double)n_prior-1)*((double)i);
		if (mu_prior[i]==0){
			mu_prior[i] = 1e-10;
		}
		weight_prior[i] = exp(-0.5*mu_prior[i]*mu_prior[i]/100.);
		//~ mu_prior[i] = 1000./((double)n_prior-1)*((double)i);
		//~ weight_prior[i] = exp(-0.5*mu_prior[i]*mu_prior[i]/1000.);
		norm+= weight_prior[i];
	}
	for (i=0;i<n_prior;++i){
		weight_prior[i]*= (0.5/norm);
	}
	int n_model_var = 3;
	npy_intp unknown_var_dims[] = {n_model_var};
	PyObject* py_unknown_var = PyArray_SimpleNew(1,unknown_var_dims,NPY_DOUBLE);
	PyObject* py_unknown_external_var = PyArray_SimpleNew(1,unknown_var_dims,NPY_DOUBLE);
	PyObject* py_prior_var_prob = PyArray_SimpleNew(1,unknown_var_dims,NPY_DOUBLE);
	double* unknown_var = (double*) PyArray_DATA((PyArrayObject*)py_unknown_var);
	double* unknown_external_var = (double*) PyArray_DATA((PyArrayObject*)py_unknown_external_var);
	double* prior_var_prob = (double*) PyArray_DATA((PyArrayObject*)py_prior_var_prob);
	for (i=0;i<n_model_var;++i){
		unknown_external_var[i] = (double)(50.*(i+1));
		unknown_var[i] = internal_var+unknown_external_var[i];
		prior_var_prob[i] = 1./double(n_model_var);
	}
	
	// Will output 3 dictionaries with the settings of each testsuite
	PyDict_SetItemString(dict1, "n", PyInt_FromLong((long) n));
	PyDict_SetItemString(dict1, "internal_var", PyFloat_FromDouble(internal_var));
	PyDict_SetItemString(dict1, "external_var", PyFloat_FromDouble(external_var));
	PyDict_SetItemString(dict1, "model_var", PyFloat_FromDouble(model_var));
	PyDict_SetItemString(dict1, "prior_mu_mean", PyFloat_FromDouble(prior_mu_mean));
	PyDict_SetItemString(dict1, "prior_mu_var", PyFloat_FromDouble(prior_mu_var));
	PyDict_SetItemString(dict1, "dt", PyFloat_FromDouble(dt));
	PyDict_SetItemString(dict1, "T", PyFloat_FromDouble(T));
	PyDict_SetItemString(dict1, "reward", PyFloat_FromDouble(reward));
	PyDict_SetItemString(dict1, "penalty", PyFloat_FromDouble(penalty));
	PyDict_SetItemString(dict1, "iti", PyFloat_FromDouble(iti));
	PyDict_SetItemString(dict1, "tp", PyFloat_FromDouble(tp));
	PyDict_SetItemString(dict1, "internal_var", PyFloat_FromDouble(internal_var));
	PyDict_SetItemString(dict1, "cost", PyFloat_FromDouble(cost));
	
	PyDict_SetItemString(dict2, "n", PyInt_FromLong((long) n));
	PyDict_SetItemString(dict2, "internal_var", PyFloat_FromDouble(internal_var));
	PyDict_SetItemString(dict2, "external_var", PyFloat_FromDouble(external_var));
	PyDict_SetItemString(dict2, "model_var", PyFloat_FromDouble(model_var));
	PyDict_SetItemString(dict2, "discrete_prior", Py_BuildValue("(OO)",py_mu_prior,py_weight_prior));
	PyDict_SetItemString(dict2, "dt", PyFloat_FromDouble(dt));
	PyDict_SetItemString(dict2, "T", PyFloat_FromDouble(T));
	PyDict_SetItemString(dict2, "reward", PyFloat_FromDouble(reward));
	PyDict_SetItemString(dict2, "penalty", PyFloat_FromDouble(penalty));
	PyDict_SetItemString(dict2, "iti", PyFloat_FromDouble(iti));
	PyDict_SetItemString(dict2, "tp", PyFloat_FromDouble(tp));
	PyDict_SetItemString(dict2, "internal_var", PyFloat_FromDouble(internal_var));
	PyDict_SetItemString(dict2, "cost", PyFloat_FromDouble(cost));
	
	PyDict_SetItemString(dict3, "n", PyInt_FromLong((long) n));
	PyDict_SetItemString(dict3, "internal_var", PyFloat_FromDouble(internal_var));
	PyDict_SetItemString(dict3, "external_var", py_unknown_external_var);
	PyDict_SetItemString(dict3, "prior_var_prob", py_prior_var_prob);
	PyDict_SetItemString(dict3, "model_var", py_unknown_var);
	PyDict_SetItemString(dict3, "prior_mu_mean", PyFloat_FromDouble(prior_mu_mean));
	PyDict_SetItemString(dict3, "prior_mu_var", PyFloat_FromDouble(prior_mu_var));
	PyDict_SetItemString(dict3, "dt", PyFloat_FromDouble(dt));
	PyDict_SetItemString(dict3, "T", PyFloat_FromDouble(T));
	PyDict_SetItemString(dict3, "reward", PyFloat_FromDouble(reward));
	PyDict_SetItemString(dict3, "penalty", PyFloat_FromDouble(penalty));
	PyDict_SetItemString(dict3, "iti", PyFloat_FromDouble(iti));
	PyDict_SetItemString(dict3, "tp", PyFloat_FromDouble(tp));
	PyDict_SetItemString(dict3, "internal_var", PyFloat_FromDouble(internal_var));
	PyDict_SetItemString(dict3, "cost", PyFloat_FromDouble(cost));
	
	// Create DecisionModelDescriptor
	DecisionModelDescriptor* dpd1 = new DecisionModelDescriptor(model_var, prior_mu_mean, prior_mu_var,
						n, dt, T, reward, penalty, iti, tp, cost_pointer, false);
	DecisionModelDescriptor* dpd2 = new DecisionModelDescriptor(model_var, n_prior, mu_prior, weight_prior,
						n, dt, T, reward, penalty, iti, tp, cost_pointer, false);
	DecisionModelDescriptor* dpd3 = new DecisionModelDescriptor(n_model_var, unknown_var, prior_var_prob,
						prior_mu_mean, prior_mu_var,n, dt, T, reward, penalty, iti, tp, cost_pointer, false);
	// Create DecisionModel
	DecisionModel* dp1 = DecisionModel::create(*dpd1);
	DecisionModel* dp2 = DecisionModel::create(*dpd2);
	DecisionModel* dp3 = DecisionModel::create(*dpd3);
	
	// Create output arrays
	double t = 0.;
	int noutput = 100;
	npy_intp output_dims[] = {noutput};
	PyObject* py_x = PyArray_SimpleNew(1,output_dims,NPY_DOUBLE);
	PyObject* py_g1 = PyArray_SimpleNew(1,output_dims,NPY_DOUBLE);
	PyObject* py_g2 = PyArray_SimpleNew(1,output_dims,NPY_DOUBLE);
	PyObject* py_g3 = PyArray_SimpleNew(1,output_dims,NPY_DOUBLE);
	PyObject* py_dg1 = PyArray_SimpleNew(1,output_dims,NPY_DOUBLE);
	PyObject* py_dg2 = PyArray_SimpleNew(1,output_dims,NPY_DOUBLE);
	PyObject* py_dg3 = PyArray_SimpleNew(1,output_dims,NPY_DOUBLE);
	PyObject* py_x1 = PyArray_SimpleNew(1,output_dims,NPY_DOUBLE);
	PyObject* py_x2 = PyArray_SimpleNew(1,output_dims,NPY_DOUBLE);
	PyObject* py_x3 = PyArray_SimpleNew(1,output_dims,NPY_DOUBLE);
	double* x = (double*) PyArray_DATA((PyArrayObject*)py_x);
	double* g1 = (double*) PyArray_DATA((PyArrayObject*)py_g1);
	double* g2 = (double*) PyArray_DATA((PyArrayObject*)py_g2);
	double* g3 = (double*) PyArray_DATA((PyArrayObject*)py_g3);
	double* dg1 = (double*) PyArray_DATA((PyArrayObject*)py_dg1);
	double* dg2 = (double*) PyArray_DATA((PyArrayObject*)py_dg2);
	double* dg3 = (double*) PyArray_DATA((PyArrayObject*)py_dg3);
	double* x1 = (double*) PyArray_DATA((PyArrayObject*)py_x1);
	double* x2 = (double*) PyArray_DATA((PyArrayObject*)py_x2);
	double* x3 = (double*) PyArray_DATA((PyArrayObject*)py_x3);
	for (i=0;i<noutput;++i){
		x[i] = -30. + 60./(double(noutput-1))*double(i);
		g1[i] = dp1->x2g(t,x[i]);
		g2[i] = dp2->x2g(t,x[i]);
		g3[i] = dp3->x2g(t,x[i]);
		dg1[i] = dp1->dx2g(t,x[i]);
		dg2[i] = dp2->dx2g(t,x[i]);
		dg3[i] = dp3->dx2g(t,x[i]);
		x1[i] = dp1->g2x(t,g1[i]);
		x2[i] = dp2->g2x(t,g2[i]);
		x3[i] = dp3->g2x(t,g3[i]);
	}
	
	delete dpd1;
	delete dpd2;
	delete dpd3;
	delete dp1;
	delete dp2;
	delete dp3;
	return Py_BuildValue("(OOOOOOOOOOOOOO)", dict1, dict2, dict3, PyFloat_FromDouble(t), py_x, py_g1, py_g2, py_g3, py_dg1, py_dg2, py_dg3, py_x1, py_x2, py_x3);
}

static PyMethodDef DPMethods[] = {
    {"xbounds", (PyCFunction) dpmod_xbounds, METH_VARARGS | METH_KEYWORDS,
     "Computes the decision bounds in x(t) space (i.e. the accumulated sensory input space)\n\n  (xub, xlb) = xbounds(dp, tolerance=1e-12, set_rho=False, set_bounds=False, return_values=False, root_bounds=None)\n\n(xub, xlb, value, v_explore, v1, v2) = xbounds(dp, ..., return_values=True)\n\nComputes the decision bounds for a DecisionModel instance specified in 'dp'.\nThis function is more memory and computationally efficient than calling dp.invert_belief();dp.value_dp(); xb = dp.belief_bound_to_x_bound(b); from python. Another difference is that this function returns a tuple of (upper_bound, lower_bound) instead of a numpy array whose first element is upper_bound and second element is lower_bound.\n'tolerance' is a float that indicates the tolerance when searching for the rho value that yields value[int(n/2)]=0.\n'set_rho' must be an expression whose 'truthness' can be evaluated. If set_rho is True, the rho attribute in the python dp object will be set to the rho value obtained after iteration. If false, it will not be set.\n'set_bounds' must be an expression whose 'truthness' can be evaluated. If set_bounds is True, the python DecisionModel object's ´bounds´ attribute will be set to the upper and lower bounds in g space computed in the c++ instance. If false, it will do nothing.\nIf 'return_values' evaluates to True, then the function returns four extra numpy arrays: value, v_explore, v1 and v2. 'value' is an nT by n shaped array that holds the value of a given g at time t. 'v_explore' has shape nT-1 by n and holds the value of exploring at time t with a given g. v1 and v2 are values of immediately deciding for option 1 or 2, and are one dimensional arrays with n elements.\n'root_bounds' must be a tuple of two elements: (lower_bound, upper_bound). Both 'lower_bound' and 'upper_bound' must be floats that represent the lower and upper bounds in which to perform the root finding of rho.\n\n"},
    {"xbounds_fixed_rho", (PyCFunction) dpmod_xbounds_fixed_rho, METH_VARARGS | METH_KEYWORDS,
     "Computes the decision bounds in x(t) space (i.e. the accumulated sensory input space) without iterating the value of rho\n\n  (xub, xlb) = xbounds_fixed_rho(dp, rho=None, set_bounds=False, return_values=False)\n\n(xub, xlb, value, v_explore, v1, v2) = xbounds_fixed_rho(dp, ..., return_values=True)\n\nComputes the decision bounds for a DecisionModel instance specified in 'dp' for a given rho value.\nThis function is more memory and computationally efficient than calling dp.invert_belief();dp.value_dp(); xb = dp.belief_bound_to_x_bound(b); from python. Another difference is that this function returns a tuple of (upper_bound, lower_bound) instead of a numpy array whose first element is upper_bound and second element is lower_bound.\n'rho' is the fixed reward rate value used to compute the decision bounds and values. If rho=None, then the DecisionModel instance's rho is used.\n'set_bounds' must be an expression whose 'truthness' can be evaluated. If set_bounds is True, the python DecisionModel object's ´bounds´ attribute will be set to the upper and lower bounds in g space computed in the c++ instance. If false, it will do nothing.\nIf 'return_values' evaluates to True, then the function returns four extra numpy arrays: value, v_explore, v1 and v2. 'value' is an nT by n shaped array that holds the value of a given g at time t. 'v_explore' has shape nT-1 by n and holds the value of exploring at time t with a given g. v1 and v2 are values of immediately deciding for option 1 or 2, and are one dimensional arrays with n elements.\n\n"},
    {"values", (PyCFunction) dpmod_values, METH_VARARGS | METH_KEYWORDS,
     "Computes the values for a given reward rate, rho, and DecisionModel parameters.\n\n(value, v_explore, v1, v2) = values(dp,rho=None)\n\nComputes the value for a given belief g as a function of time for a supplied reward rate, rho. If rho is set to None, then the DecisionModel instance's rho attribute will be used.\nThis function is more memory and computationally efficient than calling dp.invert_belief();dp.value_dp(); from python. The function returns a tuple of four numpy arrays: value, v_explore, v1 and v2. 'value' is an nT by n shaped array that holds the value of a given g at time t. 'v_explore' has shape nT-1 by n and holds the value of exploring at time t with a given g. v1 and v2 are values of immediately deciding for option 1 or 2, and are one dimensional arrays with n elements.\n"},
    {"rt", (PyCFunction) dpmod_rt, METH_VARARGS | METH_KEYWORDS,
     "Computes the rt distribution for a given drift rate, mu, variance rate, DecisionModel parameters and decision bounds in x space, bounds.\n\n(g1, g2) = values(dp,mu,model_var=None,bounds=None)\n\nInput:\n  dp:        DecisionModel instace\n  mu:        Float that encodes the diffusion drift rate (net evidence).\n  bounds:    By default None. If None, the method internally calls the function xbounds(dp) with default parameter values to compute the decision bounds in x space. To avoid this, supply a tuple (xub,xlb) as the one that is returned by the function xbounds. xub and xlb must be one dimensional numpy arrays with the same elements as dp.t.\n  model_var: An input that is mandatory for DecisionModel instances with unknown variance that represents the variance rate of the diffusion process. If the DecisionModel instance has known variance and model_var is None, the instance's model_var is used as the diffusion's variance rate.\n\nOutput:\n  (g1,g2):   Each of these outputs are 1D numpy arrays with dp.nT number of elements representing the first passage time probability density for option 1 and 2 respectively.\n\n"},
    {"fpt_conf_matrix", (PyCFunction) dpmod_fpt_conf_matrix, METH_VARARGS | METH_KEYWORDS,
     "This method takes the confidence report as a function of time (CR)\nand converts it to a matrix. This matrix is filled with zeroes\nexcept for the entries that are touched by the plot of CR. The\nvalue of each entry is given by the first passage time probability\ndensity (FPT).\n\n\nfpt_conf_matrix(self,first_passage_time, confidence_response, confidence_partition=100)\n\nInput:\n	first_passage_time: A 2D numpy array of doubles with the FPT.\n		Axis=0 represents different responses and axis=1 time. The\n		shape of axis=1 must be equal to self.nT.\n	confidence_response: A 2D numpy array of doubles with the CR.\n		It must have the same shape as the first_passage_time input.\n	confidence_partition: An int that determines the number of\n		discrete confidence report values, uniformly distributed in\n		the interval [0,1], that will be used to construct the\n		output array.\n\nOutput: A 3D numpy array of shape:\n	(first_passage_time.shape[0],confidence_partition,first_passage_time.shape[1])\n	The values are such that np.sum(output,axis=1)==first_passage_time\n	and calling imshow(output[0]>0,origin='lower') and\n	plot(confidence_partition) will show almost overlapping curves.\n\n"},
    {"testsuite", (PyCFunction) dpmod_testsuite, METH_VARARGS,""},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
    /* module initialisation for Python 3 */
    static struct PyModuleDef dpmodule = {
       PyModuleDef_HEAD_INIT,
       "dp",   /* name of module */
       "Module to compute the decision bounds and values for bayesian inference",
       -1,
       DPMethods
    };

    PyMODINIT_FUNC PyInit_dp(void)
    {
        PyObject *m = PyModule_Create(&dpmodule);
        import_array();
        return m;
    }
#else
    /* module initialisation for Python 2 */
    PyMODINIT_FUNC initdp(void)
    {
        Py_InitModule("dp", DPMethods);
        import_array();
    }
#endif
