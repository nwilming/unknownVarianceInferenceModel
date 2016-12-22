# unknownVarianceInferenceModel

This is a work in progress, in a relatively early stage, not a properly packaged python bundle.

The code in this repo implements a Cognitive Computational decision making model based on optimal inference. It implements an extension to the model presented in "J. Drugowitsch, R. Moreno-Bote, A. K. Churchland, M. N. Shadlen, and A. Pouget. The cost of accumulating evidence in perceptual decision making. J. Neurosci., 32(11):3612–28, 2012.". The repo implements a flexible way to fit the model parameters to experimental behavioral data.

If you use the code in this repo for an academic purpose, we please ask that you cite the following paper [COMING SOON]

INSTALATION
===========

This is not a package yet, it is a collection of scripts and a custom c++ extension that must be locally compiled. You must then place the .py files in the PYTHONPATH to be able to import them into other appliactions, or you must cd to their containing src folder. The custom c++ extension must be compiled using

python setup.py install --prefix=./ && mv lib/python2.7/site-packages/*.so . && rm -r build/ lib/

This will build the cdm.so shared library and place it in the current folder so it can be imported by the rest of the .py functions in this repo.

USAGE
=====

This repo has 3 scripts that are the ones intended to be used:

1. fits_module.py: Implements the Fitter class that is used to fit the model parameters to experimental data. It can be called from the terminal. Type python fits_module.py --help for a detailed list of options to supply
2. data_io.py: Implements the SubjectSession class that is used to load the experimental data by subject, session and experiment.
3. decision_model.py: Implements the DecisionModel class that holds the entire implementation of the cognitive computational model.

In order to be able to perform fits you must read through fits_module.py's command line help, but more importantly, must write a proper experiment_details.txt file and place the experimental data in a certain file structure.

The required file structure is as follows:

.
+--raw_data_dir:
|   +--Experiment1Name:
|        +-- Subject1Name: Raw subject1 experiment1 data for all sessions
|        +-- Subject1Name: Raw subject2 experiment1 data for all sessions
|   +--Experiment2Name:
|         +-- Subject1Name: Raw subject1 experiment2 data for all sessions
|         +-- Subject2Name: Raw subject2 experiment2 data for all sessions

IMPORTANT! The sessions must be clearly identifiable from the data file name.

The experiment_details.txt specifies the raw_data_dir's full path and the handled experiment names with their properties. It must have a structure as shown below:

```
\# Comments can be placed with the # character before
raw_data_dir: raw_data_dir/

begin experiment TestExperimentName
  \# This must be something like keyName: keyValue
	\#----- DecisionModel keys \-----#
	tp: 0.
	T: 5.
	iti: 3. 
	dt: 1e-3
	reward: 1
	penalty: 0
	n: 101
	\# external_var can be a float or a list of floats (e.g. [10.,20.,30.])
	external_var: 1000. # The external_var units must be stimulus variance over time (e.g. contrast\**2/s).
	\# prior_var_prob is only used if external_var is a list.
	\# It represents the prior probability of each variance.
	\# The probabilities are normalized when they are loaded.
	prior_var_prob: [0.3,0.3,0.3]
	\#----- Fitter keys -----#
	ISI: 0.04
	rt_cutoff: 14.
	distractor: 0.
	forced_non_decision_time: 0.
	rt_measured_from_stim_end: False \# If False, rt are measured from stim start. If True, they are measured from the stim end
	time_available_to_respond: inf
	is_binary_confidence: True \# Can be True or False to indicate if the confidence data is binary or not
	\#----- IO keys -----#
	\# session_parser: A lambda expression or callable that is able to parse the session number from the data filename
	\# file_extension: A str with the data file extension, can be .txt or .mat
	\# time_conversion_to_seconds: A float that will be multiplied to the RT data to convert the units to seconds
	\# data_structure: A json structure with fields "delimiter" and "data_fields"
	\#				 "data_fields" must also be a json structure whose
	\#				 field names equal to the measured fields, and whose
	\#				 values must be lambda expressions or callables that
	\#				 will extract the field name data from the loaded
	\#				 raw data
end experiment TestExperimentName
```
