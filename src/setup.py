from numpy.distutils.core import setup, Extension

module1 = Extension('cdm',
                    sources = ['dmmodule.cpp', 'decision_model.cpp'],
                    extra_compile_args = ['-Wno-write-strings','-O2'])

setup (name = 'CDM',
       version = '1.0',
       description = 'Methods to compute the decision bounds and joint response time and confidence distributions for bayesian inference',
       ext_modules = [module1])
