#!/bin/bash
source scl_source enable devtoolset-6
source loadLSST.bash
setup lsst_sims
# Get the test data for afw
mkdir tmp; cd tmp; git clone https://github.com/lsst/afw.git ; cd .. 
ln -sf `pwd`/tmp/afw/tests /opt/lsst/stack/stack/miniconda3-4.3.21-10a4fa6/Linux64/afw/14.0-52-g19103d347+1/
pip install nose
pip install coveralls
pip install pylint
eups declare sims_GalSimInterface -r ${TRAVIS_BUILD_DIR} -t current
setup sims_GalSimInterface
cd ${TRAVIS_BUILD_DIR}
scons
nosetests -s --with-coverage --cover-package=lsst.sims.GalSimInterface
