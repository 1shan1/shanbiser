export PYTHONPATH=./:/home/pengshanzhen/caffe/python:$PYTHONPATH
LOG=/data_1/new_people/model5/log-`date +%Y-%m-%d-%H-%M-%S`.log 
/home/pengshanzhen/caffe/build/tools/caffe train -solver solver.prototxt -weights /home/pengshanzhen/high-quality-densitymap/bishe/model_iter_30000.caffemodel  2>&1  | tee $LOG $@






