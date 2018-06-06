# Run Python files when the container launches
python setup.py build_ext --inplace
wait

python run_script.py $EXPERIMENT $FULLRUN
wait