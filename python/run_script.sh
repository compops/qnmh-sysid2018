# Run Python files when the container launches
python setup.py build_ext --inplace
wait

python run_script.py 1
wait

python run_script.py 2
wait

python run_script.py 3
wait

# Compress the results into one file
tar -zcvf /app/qnmh-sysid2018-results.tgz /app/results/*