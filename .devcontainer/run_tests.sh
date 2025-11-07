# save stdout and stderr in a file for now
mkdir /temp_grader
cp -r /autograder/tests /temp_grader/tests
cp -r . /temp_grader
cd /temp_grader
python -m unittest -v tests/test_all.py > /workspaces/week-9/output.txt 2>&1
cd /workspaces/week-9
rm -r /temp_grader

# markdowntest in development ...

# cd /autograder

# add submission modules to PYTHONPATH:
# export PYTHONPATH=$PYTHONPATH:/workspaces/week-x

# run the tests:
# ~/miniconda3/bin/python run_tests.py 

# mv /autograder/test_results.md /workspaces/week-x/test_results.md
