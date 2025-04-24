# Dependencies: conda

# Create a machine learning environment

conda create -n triage_env python=3.6.13

# Activate conda environment
source activate triage_env

# If pip cant find the module try command below: 
#    pip install --upgrade pip 

pip install --upgrade pip
pip install -r requirements.txt

# link custom environment with modules to a kernel in jupyter
#ipython kernel install --user --name triage_env

source deactivate