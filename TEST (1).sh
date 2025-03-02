#!/bin/bash -l
#SBATCH --job-name=GCN_Training_test      # Job name
#SBATCH --output=output_%j.log            # Output log file (with job ID)
#SBATCH --error=error_%j.log              # Error log file
#SBATCH --time=01:00:00                   # Time limit (hh:mm:ss)
#SBATCH --mem-per-cpu=16G                  # Memory request
#SBATCH --cpus-per-task=4                 # Number of CPU cores
#SBATCH --mail-type=END,FAIL              # Notifications for job status
#SBATCH --mail-user=felix.vankerschaver@student.kuleuven.be   # Your email

cd $VSC_DATA

# Load the required environment module
module load intel/2021a
module load Python/3.9.5-GCCcore-10.3.0

# Create and activate a virtual environment
python3 -m venv gcn_env
source gcn_env/bin/activate

# Download get-pip.py to reinstall pip properly
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py --force-reinstall

# Install specific compatible versions
pip install torch==2.6.0
pip install numpy==1.22.4  # Changed to be compatible with scipy
pip install scipy==1.7.0
pip install pandas==2.2.2
pip install scikit-learn==1.4.2
pip install matplotlib==3.4.2
pip install tabulate==0.9.0

# Install torch-geometric with correct CUDA version
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.6.0+cu121.html
pip install torch-geometric==2.6.1

# Run the GCN model
python GCN.py