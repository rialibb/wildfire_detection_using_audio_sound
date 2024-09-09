#!/bin/bash
#SBATCH --job-name=train-cnn-crnn  # create a short name for your job
#SBATCH --nodes=1                  # node count
#SBATCH --ntasks=1                 # total number of tasks across all nodes
#SBATCH --cpus-per-task=1          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                   # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1               # number of gpus per node
#SBATCH --time=02:00:00            # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin          # send mail when job begins
#SBATCH --mail-type=end            # send mail when job ends
#SBATCH --mail-type=fail           # send mail if job fails
#SBATCH --mail-user=mathis.pastorelli@student-cs.Fr

#SBATCH --partition=gpu

module purge
module load python/3.9.10/intel-20.0.4.304
source $HOME/Workspace/etude-de-cas-feux-de-foret/.env/bin/activate

python main.py
