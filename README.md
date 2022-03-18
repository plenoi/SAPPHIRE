# SAPPHIRE
SAPPHIRE: a stacking-based ensemble learning framework for accurate prediction of thermophilic proteins.

## Dependency
The packages that this program depends on is 
`scikit-learn==0.24.1 or higher`. You can run following command in terminal.<br>
`pip install scikit-learn==0.24.1`

## How to use SAPPHIRE
1. Copy your fasta file into `./input` and change the name to seq.fasta<br>
2. Extract PSSM features from your fasta file using <b>pssmpro</b> (https://github.com/deeprob/pssmpro). Please follow their tutorial carefully.<br>
3. After finish feature extraction, copy the required PSSM features for SAPPHIRE into the `./input` folder. These required PSSM features are listed as the following.
  - psssm_composition.csv
  - rpm_pssm.csv
  - s_fpssm.csv 
4. Run command<br>
`python sapphire.py`
5. The result including sequence Header, label and probability will be saved in `./output/predicted_result.csv`
