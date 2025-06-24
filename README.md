# MD4DSP-m2python

## Prerequisites

- Anaconda Environment
- Python 3.11 (as it is the used version in the project)
- Libraries specified in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/franjmelchor/MD4DSP-m2python.git
    ```

2. Navigate to the project directory:
    ```bash
    cd your-project-directory
    ```
   
3. Crate a new conda environment:
   ```bash
   conda create --name md4dsp python=3.11 --yes
   ```
   
4. Deactivate any previous environment and activate the new one:
    ```bash
    conda deactivate
    conda activate md4dsp
    ```

5. Clean conda and pip caches:
    ```shell
    conda clean --all --yes
    pip cache purge
    ```
   This step will prevent you from retrieving libraries from the conda or pip caches, which may be incompatible with
   the project's requirements. If you are sure that the libraries in the cache are compatible, you can skip this step.

6. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

7. (Optional) Remove the environment created previously:
   ```bash
   conda deactivate
   conda remove --name md4dsp --all --yes
   ```

## Tests execution for data transformations

To run the data transformation tests, follow the next steps:

1. Run the data transformation tests:
   ```bash
    python3 test_data_transformations.py
    ```

2. Check the results in the logs:

Once the tests have finished, one log will be created for each execution of the python script. The test logs are located in the `logs/test_data_transformations` directory. By default, the logs are named as follows: `test_data_transformations_log_<number>.log`.
   
## Tests execution for contracts

To run the contract tests, follow the next steps:


1. Run the contract tests:
   ```bash
    python3 test_contracts.py
    ```

2. Check the results in the logs:

Once the tests have finished, one log will be created for each execution of the python script. The test logs are located in the `logs/test_contracts` directory. By default, the logs are named as follows: `test_contracts_log_<number>.log`.

## Tests execution for Data Smells

To run the data smell tests, follow the next steps:

1. Run the data smell tests:
   ```bash
    python3 test_data_smells.py
    ```
   
2. Check the results in the logs:

Once the tests have finished, one log will be created for each execution of the python script. The test logs are located in the `logs/test_data_smells` directory. By default, the logs are named as follows: `test_data_smells_log_<number>.log`.


## Project Structure

The project structure follows the next structure:

```bash
MD4DSP-m2python/
│
├── functions/
│ ├── contract_invariants.py
│ ├── contract_pre_post.py
│ └── data_transformations.py
│
├── helpers/
│ ├── auxiliar.py
│ ├── enumerations.py
│ ├── invariant_aux.py
│ ├── logger.py
│ └── transform_aux.py
│
├── logs/
│ └── transformations/
│   ├── ...
│   └── transformations_log_<number>.log
│ └── contracts/
│   ├── ...
│   └── contracts_log_<number>.log
│ └── dataProcessing/
│   ├── ...
│   └── dataProcessing_log_<number>.log
│ └── test_contracts/
│   ├── ...
│   └── test_contracts_log_<number>.log
│ └── test_data_transformations/
│   ├── ...
│   └── test_data_transformations_log_<number>.log
│
├── test_datasets/
│ ├── spotify_songs/
│   ├── spotify_songs.csv
│   └── readme.md
│
├── tests/
│ ├── contract_invariants/
│ │ ├── simple_test.py
│ │ └── tests_spotify_dataset.py
│ │
│ ├── contract_pre_post/
│ │ ├── simple_test.py
│ │ └── tests_spotify_dataset.py
│ │
│ └── data_transformations/
│   ├── simple_test.py
│   └── tests_spotify_dataset
│
├── .gitignore
├── README.md
├── requirements.txt
├── test_contracts.py
└── test_data_transformations.py

```

- **`functions/`**: contains the main functions of the project. The functions are divided into three files: `contract_invariants.py`, `contract_pre_post.py` and `data_transformations.py`. The first file contains the functions of the invariants, the second file contains the functions of the contracts, and the third file contains the functions of the data transformations.


- **`helpers/`**: contains auxiliary functions that are used in the main functions. The file `auxiliar.py` contains the auxiliary functions, `enumerations.py` contains the enumerations used in the project, `invariant_aux.py` contains the auxiliary functions of the invariants, `logger.py` contains the logger functions, and `transform_aux.py` contains the auxiliary functions of the data transformations.


- **`logs/`**: contains the logs of the tests. The logs are stored in the directory `test`.


- **`test_datasets/`**: contains the external datasets used in the tests. The datasets are divided into directories, and each directory contains the dataset and a readme file with the description of the dataset.


- **`tests/`**: contains the tests to make exhaustive evaluations of the functions. The tests are divided into five directories: `transformations`, `contracts`, `dataProcessing`, `test_contracts` and `test_data_transformations`. The first directory contains the execution logs of the transformations generated file. The second directory contains the execution logs of the contracts generated file. The third directory contains the execution logs of the dataProcessing generated file. The fourth directory contains the execution logs of the contract tests, and the fifth directory contains the execution logs of the data transformation tests.


- **`.gitignore`**: file that contains the files and directories to be ignored by Git.


- **`README.md`**: file that contains the documentation of the project.
  

- **`requirements.txt`**: file that contains the libraries needed to run the project.


- **`test_contracts.py`**: file to be executed to run the contract tests.


- **`test_data_transformations.py`**: file to be executed to run the data transformation tests.


## External Documentation
The external documentation of the project is available in the following link: https://unexes.sharepoint.com/:w:/s/PDI_i3lab/EYNMm7pMsX1HuIKz_PMWCi8Bl_ssrzRnvp3hQHimY363ng?e=d8Cvvh
  
## Authors
- Carlos Breuer Carrasco
- Carlos Cambero Rojas

## Questions
If you have any questions, please contact any of the authors.
