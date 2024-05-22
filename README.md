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
   
3. Crate a new conda enviroment:
   ```bash
   conda create --name md4dsp python=3.11 -y
   ```
   
4. Deactivate any previous environment and activate the new one:
    ```bash
    $ conda deactivate
    $ conda activate md4dsp
    ```

5. Clean conda and pip caches:
    ```shell
    $ conda clean --all -y
    $ pip cache purge
    ```
   This step will prevent you from retrieving libraries from the conda or pip caches, which may be incompatible with
   the proyect's requirements. If you are sure that the libraries in the cache are compatible, you can skip this step.


6. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

7. (Optional) Remove the environment created previously:
   ```bash
   conda deactivate
   conda remove --name md4dsp --all -y
   ```
   
## Contract tests execution

To run the contract tests, follow the next steps:


1. Run the contract tests:
   ```bash
    python ./test_contracts.py
    ```

2. Check the results in the logs:

Once the tests have finished, one log will be created for each execution of the python script. The test logs are located in the `logs/test` directory. By default, the logs are named as follows: `testLog_<number>.log`.

## Data transformation tests execution

To run the data transformation tests, follow the next steps:

1. Run the data transformation tests:
   ```bash
    python ./test_data_transformations.py
    ```

2. Check the results in the logs:

Once the tests have finished, one log will be created for each execution of the python script. The test logs are located in the `logs/test` directory. By default, the logs are named as follows: `testLog_<number>.log`.

# Generated contract calls via Acceleo (Code Generation)

The contract calls are generated in a Python script via Acceleo. Once the generated is generated, it must be moved to the `generated_code` directory. The script to test the code generation is named `dataProcessing.py`. You'll just need to run the script to call the contracts. The dataset used to test this generated script must be named 'data_model.csv' and must be located in a parent directory to the generated script.
The generated files can be executed by running one of the following commands:

1. Execute just the data transformations script:
   ```bash
    python -m generated_code.data_transformations
    ```

2. Execute just the contracts script:
   ```bash
    python -m generated_code.contracts
    ```

3. Execute both, the data transformations and the contracts by running the following command:
   ```bash
    python -m generated_code.dataProcessing
    ```

## Project Structure

The project structure must follow the next structure:

```bash
MD4DSP-m2python/
│
├── functions/
│ ├── contract_invariants.py
│ ├── contract_pre_post.py
│ └── data_transformations.py
│
├── generated_code/
│ └── dataProcessing.py
│
├── helpers/
│ ├── auxiliar.py
│ ├── enumerations.py
│ ├── invariant_aux.py
│ ├── logger.py
│ └── transform_aux.py
│
├── logs/
│ └── test/
│   ├── ...
│   └── testLog_<number>.log
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
├── data_model.csv
├── README.md
├── requirements.txt
├── test_contracts.py
└── test_data_transformations.py

```

- **`functions/`**: contains the main functions of the project. The functions are divided into three files: `contract_invariants.py`, `contract_pre_post.py` and `data_transformations.py`. The first file contains the functions of the invariants, the second file contains the functions of the contracts, and the third file contains the functions of the data transformations.


- **`generated_code/`**: contains the generated code via Acceleo. The generated code must be located in this directory.


- **`helpers/`**: contains auxiliary functions that are used in the main functions. The file `auxiliar.py` contains the auxiliary functions, `enumerations.py` contains the enumerations used in the project, `invariant_aux.py` contains the auxiliary functions of the invariants, `logger.py` contains the logger functions, and `transform_aux.py` contains the auxiliary functions of the data transformations.


- **`logs/`**: contains the logs of the tests. The logs are stored in the directory `test`.


- **`test_datasets/`**: contains the external datasets used in the tests. The datasets are divided into directories, and each directory contains the dataset and a readme file with the description of the dataset.


- **`test/`**: contains the tests to make exhaustive evaluations of the functions. The tests are divided into two directories: `contract_pre_post`, `contract_invariants` and `data_transformations`. Each directory contains the tests of the functions of the contracts, the invariants, and the data transformations, respectively.


- **`data_model.csv`**: dataset used to test the contracts code calls generated via Acceleo.


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
If you have any questions, please contact to any of the authors.
