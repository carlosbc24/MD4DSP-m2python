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
   
3. Install the required dependencies using Anaconda:

   ```bash
   conda create --name your-env-name python=3.11
   conda activate your-env-name
   pip install -r requirements.txt
   ```

4. Run the tests:

   ```bash
    python ./contracts_main.py
    ```

5. Check the results in the logs:

Once the tests have finished, one log will be created for each execution of the python script. The test logs are located in the `logs/test` directory. By default, the logs are named as follows: `testLog_<number>.log`.

6. (Optional) Remove the environment created previously:

   ```bash
   conda deactivate
   conda remove --name your-env-name --all
   ```

## Project Structure

The project structure must follow the next structure:

```bash
MD4DSP-m2python/
│
├── functions/
│ ├── contract_invariants.py
│ └── contract_pre_post.py
│
├── helpers/
│ ├── auxiliar.py
│ ├── enumerations.py
│ └── logger.py
│
├── tests/
│ ├── contract_pre_post/
│ │ ├── simple_test.py
│ │ └── tests_spotify_dataset.py
│ │
│ └── invariants/
│   ├── simple_test.py
│   └── tests_spotify_dataset
│
├── test_datasets/
│ ├── spotify_songs/
│   ├── spotify_songs.csv
│   └── readme.md
│
├── .gitignore
├── contracts_main.py
├── README.md
└── requirements.txt

```

- **`functions/`**: contains the main functions of the project. The functions are divided into two files: `contract_invariants.py` and `contract_pre_post.py`. The first file contains the functions of the invariants, and the second file contains the functions of the contracts.

- **`helpers/`**: contains auxiliary functions that are used in the main functions. The file `auxiliar.py` contains the auxiliary functions, `enumerations.py` contains the enumerations used in the project, and `logger.py` contains the functions to log the results of the tests.

- **`test/`**: contains the tests to make exhaustive evaluations of the functions. The tests are divided into two directories: `contract_pre_post` and `invariants`. The first directory contains the tests of the contracts, and the second directory contains the tests of the invariants. Each package contains simple tests and tests with the Spotify dataset.

- **`test_datasets/`**: contains the external datasets used in the tests. The datasets are divided into directories, and each directory contains the dataset and a readme file with the description of the dataset.

- **`requirements.txt`**: file that contains the libraries needed to run the project.

- **`contracts_main.py`**: main file of the project. It is the file that must be executed to run the tests.

- **`README.md`**: file that contains the documentation of the project.
  
## External Documentation
The external documentation of the project is available in the following link: https://unexes.sharepoint.com/:w:/s/PDI_i3lab/EYNMm7pMsX1HuIKz_PMWCi8Bl_ssrzRnvp3hQHimY363ng?e=d8Cvvh
  
## Authors
- Carlos Breuer Carrasco
- Carlos Cambero Rojas

## Questions
If you have any questions, please contact to any of the authors.