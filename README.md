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
│ ├── contract_pre_post.py
│ └── ...
│
├── helpers/
│ ├── auxiliar.py
│ ├── enumerations.py
│ ├── logger.py
│ └── ...
│
├── test/
│ ├── simple_test.py
│ ├── tests_spotify_dataset.py
│ └── ...
│
├── test_datasets/
│ ├── spotify_songs
│   ├── readme.md
│   └── spotify_songs.csv
│
├── .gitignore
├── contracts_main.py
├── README.md
└── requirements.txt

```

- **`functions/`**: contains the main functions of the project.

- **`helpers/`**: contains auxiliary functions that are used in the main functions.

- **`test/`**: contains the tests to make exhaustive evaluations of the functions.

- **`test_datasets/`**: contains the datasets used in the tests. We have decided to upload it to the repository to facilitate the execution of the tests, using the same version that the developers used to execute the tests.

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