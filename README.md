# MD4DSP-m2python

## Prerequisites

- Anaconda Environment
- Python 3.11
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
│ ├── 
│ └── ...
│
├── .gitignore
├── contracts_main.py
├── README.md
└── requirements.txt

```

- **`functions/`**: contains the main functions of the project.

- **`helpers/`**: contains auxiliary functions that are used in the main functions.

- **`test/`**: contains the tests to make exhaustive evaluations of the functions.

- **`requirements.txt`**: file that contains the libraries needed to run the project.

  ```bash
  conda create --name tu_entorno --file requirements.txt
    ```
  
## Authors
- Carlos Breuer Carrasco
- Carlos Cambero Rojas

## Questions
If you have any questions, please contact to any of the authors.