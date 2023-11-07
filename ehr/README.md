This code implements the EHR component of the INSPECT code and benchmark.

It is recommended to run Python 3.10 with this code as it has only been tested with Python 3.10

In order to use:

- pip install -r requirements.txt
- Run run_all_ehr.py with the path to the INSPECT data release, a path to store output, a path to a download from Athena, and a path to a pretrained MOTOR model.

Athena is an OHDSI service for downloading ontologies. Simply visit https://athena.ohdsi.org, create an account, and click download at the top.

MOTOR will be obtainable from  https://huggingface.co/StanfordAIMI (release still in progress).

python run_all_ehr.py --path_to_data PATH_TO_DATASET --path_to_output PATH_TO_OUTPUT --path_to_athena PATH_TO_ATHENA_DOWNLOAD --path_to_motor PATH_TO_MOTOR

The output of this command can be analyzed with get_model_performance.py in the parent folder.
