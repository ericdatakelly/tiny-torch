import os

import hera
from dotenv import load_dotenv
from hera import DAG, Parameter, Task, Workflow

from tiny_torch.evaluate import evaluate
from tiny_torch.main import main
from tiny_torch.visualize import view_examples

load_dotenv()
hera.set_global_token = os.getenv('ARGO_TOKEN')
hera.set_global_host = os.getenv('ARGO_SERVER')

with Workflow('pipeine') as wf:
    Task(
        'train', main
    )  # >> [Task('evaluate', evaluate), Task('visualize', view_examples)]

wf.create()
