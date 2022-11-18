import os

import hera
from hera import DAG, Parameter, Task, Workflow

from tiny_torch.evaluate import evaluate
from tiny_torch.main import main
from tiny_torch.visualize import view_examples

hera.set_global_token = os.environ['ARGO_TOKEN']
hera.set_global_host = os.environ['ARGO_HOST']

with Workflow('pipeine') as wf:
    Task(
        'train', main
    )  # >> [Task('evaluate', evaluate), Task('visualize', view_examples)]

wf.create()
