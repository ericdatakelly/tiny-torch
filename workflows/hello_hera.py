"""This example showcases the hello world example of Hera"""
import os

import hera
from dotenv import load_dotenv
from hera import Task, Workflow

load_dotenv()
hera.set_global_token(os.getenv('ARGO_TOKEN'))
hera.set_global_host(os.getenv('ARGO_SERVER'))


def hello():
    print("Hello, Hera!")


# assumes you used `hera.set_global_token` and `hera.set_global_host` so that the workflow can be submitted
with Workflow("hello-hera") as w:
    Task("t", hello)

w.create()
