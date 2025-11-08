# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import sys

home = os.path.expanduser('~')  
project_path = os.path.join(home, 'Designing complex biological circuits with deep neural networks')

if project_path not in sys.path:
    sys.path.append(project_path)
