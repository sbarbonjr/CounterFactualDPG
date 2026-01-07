# CounterFactualDPG

For Linux/Mac Users:
  ```bash
  # Create a virtual environment
  python -m venv .venv

  # Activate the virtual environment
  source .venv/bin/activate

  # Install DPG
  pip install -r ./requirements.txt
  ```
nbstripout is recommended to keep notebooks clean of output cells. This one is better installed globally, then enabled as a hook in the repository:

  ```bash
  pip install nbstripout
  nbstripout --install
```