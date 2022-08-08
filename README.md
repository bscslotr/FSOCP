If you want to use `FS` package inside `archive/pso_feature_selection.py`, you can get
the `FS` package from https://github.com/JingweiToo/Wrapper-Feature-Selection-Toolbox-Python

---

To prepare your development environment for `socp`:

```sh
cd socp
python3.8 -m venv .venv                  # Create virtual env
source .venv/bin/activate                # Activate virtual env
pip install -r requirements_py38.txt     # For Python3.8: Install dependencies for working with socp
pip install -r requirements_py37.txt     # For Python3.7: Install dependencies for working with socp
```

---

To run your `socp` scripts:

```sh
cd socp
source .venv/bin/activate            # Do this, if it is not already activated
python socp.py                       # Run your script
deactivate                           # Do this, if you want to deactive virtual env
```

---
  




  




