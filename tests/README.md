Run pytests with the following command otherwise you get an package name not found error:

```sh
python -m pytest tests/unit
```

To check for code coverage using the pytest-cov plugin (if installed):

```sh
python -m pytest --cov=curve_curator tests/unit
```
