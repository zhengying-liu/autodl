How to Contribute
=================

## Contributing License Agreement
TODO (@madclam)

## Python coding style

All Python code in our project should follow
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md) (which is very similar to common [PEP 8](https://www.python.org/dev/peps/pep-0008/) Python style guide).

Use `pylint` to check your Python changes. To install `pylint` and
retrieve TensorFlow's custom style definition:

```bash
pip install pylint
wget -O /tmp/pylintrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc
```

To check a file with `pylint`:

```bash
pylint --rcfile=/tmp/pylintrc myfile.py
```

## How to contribute your code using pull request
For this project, we follow the *fork and pull model* collaborative development. This means that collaborators have to suggest code changes by making pull requests from their own fork. These changes can then be pulled into the source repository by the project maintainer.

For collaborators, the steps to follow are:
1. **Create a fork**, i.e. a copy of the source repository under your own GitHub account. To do this, go to the [project main page](https://github.com/zhengying-liu/autodl), sign in and click on the *Fork* button on right top of the page;
2. **Make changes to your own repo**. Do the changes that you would like to suggest in your own repo.
3. **Make a pull request**. TODO(zhengying-liu)
Then the project maintainer will review your code and integrate it into the 
source repo if it goes well.

## Tips
1. For changes of Jupyter notebooks, please do test by *Cell -> Run All* then save the notebook after doing *Kernel -> Restart & Clear Output*. This is to avoid the useless cell numbering, metadata and images changes.

## Further reading
- [About pull requests](https://help.github.com/en/articles/about-pull-requests)
- [Creating a pull request from a fork](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork)
- [Allowing changes to a pull request branch created from a fork](https://help.github.com/en/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork)
