How to Contribute
=================

## Contributing License Agreement
By contributing to this repo, you agree to become a [Codalab framework contributor](https://github.com/codalab/codalab-competitions/blob/master/docs/Community-Governance.md).

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
For this project, we follow the *fork and pull model* of collaborative development. 
This means that collaborators have to suggest code changes by making pull requests from their own fork. These changes can then be pulled into the source repository by the project maintainer.

For collaborators, the steps to follow are:
1. **Create a fork**, i.e. a copy of the source repository under your own GitHub account. To do this, go to the [project main page](https://github.com/zhengying-liu/autodl), sign in and click on the *Fork* button on right top of the page;
2. **Clone your Fork**. The standard clone command creates a local git repository from your remote fork on GitHub.
    ```
    git clone https://github.com/USERNAME/autodl.git
    ```
    **WARNING:** Remember to change `USERNAME` to your own username.
3. **Modify the Code**. In your local clone, modify the code and commit them to your local clone using the `git commit` command.
4. **Push your Changes**. In your workspace, use the `git push` command to upload your changes to your remote fork on GitHub.
5. **Create a Pull Request**. On the GitHub page of your remote fork, click the “Pull request” button. Wait for the owner to merge or comment your changes and be proud when it is merged :). If the owner suggests some changes before merging, you can simply push these changes into your fork by repeating steps #3 and #4 and the pull request is updated automatically.

<!--
Some material is borrowed from https://reflectoring.io/github-fork-and-pull/)
-->

## Tips
1. For changes of Jupyter notebooks, please do test by *Cell -> Run All* then save the notebook after doing *Kernel -> Restart & Clear Output*. This is to avoid the useless cell numbering, metadata and images changes.
2. To update your repo and synchronize it with the original repo (`zhengying-liu/autodl`):
    ```
    git pull upstream master
    ```

## Further reading
- [About pull requests](https://help.github.com/en/articles/about-pull-requests)
- [Creating a pull request from a fork](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork)
- [Allowing changes to a pull request branch created from a fork](https://help.github.com/en/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork)
