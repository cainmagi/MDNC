# MDNC Documentation

## Configurations

This branch is used for storing the source files of the documents. It requires a submodule [mkdocs-material][git-mkmaterial]. To initialize the theme, use

```bash
git submodule update --init --recursive
```

To update the theme, use

```bash
git submodule update --remote --recursive
```

## Run

Use the following command to run the debug mode. The website would be served on http://localhost:8000/.

```bash
mkdocs serve
```

Use the following command to build the website. The website could be uploaded in git-page branch.

```bash
mkdocs build
```

[git-mkmaterial]:https://github.com/squidfunk/mkdocs-material "MkDocs Material"
