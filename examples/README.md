<!--
 Copyright (c) 2023 Victor I. Afolabi

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Jupyter Kernel for Rust Programming Language

[evcxr] is an evaluation context for Rust. It is an unofficial google project
that offers several related crates, one of which is [evcxr_jupyter] which is a
Juypter Kernel for the Rust programming language.

BTW, if you're wondering how to pronounce `EvCxR`; it's pronounced *"Evic-ser"*
cos it's an **EV**aluation Conte**X**t for **R**ust.

You can take a [tour of the Juypter Kernel][tour] to get a feel for using Rust
in Jupyter Notebooks.

## Setup

You can setup `evcxr_jupyter` by running the following commands:

```sh
cargo install evcxr_jupyter
evcxr_jupyter --install
```

Check full installation instructions [here][install].

## Plotly Jupyter Support

To show plots in Jupyter notebook, install the jupyterlab extension by executing
the following command:

```sh
jupyter labextension install jupyterlab-plotly
```

Visit the [Jupyter Support] for plotly guide for more installation help.

[evcxr]: https://github.com/google/evcxr
[evcxr_jupyter]: https://github.com/google/evcxr/blob/main/evcxr_jupyter/README.md
[install]: https://github.com/google/evcxr/blob/main/evcxr_jupyter/README.md#installation
[tour]: https://github.com/google/evcxr/blob/main/evcxr_jupyter/samples/evcxr_jupyter_tour.ipynb
[Jupyter Support]: https://igiagkiozis.github.io/plotly/content/fundamentals/jupyter_support.html
