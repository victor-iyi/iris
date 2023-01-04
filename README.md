<!--
 Copyright (c) 2022 Victor I. Afolabi

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Iris

Experiment to test loading and processing iris dataset in Rust and using the
processed data in Python for Machine Learning tasks.

## Method

Load and process the  iris dataset with `polars`, a fast, Rust library for
working with `DataFrame`s.

DataFrame is split into features and labels, and labels are converted into
categorical values. The result will be two dataframe *(one for features & one
for label)*.

These `DataFrame`s are then converted into an `ndarray`, which will be attempted
to send to a Python process or an API or just saved to disk in a way that can be
read natively by Python.

## Examples

There's an [`examples/`] folder containing both Rust & Python notebooks.

[`examples/`]: ./examples/

## Contribution

You are very welcome to modify and use them in your own projects.

Please keep a link to the [original repository]. If you have made a fork with
substantial modifications that you feel may be useful, then please [open a new
issue on GitHub][issues] with a link and short description.

## License (MIT)

This project is opened under the [MIT][license] which allows very
broad use for both private and commercial purposes.

A few of the images used for demonstration purposes may be under copyright.
These images are included under the "fair usage" laws.

[original repository]: https://github.com/victor-iyi/iris
[issues]: https://github.com/victor-iyi/iris/issues
[license]: ./LICENSE
