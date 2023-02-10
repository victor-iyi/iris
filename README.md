<!--
 Copyright (c) 2023 Victor I. Afolabi

 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# Iris

[![CI](https://github.com/victor-iyi/iris/actions/workflows/ci.yml/badge.svg)](https://github.com/victor-iyi/iris/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/victor-iyi/iris/main.svg)](https://results.pre-commit.ci/latest/github/victor-iyi/iris/main)

Experiment to test loading and processing iris dataset in Rust and using the
processed data in Python for Machine Learning tasks.

## Method

Load and process the  iris dataset with `polars`, a fast, Rust library for
working with `DataFrame`s.

`DataFrame` is split into features and labels, and labels are converted into
categorical values. The result will be two dataframe *(one for features & one
for label)*.

A `Dataset` object is created with `linfa` and then split into training and
validation set. The training set is trained on a `linfa_trees::DecisionTree` model,
and predictions are made on the validation dataset to compute the confusion matrix,
accuracy, precision, recall and other metrices.

```sh
classes    | setosa     | versicolor | virginica
setosa     | 3          | 0          | 0
versicolor | 0          | 4          | 0
virginica  | 0          | 1          | 7

Metrics:
[src/model.rs:75] &cm.accuracy() = 0.93333334
[src/model.rs:75] &cm.precision() = 0.93333334
[src/model.rs:75] &cm.recall() = 0.9583333
[src/model.rs:75] &cm.mcc() = 0.8994902
```

> In practice, large dataframe can be loaded and processed with Rust's `polars`
> crate and then saved to disk or a data warehouse which can be cheaply used by
> python to train or perform other data science tasks.

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
