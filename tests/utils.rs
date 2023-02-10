// Copyright (c) 2023 Victor I. Afolabi
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use linfa_trees::DecisionTree;
use polars::prelude::*;
use std::path::Path;

// Load data into `DataFrame`.
#[allow(dead_code)]
pub fn load_df() -> DataFrame {
  let df = iris::load_data(Some(Path::new("data/iris.csv")));
  assert!(df.is_ok());
  df.unwrap()
}

#[allow(dead_code)]
pub fn get_features_df() -> DataFrame {
  let df = load_df();
  let column_names = &df.get_column_names();

  let features = df.select(&column_names[0..4]);
  assert!(features.is_ok());
  features.unwrap()
}

#[allow(dead_code)]
pub fn get_target_df() -> DataFrame {
  let df = load_df();
  let column_names = &df.get_column_names();

  let target = df.select(&[column_names[4]]);
  assert!(target.is_ok());
  target.unwrap()
}

/// Process dataframe and retrun trained DecisionTree model.
#[allow(dead_code)]
pub fn get_model<'a>(df: &'a DataFrame) -> DecisionTree<f64, &'a str> {
  // Process dataframe to ndarray.
  let processed = iris::process::pre_process(&df);
  assert!(processed.is_ok());
  let (data, feature_names, target_values) = processed.unwrap();

  // train model.
  let model = iris::model::train(&data, &feature_names, &target_values);
  assert!(model.is_ok());
  model.unwrap()
}
