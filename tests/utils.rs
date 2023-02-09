// Copyright (c) 2023 Victor I. Afolabi
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use iris::data::load_data;
use polars::prelude::*;
use std::path::Path;

// Load data into `DataFrame`.
#[allow(dead_code)]
pub fn load_df() -> DataFrame {
  let df = load_data(Some(Path::new("data/iris.csv")));
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
