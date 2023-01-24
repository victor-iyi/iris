// Copyright (c) 2023 Victor I. Afolabi
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
use polars::prelude::*;

use iris::Data;

mod data;

fn get_features_df() -> DataFrame {
  let df = data::load_df();
  let column_names = &df.get_column_names();
  df.select(&column_names[0..4]).unwrap()
}

fn get_target_df() -> DataFrame {
  let df = data::load_df();
  let column_names = &df.get_column_names();
  df.select(&[column_names[4]]).unwrap()
}

#[test]
fn test_features_data() {
  let df = get_features_df();
  let features = Data::new(&df);
  assert_eq!(
    features.names,
    &["sepal_length", "sepal_width", "petal_length", "petal_width"]
  );
  assert_eq!(features.data.shape(), [150, 4]);
}

#[test]
fn test_target_data() {
  let df = get_target_df();
  let target = Data::new(&df);
  assert_eq!(target.names, &["species"]);
  assert_eq!(target.data.shape(), [150, 1]);
}
