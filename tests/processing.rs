// Copyright (c) 2023 Victor I. Afolabi
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
use iris::Data;

mod utils;

#[test]
fn test_features_data() {
  let df = utils::get_features_df();
  let features = Data::new(&df);
  assert_eq!(
    features.names,
    &["sepal_length", "sepal_width", "petal_length", "petal_width"]
  );
  assert_eq!(features.data.shape(), [150, 4]);
}

#[test]
fn test_target_data() {
  let df = utils::get_target_df();
  let target = Data::new(&df);
  assert_eq!(target.names, &["species"]);
  assert_eq!(target.data.shape(), [150, 1]);
}

#[test]
fn test_data() {
  let df = utils::get_target_df();
  let target = Data::try_from(&df);
  assert!(target.is_ok());
}
