// Copyright (c) 2023 Victor I. Afolabi
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
use iris::{Data, Dataset};
use ndarray::prelude::*;

mod utils;

#[test]
fn test_features_data() {
  let df = utils::get_features_df();
  let features = Data::try_from(&df).unwrap();
  assert_eq!(
    features.names,
    &["sepal_length", "sepal_width", "petal_length", "petal_width"]
  );
  assert_eq!(features.data.shape(), [150, 4]);
}

#[test]
fn test_target_data() {
  let df = utils::get_target_df();
  let target = Data::try_from(&df).unwrap();
  assert_eq!(target.names, &["species"]);
  assert_eq!(target.data.shape(), [150, 1]);
}

#[test]
fn test_data() {
  let df = utils::get_target_df();
  let target = Data::try_from(&df);
  assert!(target.is_ok());
}

#[test]
fn test_data_new() {
  let d = Data::new()
    .with_names(&["a", "b", "c"])
    .with_data(arr2(&[[1., 2., 3.], [4., 5., 6.]]));

  assert!(d.names == &["a", "b", "c"]);
  assert!(d.data == arr2(&[[1., 2., 3.], [4., 5., 6.]]));
}

#[test]
fn test_dataset() {
  let df = utils::load_df();
  let ds = Dataset::try_from(&df).unwrap();

  assert_eq!(
    ds.features().names(),
    &["sepal_length", "sepal_width", "petal_length", "petal_width"]
  );
  assert_eq!(ds.features().data().shape(), [150, 4]);
  assert_eq!(ds.target().names(), &["species"]);
  assert_eq!(ds.target().data().shape(), [150, 1]);
}

#[test]
fn test_dataset_new() {
  let ds = Dataset::new()
    .with_features(
      Data::new()
        .with_names(&["a", "b", "c"])
        .with_data(arr2(&[[1., 2., 3.], [4., 5., 6.]])),
    )
    .with_target(Data::new().with_names(&["d"]));

  assert_eq!(ds.features().names(), &["a", "b", "c"]);
  assert_eq!(ds.target().names(), &["d"]);
  assert!(ds.target().data().is_empty());
}
