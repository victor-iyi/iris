use iris::train::pre_process;
mod utils;

#[test]
fn test_pre_process() {
  let df = utils::load_df();
  let processed = pre_process(&df);
  assert!(processed.is_ok());

  let (data, feature_names, target_values) = processed.unwrap();

  assert_eq!(data.shape(), [150, 5]);
  assert_eq!(
    feature_names,
    ["sepal_length", "sepal_width", "petal_length", "petal_width"]
  );
  assert_eq!(target_values, ["setosa", "versicolor", "verginica"]);
}
