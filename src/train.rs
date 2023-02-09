use anyhow::Result;
use linfa::prelude::*;
use linfa_trees::*;
use ndarray::{prelude::*, OwnedRepr};
use rand::thread_rng;

/// Train a Decision Tree model on the Iris dataset.
///
/// ## Arguments
///
/// - `data`: A reference to a 2D array where the first dimension is of shape
/// `[N, F]` where `F` is the number of features and `N` is the number of
/// samples and the second dimension is of shape `[N, 1]`.
///
/// - `feature_names`: List of feature names of the iris dataset.
///
/// - `target_values`: List of target values of the iris dataset.
///
/// See [`pre_process`](fn@pre_process) for more info.
///
pub fn train<'a>(
  data: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
  feature_names: &[&str],
  target_values: &[&'a str],
) -> Result<DecisionTree<f64, &'a str>> {
  let num_features = feature_names.len();
  // Create features.
  let features = data
    .slice(s![.., ..num_features])
    .as_standard_layout()
    .to_owned();

  // Create targets.
  let target = data.column(num_features).map(|x| *x as usize);

  // Shuffle dataset.
  let mut rng = thread_rng();
  let dataset = Dataset::new(features, target)
    .with_feature_names(feature_names.to_owned())
    .map_targets(|t| target_values[*t])
    .shuffle(&mut rng);

  // dbg!(&dataset);

  // Split into train & validation set.
  let (train, valid) = dataset.split_with_ratio(0.9);

  let tree = DecisionTree::params()
    .split_quality(SplitQuality::Gini)
    .fit(&train)?;

  let pred = tree.predict(&valid);
  println!("Ground truth: {:?}", valid.targets().to_vec());
  println!("Predicted: {:?}", pred.to_vec());

  let cm = pred.confusion_matrix(&valid)?;
  println!("{cm:?}");

  // Accuracy
  println!("Accuracy: {}", cm.accuracy());
  println!("Precision: {} | Recall: {}", cm.precision(), cm.recall());
  println!("Correlation coefficient: {}", cm.mcc());

  Ok(tree)
}
