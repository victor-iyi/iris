use anyhow::Result;
use linfa::prelude::*;
use linfa_trees::*;
use ndarray::{prelude::*, OwnedRepr};
use polars::prelude::*;
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

/// Pre-process dataframe into an ndarray.
///
/// ## Returns
///
/// - `data`: A 2D array of the dataframe including the features & target.
///
/// - `feature_names`: Column names of all features.
///
/// - `target_values`: Values to be predicted by the model.
///   Target values are: `setosa`, `versicolor` & `verginica`.
///
/// ## Example
///
/// ```rust
/// # use anyhow::Result;
/// # fn main() -> Result<()> {
/// # use std::path::Path;
/// # use iris::{data::load_data, train::pre_process};
/// #
/// # let df = load_data(Some(Path::new("data/iris.csv")))?;
/// let (data, feature_names, target_values) = pre_process(&df)?;
///
/// assert_eq!(data.shape(), [150, 5]);
/// assert_eq!(feature_names, ["sepal_length", "sepal_width", "petal_length", "petal_width"]);
/// assert_eq!(target_values, ["setosa", "versicolor", "verginica"]);
///
/// # Ok(())
/// # }
/// ```
pub fn pre_process(
  df: &DataFrame,
) -> Result<(
  ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
  Vec<&str>,
  Vec<&str>,
)> {
  // Get list of column names.
  let column_names = df.get_column_names();
  let num_features = column_names.len() - 1;

  // Get the list of feature names.
  let feature_names = &column_names[..num_features];
  let target_name = &*column_names[num_features];
  // Get the list of target values.
  let target_values = ["setosa", "versicolor", "verginica"];

  let unique = df
    .select(&[target_name])?
    .unique(Some(&[target_name.to_owned()]), UniqueKeepStrategy::First)?;
  dbg!(&unique);

  // Convert dataframe into ndarray.
  let data = df.to_ndarray::<Float64Type>()?;

  Ok((data, feature_names.to_owned(), target_values.to_vec()))
}
