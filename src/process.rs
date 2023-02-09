use anyhow::Result;
use ndarray::{prelude::*, OwnedRepr};
use polars::prelude::*;

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