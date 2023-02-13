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
/// # use iris::{data::load_data, process::pre_process};
/// #
/// # let df = load_data(Some(Path::new("data/iris.csv")))?;
/// let (data, feature_names, target_values) = pre_process(&df)?;
///
/// assert_eq!(data.shape(), [150, 5]);
/// assert_eq!(feature_names, ["sepal_length", "sepal_width", "petal_length", "petal_width"]);
/// assert_eq!(target_values, ["setosa", "versicolor", "virginica"]);
///
/// # Ok(())
/// # }
/// ```
pub fn pre_process<'a>(
  df: &'a DataFrame,
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

  // Get list of unique species.
  // let target_name = &*column_names[num_features];
  // let species = df.column(target_name)?.unique()?;
  // let target_values: Vec<String> = (0..species.len())
  //   .map(|i| species.get(i).unwrap().to_string().replace("\"", ""))
  //   .collect();

  // Get the list of target values.
  let target_values = ["setosa", "versicolor", "virginica"];
  // dbg!(&target_values);
  // let species_arr = df.column("species")?.utf8()?.to_owned();
  // let species_vec: Vec<String> = Vec::from(&species_arr)
  //   .iter()
  //   .map(|s| s.unwrap_or_default().to_string())
  //   .collect();
  // dbg!(&species_vec);

  // Convert dataframe into ndarray.
  let data = df.to_ndarray::<Float64Type>()?;

  Ok((data, feature_names.to_owned(), target_values.to_vec()))
}
