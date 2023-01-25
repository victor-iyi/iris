use ndarray::prelude::*;
use polars::prelude::*;

pub mod data;

/// Abstraction for spliting dataframe into ndarray.
pub struct Data {
  /// Column names.
  pub names: Vec<String>,
  /// Data as ndarray.
  pub data: Array2<f64>,
}

impl Data {
  /// Create a new Data split from a dataframe.
  ///
  /// Converts the dataframe into an ndarray.
  pub fn new(df: &DataFrame) -> Self {
    let names = df.get_column_names_owned();
    let data = df.to_ndarray::<Float64Type>().unwrap();
    Self { names, data }
  }
}

impl TryFrom<DataFrame> for Data {
  type Error = PolarsError;
  fn try_from(value: DataFrame) -> Result<Self, Self::Error> {
    Self::try_from(&value)
  }
}

impl TryFrom<&DataFrame> for Data {
  type Error = PolarsError;
  fn try_from(value: &DataFrame) -> Result<Self, Self::Error> {
    let names = value.get_column_names_owned();
    let data = value.to_ndarray::<Float64Type>()?;
    Ok(Self { names, data })
  }
}

impl std::fmt::Debug for Data {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Data")
      .field("names", &self.names)
      .field("data", &self.data)
      .finish()
  }
}
