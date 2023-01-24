use ndarray::prelude::*;
use polars::prelude::*;

pub mod data;

pub struct Data {
  pub names: Vec<String>,
  pub data: Array2<f64>,
}

impl Data {
  pub fn new(df: &DataFrame) -> Self {
    let names = df.get_column_names_owned();
    let data = df.to_ndarray::<Float64Type>().unwrap();
    Self { names, data }
  }
}

impl From<DataFrame> for Data {
  fn from(df: DataFrame) -> Self {
    let names = df.get_column_names_owned();
    let data = df.to_ndarray::<Float64Type>().unwrap();
    Self { names, data }
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
