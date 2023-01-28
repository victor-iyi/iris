use ndarray::prelude::*;
use polars::prelude::*;

pub mod data;

#[derive(Clone, Default)]
pub struct Dataset {
  features: Data,
  target: Data,
}

impl Dataset {
  /// Create a new Dataset from a dataframe.
  pub fn new() -> Self {
    Self::default()
  }

  pub fn features(&self) -> &Data {
    &self.features
  }

  pub fn target(&self) -> &Data {
    &self.target
  }
}

impl Dataset {
  pub fn with_features(mut self, features: Data) -> Self {
    self.features = features;
    self
  }

  pub fn with_target(mut self, target: Data) -> Self {
    self.target = target;
    self
  }
}

impl TryFrom<DataFrame> for Dataset {
  type Error = PolarsError;
  fn try_from(df: DataFrame) -> Result<Self, Self::Error> {
    Self::try_from(&df)
  }
}

impl TryFrom<&DataFrame> for Dataset {
  type Error = PolarsError;
  fn try_from(df: &DataFrame) -> Result<Self, Self::Error> {
    // Shuffle dataframe.
    let df = df.sample_frac(1., false, true, None)?;
    let column_names = df.get_column_names();

    // Create features & target.
    let features = Data::try_from(&df.select(&column_names[0..4])?)?;
    let target = Data::try_from(&df.select(&[&column_names[4]])?)?;

    Ok(Self { features, target })
  }
}

/// Abstraction for spliting dataframe into ndarray.
#[derive(Clone, Default)]
pub struct Data {
  /// Column names.
  pub names: Vec<String>,
  /// Data as ndarray.
  pub data: Array2<f64>,
}

impl Data {
  /// Create an empty Data.
  pub fn new() -> Self {
    Self::default()
  }

  /// Return a reference to the column names.
  pub fn names(&self) -> &[String] {
    &self.names
  }

  /// Return a reference to the data.
  pub fn data(&self) -> &Array2<f64> {
    &self.data
  }
}

impl Data {
  pub fn with_names(mut self, names: &[&str]) -> Self {
    self.names = names.iter().map(|s| s.to_string()).collect();
    self
  }

  pub fn with_data(mut self, data: Array2<f64>) -> Self {
    self.data = data;
    self
  }
}

impl TryFrom<DataFrame> for Data {
  type Error = PolarsError;
  fn try_from(df: DataFrame) -> Result<Self, Self::Error> {
    Self::try_from(&df)
  }
}

impl TryFrom<&DataFrame> for Data {
  type Error = PolarsError;
  fn try_from(df: &DataFrame) -> Result<Self, Self::Error> {
    let names = df.get_column_names_owned();
    let data = df.to_ndarray::<Float64Type>()?;
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
