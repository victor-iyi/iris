use anyhow::Result;

use std::{fs::File, io::Write, path::Path};

fn main() -> Result<()> {
  // Load or download data into a DataFrame object.
  let path = Path::new("data/iris.csv");
  let df = iris::data::load_data(Some(path))?;

  // Process dataframe.
  let (data, feature_names, target_values) = iris::process::pre_process(&df)?;

  // Train data on DecisionTree model.
  let v: Vec<&str> = target_values.iter().map(|s| s.as_ref()).collect();
  let model = iris::train::train(&data, &feature_names, &v)?;
  // dbg!(&model);

  // Save the trained model to a file.
  File::create("images/iris.tex")?
    .write_all(model.export_to_tikz().with_legend().to_string().as_bytes())?;

  Ok(())
}
