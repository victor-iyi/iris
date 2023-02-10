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
  let model = iris::model::train(&data, &feature_names, &v)?;
  // dbg!(&model);
  println!("\nModel characteristics:");
  dbg!(&model.max_depth());
  dbg!(&model.num_leaves());
  dbg!(&model.feature_importance());
  dbg!(&model.relative_impurity_decrease());

  // Save the trained model to a file.
  File::create("images/iris.tex")?
    .write_all(model.export_to_tikz().with_legend().to_string().as_bytes())?;

  Ok(())
}
