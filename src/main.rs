use eyre::Result;

use ndarray::prelude::*;
use polars::prelude::*;

use linfa::prelude::*;
use linfa_trees::{DecisionTree, SplitQuality};

use std::{fs::File, io::Write, path::Path};

use iris::{data, Data};

///  Train a DecisionTree model on Iris data.
fn train() -> Result<()> {
  // Load dataframe.
  let df = data::load_data(Some(Path::new("data/iris.csv")))?
    .collect()?
    .sample_frac(1., false, true, Some(42))?;
  println!("{:?}", df);

  // Names of columns in dataframe.
  let colum_names = df.get_column_names();
  let num_features = colum_names.len() - 1;

  // Feature & target names.
  let feature_names = &colum_names[0..num_features];
  let target_name = &colum_names[num_features];

  println!("\nFeatures({}): {:?}", num_features, feature_names);
  println!("Target: {}", target_name);

  // Create ndarray data from dataframe.
  let data = df.to_ndarray::<Float64Type>()?;

  // Split data into features & target.
  let features = data
    .slice(s![.., ..num_features])
    .as_standard_layout()
    .to_owned();
  let target = data.column(num_features).map(|x| x.to_owned() as usize);

  println!("Data: {:?}", data.shape());
  println!("Features: {:?}", features.shape());
  println!("Target: {:?}", target.shape());

  // Create dataset from features & target.
  let dataset = Dataset::new(features, target)
    .with_feature_names(feature_names.to_owned())
    .map_targets(|t| match t {
      0 => "setosa",
      1 => "versicolor",
      2 => "virginica",
      _ => unreachable!(),
    });
  dbg!(&dataset);

  // Split into train & validation set.
  let (train, valid) = dataset.split_with_ratio(0.9);

  // Train the model.
  let tree = DecisionTree::params()
    .split_quality(SplitQuality::Gini)
    .fit(&train)?;
  dbg!(&tree);

  // Make prediction.
  let pred = tree.predict(&valid);
  println!("Ground truth: {:?}", valid.targets().to_vec());
  println!("Predicted: {:?}", pred.to_vec());

  // Confusion matrix.
  let cm = pred.confusion_matrix(&valid)?;
  println!("{:?}", cm);

  // Accuracy.
  println!("Acc: {}", cm.accuracy());
  println!("Precision: {} | Recall: {}", cm.precision(), cm.recall());
  println!("Correlation coeffient: {}", cm.mcc());

  // Save the trained model to a file.
  File::create("images/iris.tex")?
    .write_all(tree.export_to_tikz().with_legend().to_string().as_bytes())?;

  Ok(())
}

/// Load and process the Iris data.
#[allow(dead_code)]
fn process() -> Result<()> {
  // Path to dataset.
  let path = Path::new("data/iris.csv");

  // Download (if it doesn't exist) and load iris dataframe.
  let df_lazy = data::load_data(Some(&path))?;

  // TODO: Perform operations here...

  // Execute all lazy operations.
  let mut df = df_lazy.collect()?;
  let shuffled = df.sample_frac(1., false, true, None)?;
  dbg!(&shuffled);
  dbg!(&df);

  // let column_names = df.get_column_names_owned();
  let column_names = &df.get_column_names();
  let features = Data::try_from(&df.select(&column_names[0..4])?)?;
  dbg!(&features.names);
  dbg!(&features.data.shape());

  let target = Data::try_from(&df.select(&[column_names[4]])?)?;
  dbg!(&target.names);
  dbg!(&target.data.shape());

  // Save dataframe to disk.
  data::save_df(&mut df, &path).unwrap();

  Ok(())
}

fn main() -> Result<()> {
  // process()?;

  train()?;

  Ok(())
}
