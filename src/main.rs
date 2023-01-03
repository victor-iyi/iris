use eyre::Result;
use iris::{load_data, save_df};
use std::path::Path;

fn main() -> Result<()> {
  // Path to dataset.
  let path = Path::new("data/iris.csv");

  // Download (if it doesn't exist) and load iris dataframe.
  let mut df = load_data(Some(&path))?;
  dbg!(&df);

  // Save dataframe to disk.
  save_df(&mut df, &path).unwrap();

  Ok(())
}
