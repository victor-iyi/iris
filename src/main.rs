use eyre::Result;
use iris::{data::load_data, Data};
use std::path::Path;

fn main() -> Result<()> {
  // Path to dataset.
  let path = Path::new("data/iris.csv");

  // Download (if it doesn't exist) and load iris dataframe.
  let df_lazy = load_data(Some(&path))?;

  // TODO: Perform operations here...

  // Execute all lazy operations.
  let df = df_lazy.collect()?;
  dbg!(&df);

  // let column_names = df.get_column_names_owned();
  let column_names = &df.get_column_names();
  let features = Data::new(&df.select(&column_names[0..4])?);
  dbg!(&features.names);
  dbg!(&features.data.shape());

  let target = Data::new(&df.select(&[column_names[4]])?);
  dbg!(&target.names);
  dbg!(&target.data.shape());

  // // Save dataframe to disk.
  // save_df(&mut df, &path).unwrap();
  Ok(())
}
