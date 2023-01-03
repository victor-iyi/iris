use eyre::Result;
use polars::prelude::*;
use reqwest::blocking::Client;

use std::{fs::File, io::Cursor, path::Path};

/// Save dataframe to disk.
pub fn save_df(df: &mut DataFrame, path: &Path) -> Result<()> {
  if !path.exists() {
    // See if parent folder exists.
    let parent = path.parent().unwrap();
    if !parent.is_dir() {
      std::fs::create_dir_all(&parent).unwrap();
    }
    // Create file.
    let mut file = File::create(&path)?;

    // Save dataframe.
    CsvWriter::new(&mut file).finish(df)?;
    println!("File saved to:  {}", path.display());
  } else {
    println!("File already exists.");
  }

  Ok(())
}

/// Load Iris dataset into a dataframe from file path if given, otherwise,
/// download it.
pub fn load_data(path: Option<&Path>) -> Result<DataFrame> {
  let df = match path {
    Some(p) if p.is_file() => CsvReader::from_path(&p)?.has_header(true).finish()?,
    _ => {
      let data: Vec<u8> = Client::new()
        .get("https://j.mp/iriscsv")
        .send()?
        .text()?
        .bytes()
        .collect();

      CsvReader::new(Cursor::new(data))
        .has_header(true)
        .finish()?
    }
  };

  Ok(df)
}
