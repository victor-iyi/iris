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
  // Datatypes.
  let dtypes: &[DataType; 5] = &[
    DataType::Float64,           // sepal_length
    DataType::Float64,           // sepal_width
    DataType::Float64,           // petal_length
    DataType::Float64,           // petal_width
    DataType::Categorical(None), // species
  ];

  let df = match path {
    // Load data from file (if it exists).
    Some(p) if p.is_file() => {
      println!("Loading data from {}", p.display());
      CsvReader::from_path(&p)?
        .has_header(true)
        .with_dtypes_slice(Some(dtypes))
        .finish()?
    }
    _ => {
      // Download data.
      println!("Downloading data...");
      let data: Vec<u8> = Client::new()
        .get("https://j.mp/iriscsv")
        .send()?
        .text()?
        .bytes()
        .collect();

      CsvReader::new(Cursor::new(data))
        .has_header(true)
        .with_dtypes_slice(Some(dtypes))
        .finish()?
    }
  };

  Ok(df)
}
