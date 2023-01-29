// Copyright (c) 2023 Victor I. Afolabi
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use anyhow::Result;
use polars::prelude::*;
use reqwest::{blocking::Client, IntoUrl};

use std::{fs::File, io::Cursor, path::Path};

/// Save dataframe to disk.
pub fn save_df(df: &mut DataFrame, path: &Path) -> PolarsResult<()> {
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
  // Overwrite the "species" schema.
  let fields = [Field::new("species", DataType::Categorical(None))];
  let schema = Schema::from(fields.into_iter());

  let df = match path {
    // Load data from file (if it exists).
    Some(p) if p.is_file() => {
      println!("Loading data from {}", p.display());

      CsvReader::from_path(p)?
        .has_header(true)
        .with_dtypes(Some(&schema))
        .finish()?
    }
    // Download data.
    _ => {
      println!("Downloading data...");

      download_as_df("https://j.mp/iriscsv", Some(&schema))?
    }
  };

  Ok(df)
}

/// Load Iris dataset into a lazy dataframe from file path if given, otherwise,
/// download it.
pub fn load_lazy_data(path: Option<&Path>) -> Result<LazyFrame> {
  // Overwrite the "species" schema.
  let fields = [Field::new("species", DataType::Categorical(None))];
  let schema = Schema::from(fields.into_iter());

  let df = match path {
    // Load data from file (if it exists).
    Some(p) if p.is_file() => {
      println!("Loading data from {}", p.display());

      LazyCsvReader::new(&p)
        .has_header(true)
        .with_dtype_overwrite(Some(&schema))
        .finish()?
    }
    // Download data.
    _ => {
      println!("Downloading data...");

      download_as_df("https://j.mp/iriscsv", Some(&schema))?.lazy()
    }
  };

  Ok(df)
}

/// Download data from a given URL.
fn download<U: IntoUrl>(url: U) -> Result<Vec<u8>> {
  let data: Vec<u8> = Client::new().get(url).send()?.text()?.bytes().collect();

  Ok(data)
}

fn download_as_df<'a, U: IntoUrl>(
  url: U,
  schema: Option<&'a Schema>,
) -> Result<DataFrame> {
  let data = download(url)?;

  let df = CsvReader::new(Cursor::new(data))
    .has_header(true)
    .with_dtypes(schema)
    .finish()?;

  Ok(df)
}
