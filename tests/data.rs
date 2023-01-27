use polars::prelude::*;

mod utils;

/// Test the dataframe has the correct fields & schema.
#[test]
fn test_load_data_schema() {
  let df = utils::load_df();

  let fields = [
    Field::new("sepal_length", DataType::Float64),
    Field::new("sepal_width", DataType::Float64),
    Field::new("petal_length", DataType::Float64),
    Field::new("petal_width", DataType::Float64),
    Field::new("species", DataType::Categorical(None)),
  ];
  assert_eq!(df.fields(), &fields);

  let schema = Schema::from(fields.into_iter());
  assert_eq!(df.schema(), schema);
}

/// Test the dataframe has the correct shape.
#[test]
fn test_load_data_shape() {
  let df = utils::load_df();

  assert_eq!(df.shape(), (150, 5));
}
