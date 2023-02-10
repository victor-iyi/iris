mod utils;

// #[test]
// fn test_model() {
//   // Load dataframe.
//   let df = utils::load_df();
//
//   // train model.
//   let model = utils::get_model(&df);
// }

#[test]
fn test_prediction() {
  // Make predictions.
  let data = ndarray::array!(
    [6.7, 3.1, 4.7, 1.5], // versicolor
    [5.8, 2.7, 3.9, 1.2], // versicolor
    [4.4, 3.0, 1.3, 0.2], // setosa
    [5.4, 3.4, 1.5, 0.4], // setosa
    [5.5, 4.2, 1.4, 0.2], // virginica
    [6.9, 3.2, 5.7, 2.3], // versicolor
  );
  dbg!(data.shape());
  assert_eq!(data.shape(), [6, 4]);

  // let df = utils::load_df();
  // let model = utils::get_model(&df);

  // let pred = model.predict(&data);
  // dbg!(&pred);
  // let cm = pred.confusion_matrix(&data);
  // assert!(cm.is_ok());
  //
  // let cm = cm.unwrap();
  // assert!(cm.accuracy() > 0.9);
}
