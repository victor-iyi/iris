from pathlib import Path

import polars as pl


def load_data(path: str | Path) -> pl.LazyFrame:
    """Load dataframe from a path if it doesn't exist.

    Args:
      path (str | Path): Path to a csv file.

    Returns:
      pl.LazyFrame: A lazy dataframe object.
        `LazyFrame` builds a query plan, nothing is executed until
        explicit call of `.collect()` to turn into a `DataFrame` object.

    """

    if isinstance(path, str):
        path = Path(path)

    # overwrite the species datatype.
    schema = {
        'species': pl.Categorical(),
    }

    if path.is_file():
        print(f'Loading from {path}...')
        return pl.scan_csv(path, dtypes=schema)

    print('Downloading data...')

    # load data from internet.
    return pl.read_csv('https://j.mp/iriscsv', dtypes=schema).lazy()


def save_df(df: pl.DataFrame, path: str | Path, force: bool = False) -> None:
    """Save dataframe to disk.

    Args:
      df (pl.DataFrame): DataFrame to save.
      path (str | Path): Path to save dataframe.
      force (bool, optional): Overwrite saved file. Default False.

    """
    if isinstance(path, str):
        path = Path(path)

    if path.is_file() and not force:
        print(f'{path} already exist.')
        return

    print(f'File savedtto {path}')
    df.write_csv(path)


if __name__ == '__main__':
    path = Path('data/iris.csv')

    df = load_data(path)

    print(df.collect())

    save_df(df.collect(), path)
