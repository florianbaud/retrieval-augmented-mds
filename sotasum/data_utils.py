import pandas as pd
import argparse
import os

from glob import glob
from pyarrow import json
from datasets import Dataset, DatasetDict


def _read_json(path: str) -> Dataset:
    arrow_table = json.read_json(
        path,
        read_options=json.ReadOptions(
            block_size=2**21,
        ),
    )
    split = path.split('/')[-1].split('.')[0]

    # article_length = pc.list_value_length(
    #     arrow_table['article_text']).to_pylist()
    # article_id_list = pa.array(
    #     [[i]*l for i, l in zip(arrow_table['article_id'].to_pylist(), article_length)])
    # arrow_table = arrow_table.add_column(0, "article_id_list", article_id_list)

    # arrow_table = pa.table(
    #     data=[
    #         pc.list_flatten(arrow_table['article_id_list']),
    #         pc.list_flatten(arrow_table['article_text']),
    #     ],
    #     names=['article_id', 'article_text'],
    # )

    data = Dataset(
        arrow_table=arrow_table,
        split=split,
    )

    return data


def build_scientific_papers(save_path: str = None, paths: list = None):
    """
    Download arxiv-dataset first : https://github.com/armancohan/long-summarization
    """

    scientific_papers = DatasetDict(
        {p.split('/')[-1].split('.')[0]: _read_json(p) for p in paths},
    )

    if save_path is not None:
        scientific_papers.save_to_disk(save_path)

    return scientific_papers


def build_open_alex_data():
    """
    Download data from s3 with Docker :

    $ docker run --rm -v $OPENALEX_DIR:/openalex-snapshot/ -it amazon/aws-cli s3 sync "s3://openalex" "/openalex-snapshot" --no-sign-request
    """

    openalex_dir = ""  # $OPENALEX_DIR

    works = glob(f'{openalex_dir}/data/works/updated_date=*/*')
    work = works[0]

    column = ["id"]

    for i, df in enumerate(pd.read_json(work, lines=True, chunksize=2**17)):
        df[column].to_parquet(f"{openalex_dir}/arrow/part_000_{i}.parquet")

    return None


# def read_openalex_work(work_file: str, chunk_size: int = 2**10):
#     with gzip.open(work_file) as wf:
#         while lines := wf.readlines(chunk_size):
#             chunk = pa.BufferReader(b"".join(lines))
#             pd.read_json(chunk, lines=True)
#             # try:
#             #     yield pa.json.read_json(chunk)
#             # except:
#             #     yield None


def get_args(args: str = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--arxiv-directory", type=str, default=None)
    parser.add_argument("--arxiv-save-path", type=str,
                        default="./arxiv-dataset-custom/")

    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    args = get_args()

    path = os.path.join(args.arxiv_directory, "*.txt")
    paths = glob(path)
    if bool(paths):
        build_scientific_papers(
            save_path=args.arxiv_data_path,
            paths=paths,
        )
