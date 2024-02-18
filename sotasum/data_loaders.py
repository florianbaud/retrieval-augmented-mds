import re
import datasets
import argparse
import random
import rich

from torch.utils.data import Dataset as TorchDataset
from datasets import concatenate_datasets, Dataset, DatasetDict
from datasets.load import load_dataset, load_from_disk


def disable_progress_bar(f):
    def wrapper(*args, **kwargs):
        if datasets.is_progress_bar_enabled():
            datasets.disable_progress_bar()
        d = f(*args, **kwargs)
        if not datasets.is_progress_bar_enabled():
            datasets.enable_progress_bar()
        return d

    return wrapper


@disable_progress_bar
def load_multi_x_science(data_path: str, script_path: str) -> DatasetDict:
    data: Dataset = load_dataset(
        path=script_path,
        cache_dir=data_path,
    )
    r = iter(range(sum([v[0] for v in data.shape.values()])))
    data = data.map(
        lambda x: {"index": next(r)},
        load_from_cache_file=False,
    )

    return data


@disable_progress_bar
def load_mips_multi_x_science(
    data_path: str,
    script_path: str,
    column: str,
) -> Dataset:
    data = load_multi_x_science(
        data_path=data_path,
        script_path=script_path,
    )
    data = concatenate_datasets(list(data.values()))

    if column == "ref_abstract":
        df = data.to_pandas()

        df = df.apply(
            lambda x: {
                **x.to_dict(),
                "ref_abstract_abstract": x["ref_abstract"]["abstract"],
            },
            axis=1,
            result_type="expand",
        ).explode(["ref_abstract_abstract"])

        df = df[df["ref_abstract_abstract"] != ""].reset_index()

        df = (
            df.groupby("ref_abstract_abstract")
            .agg(
                {
                    "index": lambda x: x.to_list(),
                    "aid": lambda x: x.to_list(),
                }
            )
            .reset_index()
        )

        df.rename(
            columns={"ref_abstract_abstract": "mips_column"},
            inplace=True,
        )

        data = Dataset.from_pandas(df)
    else:

        def remove_cite(example):
            example["mips_column"] = re.sub(
                r"\@cite_\d+", "cite", example["related_work"]
            )
            return example

        data = data.map(
            remove_cite,
            load_from_cache_file=False,
        )

    return data


@disable_progress_bar
def load_mips_arxiv(data_path: str) -> Dataset:
    data = load_from_disk(
        dataset_path=data_path,
    )
    data = concatenate_datasets(list(data.values()))

    def join_clean(text: list) -> dict:
        abstract_text = " ".join(text).replace("<S>", "").replace("</S>", "")
        abstract_text = re.sub(r"\s{2,}", " ", abstract_text)
        abstract_text = abstract_text.strip()
        return abstract_text

    data = data.map(
        lambda x: {"abstract_text_str": [join_clean(a) for a in x["abstract_text"]]},
        desc="Cleaning data",
        batched=True,
        load_from_cache_file=False,
    )

    data = data.rename_columns(
        column_mapping={
            "article_id": "aid",
            "abstract_text_str": "mips_column",
        }
    )

    return data


# @disable_progress_bar
def load_mips_arxiv2(data_path: str) -> Dataset:
    ds = Dataset.from_parquet(data_path)

    def clean_arxiv(text: str) -> str:
        text = text.replace("\n", " ").strip()
        text = re.sub(r"\$+(.*?)\$+|\\\[(.*?)\\\]", "@math", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text

    ds = ds.map(
        lambda x: {"abstract_clean": [clean_arxiv(a) for a in x["abstract"]]},
        batched=True,
        load_from_cache_file=False,
        desc="Cleaning data",
        num_proc=4,
    )

    ds = ds.rename_columns(
        column_mapping={
            "__index_level_0__": "aid",
            "abstract_clean": "mips_column",
        }
    )

    ds = Dataset.from_pandas(
        ds.to_pandas().drop_duplicates(subset=["mips_column"]), preserve_index=False
    )

    return ds


def load_mips_arxiv_x_science(
    arxiv_data_path: str,
    multix_data_path: str,
    multix_script_path: str,
    multix_column: str,
) -> Dataset:
    data: Dataset = concatenate_datasets(
        (
            load_mips_arxiv(
                data_path=arxiv_data_path,
            ),
            load_mips_multi_x_science(
                data_path=multix_data_path,
                script_path=multix_script_path,
                column=multix_column,
            ),
        )
    ).remove_columns(
        column_names=[
            "related_work",
            "abstract",
            "mid",
            "ref_abstract",
            "index",
            "article_text",
            "abstract_text",
            "labels",
            "section_names",
            "sections",
        ],
    )
    return data


class MultiXScienceDataset(TorchDataset):
    def __init__(
        self,
        args,
        mode: str,
        tokenizer,
        tokenizer_kwargs: dict,
        query_tokenizer,
        query_tokenizer_kwargs: dict,
        select_indices: list = None,
        decoder_max_length: int = 1024,
    ) -> None:
        self.args = args

        self.data = load_multi_x_science(
            data_path=self.args.data_path,
            script_path=self.args.data_script_path,
        )[mode].to_pandas()

        self.data = self.data.merge(
            self.data["aid"].value_counts(),
            right_index=True,
            left_on="aid",
        )
        self.data.rename(columns={"count": "aid_counts"}, inplace=True)

        self.data = Dataset.from_pandas(self.data, preserve_index=False)

        if select_indices is not None:
            if select_indices == "rand":
                select_indices = random.sample(range(len(self.data)))
            self.data = self.data.select(indices=select_indices)

        if self.args.clean_cache:
            mips_data_clean = self.data.cleanup_cache_files()
            rich.print(f"{mips_data_clean} cache file(s) removed.")

        self.doc_sep: str = self.args.doc_sep

        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

        self.decoder_tokenizer_kwargs = tokenizer_kwargs.copy()
        self.decoder_tokenizer_kwargs["max_length"] = decoder_max_length + 1

        self.query_tokenizer = query_tokenizer
        self.query_tokenizer_kwargs = query_tokenizer_kwargs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict:
        # PRIMERA use abstract + ref_abstract as model input.
        item = self.data[index]

        if self.args.source_memory:
            encoder_input = item["abstract"]
        else:
            encoder_input = [item["abstract"]]
            encoder_input += [a for a in item["ref_abstract"]["abstract"] if a != ""]
            encoder_input = self.doc_sep.join(encoder_input)

        encoder_input = self.tokenizer(
            encoder_input,
            **self.tokenizer_kwargs,
        )
        encoder_input = {k: v.squeeze() for k, v in encoder_input.items()}

        if self.args.source_memory:
            query_input = [item["abstract"]]
            query_input += [a for a in item["ref_abstract"]["abstract"] if a != ""]
            query_input = " ".join(query_input)
        else:
            query_input = item["abstract"]

        query_input = self.query_tokenizer(
            query_input,
            **self.query_tokenizer_kwargs,
        )
        query_input = {f"query_{k}": v.squeeze() for k, v in query_input.items()}

        decoder_input = re.sub(r"\@cite_\d+", "cite", item["related_work"])

        target_str = {"target_str": decoder_input}

        decoder_input = self.tokenizer(
            decoder_input,
            **self.decoder_tokenizer_kwargs,
        )
        decoder_input = {
            f"decoder_{k}": v.squeeze()[1:] for k, v in decoder_input.items()
        }

        item_index = {
            "index": item["index"],
            "aid": item["aid"],
            "aid_counts": item["aid_counts"],
            "abstract": item["abstract"],
        }

        return {
            **encoder_input,
            **query_input,
            **decoder_input,
            **item_index,
            **target_str,
        }


class MultiXScienceDualDataset(MultiXScienceDataset):
    def __init__(
        self,
        args,
        mode: str,
        tokenizer,
        tokenizer_kwargs: dict,
        query_tokenizer,
        query_tokenizer_kwargs: dict,
        select_indices: list = None,
        decoder_max_length: int = 1024,
    ) -> None:
        super().__init__(
            args,
            mode,
            tokenizer,
            tokenizer_kwargs,
            query_tokenizer,
            query_tokenizer_kwargs,
            select_indices,
            decoder_max_length,
        )

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, index) -> dict:
        item = self.data[index]

        encoder_input = [item["abstract"]]
        encoder_input += [a for a in item["ref_abstract"]["abstract"] if a != ""]
        encoder_input_str = self.doc_sep.join(encoder_input)

        encoder_input = self.tokenizer(
            encoder_input_str,
            **self.tokenizer_kwargs,
        )
        encoder_input = {k: v.squeeze() for k, v in encoder_input.items()}

        query_input = item["abstract"]

        query_input = self.query_tokenizer(
            query_input,
            **self.query_tokenizer_kwargs,
        )
        query_input = {f"query_{k}": v.squeeze() for k, v in query_input.items()}

        decoder_input = re.sub(r"\@cite_\d+", "cite", item["related_work"])

        target_str = {"target_str": decoder_input}

        decoder_input = self.tokenizer(
            decoder_input,
            **self.decoder_tokenizer_kwargs,
        )
        decoder_input = {
            f"decoder_{k}": v.squeeze()[1:] for k, v in decoder_input.items()
        }

        item_index = {
            "index": item["index"],
            "aid": item["aid"],
            "aid_counts": item["aid_counts"],
            "input_str": encoder_input_str,
        }

        return {
            **encoder_input,
            **query_input,
            **decoder_input,
            **item_index,
            **target_str,
        }


class MultiXScienceAggregatedDataset(MultiXScienceDataset):
    def __init__(
        self,
        args,
        mode: str,
        tokenizer,
        tokenizer_kwargs: dict,
        query_tokenizer,
        query_tokenizer_kwargs: dict,
        select_indices: list = None,
        decoder_max_length: int = 1024,
    ) -> None:
        super().__init__(
            args,
            mode,
            tokenizer,
            tokenizer_kwargs,
            query_tokenizer,
            query_tokenizer_kwargs,
            select_indices,
            decoder_max_length,
        )

        self.data = (
            self.data.to_pandas()
            .groupby("aid")
            .agg(
                {
                    "mid": lambda x: x.to_list()[0],
                    "abstract": lambda x: x.to_list()[0],
                    "ref_abstract": lambda x: {
                        k: [j for i in x.to_list() for j in i[k] if bool(j)]
                        for k in x.to_list()[0].keys()
                    },
                    "related_work": lambda x: " ".join(x.to_list()),
                    "aid_counts": sum,
                    "index": lambda x: x.to_list()[0],
                }
            )
        )
        self.data = Dataset.from_pandas(self.data, preserve_index=True)
        # self.data.rename_column()

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, index) -> dict:
        return super().__getitem__(index)


class PretrainMultiXScienceDataset(TorchDataset):
    def __init__(
        self,
        args,
        mode: str,
        query_tokenizer,
        query_tokenizer_kwargs: dict,
        mips_tokenizer,
        mips_tokenizer_kwargs: dict,
    ) -> None:
        self.args = args
        self.mode = mode

        self.query_tokenizer = query_tokenizer
        self.mips_tokenizer = mips_tokenizer

        tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
        }
        self.query_tokenizer_kwargs = query_tokenizer_kwargs
        self.mips_tokenizer_kwargs = mips_tokenizer_kwargs

        self.query_tokenizer_kwargs.update(tokenizer_kwargs)
        self.mips_tokenizer_kwargs.update(tokenizer_kwargs)

        self.data = load_dataset(
            "multi_x_science_sum",
            cache_dir=self.args.data_path,
        )[self.mode].to_pandas()

        self.data = self.data.merge(
            self.data["aid"].value_counts(),
            right_index=True,
            left_on="aid",
            suffixes=("", "_counts"),
        )

        if self.mode == "train":
            self.data = self.data.groupby(by="mid").agg(
                {
                    "aid": lambda x: x.to_list()[0],
                    "aid_counts": lambda x: x.to_list()[0],
                    "abstract": lambda x: x.to_list()[0],
                    "related_work": lambda x: x.to_list(),
                }
            )
        else:
            self.data.drop_duplicates(["aid"], inplace=True)

        self.data = Dataset.from_pandas(self.data, preserve_index=False)

        if self.args.dry_run:
            self.data = self.data.select(range(self.args.batch_size))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict:
        item = self.data[index]

        query_input = item["abstract"]
        # query_input = self.args.doc_sep.join(
        #     [item['abstract']] + item['ref_abstract']['abstract'])
        query_input = self.query_tokenizer(
            query_input,
            **self.query_tokenizer_kwargs,
        )
        query_input = {f"query_{k}": v.squeeze() for k, v in query_input.items()}

        mips_input = (
            random.choice(item["related_work"])
            if self.mode == "train"
            else item["related_work"]
        )
        mips_input = re.sub(r"\@cite_\d+", "cite", mips_input)
        mips_input = self.mips_tokenizer(
            mips_input,
            **self.mips_tokenizer_kwargs,
        )
        mips_input = {f"mips_{k}": v.squeeze() for k, v in mips_input.items()}

        index_input = {
            "aid": item["aid"],
            "counts": item["aid_counts"],
            "abstract": item["abstract"],
        }

        return {**query_input, **mips_input, **index_input}


class PretrainAbstractMultiXScienceDataset(TorchDataset):
    def __init__(
        self,
        args,
        mode: str,
        query_tokenizer,
        query_tokenizer_kwargs: dict,
        mips_tokenizer,
        mips_tokenizer_kwargs: dict,
    ) -> None:
        self.args = args
        self.mode = mode

        self.query_tokenizer = query_tokenizer
        self.mips_tokenizer = mips_tokenizer

        tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
        }
        self.query_tokenizer_kwargs = query_tokenizer_kwargs
        self.mips_tokenizer_kwargs = mips_tokenizer_kwargs

        self.query_tokenizer_kwargs.update(tokenizer_kwargs)
        self.mips_tokenizer_kwargs.update(tokenizer_kwargs)

        self.data = load_dataset(
            "multi_x_science_sum",
            cache_dir=self.args.data_path,
        )[self.mode].to_pandas()

        self.data = self.data.groupby("mid").agg(
            {
                "aid": lambda x: x.to_list()[0],
                "abstract": lambda x: x.to_list()[0],
                "ref_abstract": lambda x: {
                    k: [j for i in x.to_list() for j in i[k] if bool(j)]
                    for k in x.to_list()[0].keys()
                },
            }
        )
        self.data["counts"] = self.data["ref_abstract"].apply(
            lambda x: len(x["abstract"])
        )

        self.data = Dataset.from_pandas(self.data, preserve_index=False)

        if self.args.dry_run:
            self.data = self.data.select(range(self.args.batch_size))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict:
        item = self.data[index]

        query_input = item["abstract"]
        query_input = self.query_tokenizer(
            query_input,
            **self.query_tokenizer_kwargs,
        )
        query_input = {f"query_{k}": v.squeeze() for k, v in query_input.items()}

        mips_input = random.choice(item["ref_abstract"]["abstract"])
        # mips_input = re.sub(r"\@cite_\d+", "cite", mips_input)
        mips_input = self.mips_tokenizer(
            mips_input,
            **self.mips_tokenizer_kwargs,
        )
        mips_input = {f"mips_{k}": v.squeeze() for k, v in mips_input.items()}

        index_input = {
            "aid": item["aid"],
            "counts": item["counts"],
            "abstract": item["abstract"],
        }

        return {**query_input, **mips_input, **index_input}


def clean(args):
    data_path = args.data_path

    mips_data = load_mips_multi_x_science(
        data_path=data_path,
        script_path="multi_x_science_sum",
    )
    mips_data_clean = mips_data.cleanup_cache_files()
    rich.print(f"{mips_data_clean} cache file(s) removed.")


def get_args(args: str = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers(
        title="Data loaders utilities",
        description="Utility for cleaning cache's datasets",
    )

    clean_parser = subparser.add_parser("clean", help="Clean cache")
    clean_parser.add_argument("--data-path", type=str)
    clean_parser.set_defaults(func=clean)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.func(args)
