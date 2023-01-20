import re
import datasets
import argparse
import random
import rich

from torch.utils.data import Dataset as TorchDataset
from datasets import concatenate_datasets
from datasets.load import load_dataset, load_from_disk
from datasets.arrow_dataset import Dataset


def load_multi_x_science(data_path: str, script_path: str) -> Dataset:
    data: Dataset = load_dataset(
        path=script_path,
        cache_dir=data_path,
    )
    r = iter(range(sum([v[0] for v in data.shape.values()])))
    datasets.disable_progress_bar()
    data = data.map(
        lambda x: {"index": next(r)},
        load_from_cache_file=False,
    )
    datasets.enable_progress_bar()
    return data


def load_mips_multi_x_science(data_path: str, script_path: str) -> Dataset:
    data = load_multi_x_science(
        data_path=data_path,
        script_path=script_path,
    )
    data = concatenate_datasets(list(data.values()))

    def remove_cite(example):
        example['mips_column'] = re.sub(
            r"\@cite_\d+", "cite", example['related_work'])
        return example

    data = data.map(
        remove_cite,
        load_from_cache_file=False,
    )
    data = data.remove_columns(
        ['mid', 'abstract', 'ref_abstract', 'related_work'])
    return data


def load_mips_arxiv(data_path: str, script_path: str) -> Dataset:
    data = load_from_disk(
        dataset_path=data_path,
    )
    data = concatenate_datasets(list(data.values()))

    data.rename_columns(
        column_mapping={
            "article_id": 'aid',
            "article_text": "mips_column",
        }
    )

    return data


class MultiXScienceDataset(TorchDataset):

    def __init__(
            self,
            args,
            mode: str,
            model_config,
            tokenizer,
            tokenizer_kwargs: dict,
            query_tokenizer,
            query_tokenizer_kwargs: dict,
            select_indices: list = None,
    ) -> None:

        self.args = args

        self.data = load_multi_x_science(
            data_path=self.args.data_path,
            script_path=self.args.data_script_path,
        )[mode]

        if select_indices is not None:
            self.data = self.data.select(indices=select_indices)

        if self.args.clean_cache:
            mips_data_clean = self.data.cleanup_cache_files()
            rich.print(f"{mips_data_clean} cache file(s) removed.")

        self.doc_sep = self.args.doc_sep

        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.decoder_tokenizer_kwargs = tokenizer_kwargs.copy()
        decoder_max_len = model_config.max_decoder_position_embeddings if self.args.decoder_max_length is None else self.args.decoder_max_length
        self.decoder_tokenizer_kwargs['max_length'] = decoder_max_len

        self.query_tokenizer = query_tokenizer
        self.query_tokenizer_kwargs = query_tokenizer_kwargs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict:
        # PRIMERA use abstract + ref_abstract as model input.
        item = self.data[index]

        encoder_input = self.doc_sep.join(
            [item['abstract']] + item['ref_abstract']['abstract'])
        encoder_input = self.tokenizer(
            encoder_input,
            **self.tokenizer_kwargs,
        )
        encoder_input = {k: v.squeeze() for k, v in encoder_input.items()}

        query_input = item['abstract']
        query_input = self.query_tokenizer(
            query_input,
            **self.query_tokenizer_kwargs,
        )
        query_input = {f"query_{k}": v.squeeze()
                       for k, v in query_input.items()}

        decoder_input = re.sub(r"\@cite_\d+", "cite", item['related_work'])

        target_str = {"target_str": decoder_input}

        decoder_input = self.tokenizer(
            decoder_input,
            **self.decoder_tokenizer_kwargs,
        )
        decoder_input = {f"decoder_{k}": v.squeeze()
                         for k, v in decoder_input.items()}

        item_index = {
            "index": item['index'],
            # "arxiv_index": item['aid'],
        }

        return {**encoder_input, **query_input, **decoder_input, **item_index, **target_str}


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

        data = load_dataset(
            'multi_x_science_sum',
            cache_dir=self.args.data_path,
        )[mode].to_pandas()
        data = data.groupby(by='mid').agg({
            "aid": lambda x: x.to_list()[0],
            "abstract": lambda x: x.to_list()[0],
            "related_work": lambda x: x.to_list(),
        })
        self.data = Dataset.from_pandas(data, preserve_index=False)

        if self.args.dry_run:
            self.data = self.data.select(range(self.args.batch_size))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict:
        item = self.data[index]

        query_input = item['abstract']
        query_input = self.query_tokenizer(
            query_input,
            **self.query_tokenizer_kwargs,
        )
        query_input = {f"query_{k}": v.squeeze()
                       for k, v in query_input.items()}

        mips_input = random.choice(item['related_work'])
        mips_input = re.sub(r"\@cite_\d+", "cite", mips_input)
        mips_input = self.mips_tokenizer(
            mips_input,
            **self.mips_tokenizer_kwargs,
        )
        mips_input = {f"mips_{k}": v.squeeze() for k, v in mips_input.items()}

        return {**query_input, **mips_input}


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


if __name__ == '__main__':
    args = get_args()
    args.func(args)
