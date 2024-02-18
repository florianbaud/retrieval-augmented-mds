import faiss
import torch
import torch.nn as nn
import numpy as np
import datasets
import multiprocess
import time
import cloudpickle
import rich
import shutil

from .data_loaders import load_mips_multi_x_science, load_mips_arxiv, load_mips_arxiv2
from .model_config import ModelConfig
from .pretrain import retriever_metrics
from pathlib import Path
from random import random
from adapters import AutoAdapterModel
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from datasets.arrow_dataset import Dataset
from transformers import (
    LongformerTokenizer,
    LongformerConfig,
    LongformerModel,
    AutoTokenizer,
)
from transformers.models.longformer.modeling_longformer import (
    LongformerBaseModelOutputWithPooling,
)


@dataclass
class MipsModelOutput(ModelOutput):
    scores: torch.FloatTensor = None
    mips_last_hidden_state: torch.FloatTensor = None
    memory_outputs: dict = None
    memory_input_ids: torch.FloatTensor = None
    memory_attention_mask: torch.FloatTensor = None
    metrics: dict = None
    examples: list = None
    query_cls: np.ndarray = None


def _layer_norm(x: torch.Tensor, variance_epsilon: float = 1e-12) -> torch.Tensor:
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    x = (x - u) / torch.sqrt(s + variance_epsilon)
    return x


# see http://ulrichpaquet.com/Papers/SpeedUp.pdf theorem 5


def get_phi(xb: np.ndarray):
    return (xb**2).sum(1).max()


def augment_xb(xb: np.ndarray, phi=None):
    norms = (xb**2).sum(1)
    if phi is None:
        phi = norms.max()
    extracol = np.sqrt(phi - norms)
    # extracol = np.where(~np.isnan(extracol), extracol, 0)
    return np.hstack((xb, extracol.reshape(-1, 1)))


def augment_xq(xq: np.ndarray):
    extracol = np.zeros(len(xq), dtype="float32")
    return np.hstack((xq, extracol.reshape(-1, 1)))


def timer(name: str = ""):
    def timer_(f):
        def wrapper(*args, **kwargs):
            start = time.time()
            r = f(*args, **kwargs)
            end = time.time()
            print(f"{name} time :", end - start)
            return r

        return wrapper

    return timer_


class MipsEncoder(nn.Module):
    def __init__(self, args: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.args = args
        self.model_name = self.args.mips_model_name
        tokenizer_kwargs = {"max_length": self.args.mips_tok_max_length}

        # self.tokenizer = LongformerTokenizer.from_pretrained(
        #     pretrained_model_name_or_path=self.model_name,
        # )

        # config = LongformerConfig.from_pretrained(
        #     pretrained_model_name_or_path=self.model_name,
        #     gradient_checkpointing=self.args.gradient_checkpointing,
        # )
        # self.model = LongformerModel.from_pretrained(
        #     pretrained_model_name_or_path=self.model_name,
        #     config=config,
        #     # add_pooling_layer=False,
        # )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoAdapterModel.from_pretrained(self.model_name)
        self.model.load_adapter(
            "allenai/specter2", source="hf", load_as="specter2", set_active=True
        )

        if self.args.mips_state_dict is not None:
            mips_state_dict = torch.load(self.args.mips_state_dict)
            self.model.load_state_dict(mips_state_dict)

        self.tokenizer_opt = tokenizer_kwargs

    def forward(self, tokens) -> LongformerBaseModelOutputWithPooling:
        input_ids = tokens["input_ids"]

        # put global attention on <s> token
        global_attention_mask = torch.zeros_like(input_ids, device=input_ids.device)
        global_attention_mask[:, 0] = 1

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=tokens["attention_mask"],
            global_attention_mask=global_attention_mask,
        )
        return outputs

    @torch.inference_mode()
    def encode(self, tokens) -> LongformerBaseModelOutputWithPooling:
        return self(tokens)

    def tokenize(self, text: list) -> dict:
        tokens = self.tokenizer(
            text=text,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            **self.tokenizer_opt,
        ).to(self.model.device)
        tokens = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
        return tokens


class Mips(nn.Module):
    def __init__(self, args: ModelConfig = ModelConfig()) -> None:
        super().__init__()
        self.args = args

        self.tmp_folder = Path(self.args.mips_tmp_folder)
        self.embeddings_tmp_folder = self.tmp_folder / "embeddings_tmp"

        self.mips_folder = self.tmp_folder / "mips"
        self.index_file = self.mips_folder / "index.faiss"
        self.max_norm_file = self.mips_folder / "max_norm.pkl"
        self.embeddings_folder = self.mips_folder / "embeddings"

        if args.mips_dataset == "multi_x_science":
            self.data = load_mips_multi_x_science(
                data_path=None,
                script_path=self.args.mips_data_script_path,
                column="ref_abstract" if self.args.source_memory else "related_work",
            )
        elif args.mips_dataset == "arxiv":
            self.data = load_mips_arxiv(
                data_path=args.mips_arxiv_data_path,
            )
        elif args.mips_dataset == "arxiv2":
            self.data = load_mips_arxiv2(
                data_path=args.mips_arxiv_data_path,
            )
        else:
            assert False, f"{args.mips_dataset} not found."

        if isinstance(args.mips_db_max_size, int):
            self.data = self.data.select(range(0, args.mips_db_max_size))

        self.encoder = MipsEncoder(self.args)

        self.memory_tokenizer_kwargs = {
            "max_length": self.args.memory_tok_max_length,
        }
        self.memory_tokenizer: LongformerTokenizer = (
            LongformerTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.args.memory_model_name,
            )
        )
        memory_config = LongformerConfig.from_pretrained(
            pretrained_model_name_or_path=self.args.memory_model_name,
            gradient_checkpointing=self.args.gradient_checkpointing,
        )
        self.memory_encoder = LongformerModel.from_pretrained(
            pretrained_model_name_or_path=self.args.memory_model_name,
            config=memory_config,
            # add_pooling_layer=False,
        )

        self.eos_id = self.memory_tokenizer.eos_token_id
        self.bos_id = self.memory_tokenizer.bos_token_id

        self.string_factory = self.args.mips_string_factory
        self.train_size = self.args.mips_train_size
        self.metric_type = self.args.mips_metric_type
        self.normalize = self.args.mips_normalize

        self.max_norm = None
        self.has_context = False
        self.rebuilt_steps = [0]
        self.load_from_cache_file = False
        self.text_column = "mips_column"
        self.index_column = "aid"
        self.index_name = "mips_embeddings"
        self.embeddings = None
        self.embeddings_column = "embeddings"
        self.scale_topk = 16

    def encode_text2(self, rank: int, num_rank: int) -> None:
        chunck_size = (self.data.shape[0] // num_rank) + 1
        stop = (rank + 1) * chunck_size if rank + 1 < num_rank else self.data.shape[0]
        indices = range(rank * chunck_size, stop)
        embeddings = self.data.select(indices=indices)
        embeddings = embeddings.map(
            self._map_encode,
            batched=True,
            batch_size=self.args.mips_batch_size,
            num_proc=1,
            desc=f"Rank {rank} : Encoding...",
        )
        embeddings.set_format(
            type="numpy",
            columns=[self.embeddings_column],
            output_all_columns=True,
        )
        save_path = self.embeddings_tmp_folder / str(rank)
        embeddings.save_to_disk(save_path)

    @rank_zero_only
    def init_embeddings_folder(self) -> None:
        shutil.rmtree(self.embeddings_folder, ignore_errors=True)
        self.embeddings_folder.mkdir(parents=True, exist_ok=True)

    def encode_text(self, num_proc: int = 1, batch_size: int = 32) -> None:
        # self.new_fingerprint = f"{self.args.mips_cache_prefix}-{self.rebuilt_steps[-1]}-{self.args.mips_db_max_size}"
        desc = "Encoding"
        device = 0
        if not torch.cuda.is_available():
            device = "cpu"
            num_proc = 1
            print("Cuda is not available, encoding memory on single cpu...")
        self.eval()
        if num_proc > 1:
            self._init_context()
            self.embeddings: Dataset = self.data.map(
                self._map_encode,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                with_rank=True,
                # new_fingerprint=self.new_fingerprint,
                load_from_cache_file=self.load_from_cache_file,
                desc=desc,
            )
        elif num_proc == 1:
            fn_kwargs = {"rank": device}
            self.embeddings: Dataset = self.data.map(
                self._map_encode,
                batched=True,
                batch_size=batch_size,
                fn_kwargs=fn_kwargs,
                # new_fingerprint=self.new_fingerprint,
                load_from_cache_file=self.load_from_cache_file,
                desc=desc,
            )
        self.embeddings.set_format(
            type="numpy",
            columns=[self.embeddings_column],
            output_all_columns=True,
        )
        self.train()

    @rank_zero_only
    def build_index(self) -> None:
        files = self.embeddings_tmp_folder.glob("*")
        self.embeddings = datasets.concatenate_datasets(
            [Dataset.load_from_disk(f) for f in files]
        )

        # max_norm_fingerprint = self.new_fingerprint + "-max_norm"
        self.max_norm = self.embeddings.map(
            self._map_norm,
            batched=True,
            desc="Calculating max norm",
            load_from_cache_file=self.load_from_cache_file,
            # new_fingerprint=max_norm_fingerprint,
        )["norm"].max()

        if self.normalize and self.metric_type == faiss.METRIC_INNER_PRODUCT:
            # self.new_fingerprint += "-normalized"
            self.embeddings = self.embeddings.map(
                self._map_normalize,
                batched=True,
                desc="Normalization",
                load_from_cache_file=self.load_from_cache_file,
                # new_fingerprint=self.new_fingerprint,
            )

        if self.metric_type == faiss.METRIC_L2:
            # self.new_fingerprint += "-augmented"
            self.phi = self.embeddings.map(
                lambda x: {"phi": (x[self.embeddings_column] ** 2).sum(1)},
                batched=True,
                desc="Calculating phi...",
                load_from_cache_file=self.load_from_cache_file,
            )
            self.phi = self.phi["phi"].max()
            self.embeddings = self.embeddings.map(
                self._map_augment_xb,
                batched=True,
                desc="Augmenting data",
                load_from_cache_file=self.load_from_cache_file,
                # new_fingerprint=self.new_fingerprint,
            )

        self.embeddings.add_faiss_index(
            column=self.embeddings_column,
            index_name=self.index_name,
            string_factory=self.string_factory,
            train_size=self.train_size,
            metric_type=self.metric_type,
            faiss_verbose=True,
        )

        if isinstance(self.args.mips_nprobe, int):
            self.embeddings.get_index(
                self.index_name
            ).faiss_index.nprobe = self.args.mips_nprobe

    def _map_norm(self, x: dict) -> dict:
        norm = np.linalg.norm(x[self.embeddings_column], axis=1, keepdims=True)
        return {"norm": norm}

    def _map_encode(self, x: dict) -> dict:
        # self.encoder.to(rank)
        tokens = self.encoder.tokenize(x[self.text_column])
        output = self.encoder.encode(tokens)
        output = output.last_hidden_state[:, 0, :].cpu().float().numpy()
        return {self.embeddings_column: output}

    def _map_normalize(self, x: dict) -> dict:
        return {
            self.embeddings_column: self.l2_normalization(x[self.embeddings_column])
        }

    def _map_augment_xb(self, x: dict) -> dict:
        return {
            self.embeddings_column: augment_xb(x[self.embeddings_column], phi=self.phi)
        }

    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        if self.normalize and self.metric_type == faiss.METRIC_INNER_PRODUCT:
            query = self.l2_normalization(query)
        if self.metric_type == faiss.METRIC_L2:
            query = augment_xq(query)
        if not query.flags.c_contiguous:
            query = np.asarray(query, order="C")
        return query.astype(np.float32)

    def _init_context(self) -> None:
        if not self.has_context:
            multiprocess.set_start_method("spawn")
            self.has_context = True

    def search(self, queries: np.ndarray, ignore_indexes: list = None, k: int = 10):
        scores, indices = self.embeddings.get_index(self.index_name).faiss_index.search(
            queries,
            k + 1 if ignore_indexes is not None else k,
        )

        if ignore_indexes is not None:
            scores = [
                [s for i, s in enumerate(score) if ignore_indexes[j] != indices[j][i]][
                    :k
                ]
                for j, score in enumerate(scores)
            ]
            indices = [
                [i for i in index if ignore_indexes[j] != i][:k]
                for j, index in enumerate(indices)
            ]

        return scores, indices

    def forward(
        self,
        queries: np.ndarray,
        aid: list = None,
        aid_counts: torch.Tensor = None,
        target_str: list = None,
        input_str: list = None,
        ignore_indexes: list = None,
        k: int = 10,
    ) -> MipsModelOutput:
        if (
            self.args.memory_forcing == "target_only"
            and self.args.multi_x_science_dataset_mode == "original"
        ):
            flat_texts = target_str
            k = 1
            scores = None
            examples = [target_str]
        else:
            queries = self._prepare_query(query=queries)
            scores, indices = self.search(
                queries=queries,
                ignore_indexes=ignore_indexes,
                k=k,
            )

            examples = [self.embeddings[i][self.text_column] for i in indices]

            if (
                self.args.memory_forcing == "target_in"
                and self.args.multi_x_science_dataset_mode == "original"
                and self.args.copy_forcing > random()
                and isinstance(target_str, list)
            ):
                flat_texts = [
                    t for i, df in enumerate(examples) for t in ([target_str[i]] + df)
                ]
                k += 1
            elif (
                self.args.memory_forcing in ["no_forcing", "retrieved_forcing"]
                and self.args.multi_x_science_dataset_mode == "original"
            ):
                flat_texts = [t for df in examples for t in df]
            elif self.args.multi_x_science_dataset_mode == "dual" and input_str != None:
                input_list = (i.split(self.args.doc_sep)[:k] for i in input_str)
                flat_texts = [
                    j
                    for e, i in zip(examples, input_list)
                    for j in i + e[: (k - len(i))]
                ]
            else:
                flat_texts = [t for df in examples for t in df]
                # assert False, "Candidates are missing, please verify memory_forcing or dataset_mode."

        metrics = None
        if aid is not None and self.args.log_retriever_metrics:
            examples_full = [self.embeddings[i] for i in indices]

            pred = torch.tensor(
                [[b == a for a in e["aid"]] for e, b in zip(examples_full, aid)]
            ).float()
            metrics = retriever_metrics(pred, aid_counts.cpu())

        tokens = self.encoder.tokenize(flat_texts)
        # TODO: Implement DataLoader ?
        mips_outputs = self.encoder(tokens)

        mips_last_hidden_state: torch.Tensor = mips_outputs[0]
        mips_sequence_len = mips_last_hidden_state.shape[1]
        mips_last_hidden_state = mips_last_hidden_state.reshape(
            queries.shape[0],
            k,
            mips_sequence_len,
            -1,
        )

        memory_tokens = self.memory_tokenizer(
            flat_texts,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            **self.memory_tokenizer_kwargs,
        ).to(self.memory_encoder.device)

        # put global attention on <s> token
        global_attention_mask = torch.zeros_like(
            memory_tokens["input_ids"], device=memory_tokens["input_ids"].device
        )
        global_attention_mask[:, 0] = 1

        memory_outputs = self.memory_encoder(
            input_ids=memory_tokens["input_ids"],
            attention_mask=memory_tokens["attention_mask"],
            global_attention_mask=global_attention_mask,
        )

        memory_input_ids = memory_tokens["input_ids"].clone()
        memory_attention_mask = (
            memory_tokens["attention_mask"]
            .masked_fill(
                (memory_input_ids == self.eos_id) | (memory_input_ids == self.bos_id), 0
            )
            .clone()
        )
        # .clone()

        output = MipsModelOutput(
            scores=scores,
            metrics=metrics,
            mips_last_hidden_state=mips_last_hidden_state,
            memory_outputs=memory_outputs,
            memory_input_ids=memory_input_ids,
            memory_attention_mask=memory_attention_mask,
            examples=examples,
            query_cls=queries,
        )

        return output

    def l2_normalization(self, x: np.ndarray) -> np.ndarray:
        if not x.flags.c_contiguous:
            x = np.asarray(x, order="C")
        faiss.normalize_L2(x)
        return x

    def np_search(self, x, k: int = 2) -> tuple:
        y = self.embeddings[self.embeddings_column]
        return inner_product(x, y, k, normalize=self.normalize)

    @rank_zero_only
    def save(self) -> None:
        shutil.rmtree(self.mips_folder, ignore_errors=True)
        self.mips_folder.mkdir(parents=True, exist_ok=True)

        self.embeddings.save_faiss_index(self.index_name, self.index_file)
        self.embeddings.drop_index(self.index_name)
        self.embeddings.save_to_disk(self.embeddings_folder)
        with open(self.max_norm_file, "wb") as f:
            cloudpickle.dump(self.max_norm, f)

        self.embeddings = None
        shutil.rmtree(self.embeddings_tmp_folder, ignore_errors=True)

    def load(self) -> None:
        self.embeddings = Dataset.load_from_disk(self.embeddings_folder)
        self.embeddings.load_faiss_index(self.index_name, self.index_file)
        with open(self.max_norm_file, "rb") as f:
            self.max_norm = cloudpickle.load(f)


def inner_product(x: np.ndarray, y: np.ndarray, k: int = 1, normalize: bool = True):
    assert len(x.shape) == len(y.shape) == 2
    if normalize:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
    scores = x @ y.T
    indices = (-scores).argsort()[:, :k]
    scores = np.array([s[i] for i, s in zip(indices, scores)])
    return scores, indices


def test_mips(mips: Mips, data: Dataset) -> None:
    mips.build_index()

    batch_size, k = 2, 10
    text = mips.data[mips.text_column][:batch_size]
    scores, examples = mips.get_nearest_exemples(text=text, k=k, return_index=True)

    embs = mips.embeddings[mips.embeddings_column][:batch_size]
    # print(mips.embeddings[mips.embeddings_column][:batch_size])
    print("Phi :", mips.max_norm**2)
    print("Indices :", examples)
    # examples_flat = [t for e in examples for t in e[mips.text_column]]
    print("Scores :", scores)

    print(embs.shape)
    scores_ip = embs @ embs.T
    print("Scores IP", scores_ip)

    # examples_flat -> batch_size * k
    # assert len(examples_flat) == batch_size * k
    # assert all((t == examples[i][mips.text_column][0]
    #            for i, t in enumerate(text)))

    # print("Start exhaustive search")
    # emb = mips.embeddings[mips.embeddings_column]
    # x, y = emb[:batch_size], emb
    # x = mips.encoder.encode(mips.encoder.tokenize(text))[
    #     mips.embeddings_column]
    # x = augment_xq(x)
    # scores, examples = inner_product(x, y, k=k, normalize=False)
    # print("Scores :", scores)
    # print(examples)

    string_factory = "Flat"
    # Number of training points should be at least as large as number of clusters
    train_size = 10000
    # metric_type = faiss.METRIC_INNER_PRODUCT
    metric_type = faiss.METRIC_L2
    normalize = True
    tokenizer_kwargs = {"max_length": 1024}

    index_file, embeddings_file, max_norm_file = (
        "test.index",
        "test.embeddings",
        "max_norm.pkl",
    )
    mips.save(index_file, embeddings_file, max_norm_file)
    mips.load_index(index_file)

    new_mips = Mips(
        # model_name=model_name,
        data=data,
        string_factory=string_factory,
        train_size=train_size,
        metric_type=metric_type,
        normalize=normalize,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    new_mips.load(index_file, embeddings_file, max_norm_file)
    print(new_mips.embeddings)

    scores, examples = new_mips.get_nearest_exemples(text=text, k=k)
    print("New MIPS Scores :", scores)
    scores, examples = mips.get_nearest_exemples(text=text, k=k)
    print("MIPS Scores :", scores)


def test_encode(model_name: str, data: Dataset) -> Mips:
    # IVF{k} where k = 4*sqrt(N) to 16*sqrt(N)
    # string_factory = "IVF256,SQ8"
    string_factory = "Flat"
    # Number of training points should be at least as large as number of clusters
    train_size = 10000
    # metric_type = faiss.METRIC_INNER_PRODUCT
    metric_type = faiss.METRIC_L2
    normalize = True
    tokenizer_kwargs = {"max_length": 1024}

    mips = Mips(
        model_name=model_name,
        data=data,
        string_factory=string_factory,
        train_size=train_size,
        metric_type=metric_type,
        normalize=normalize,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    mips.encode_text(num_proc=2, batch_size=32)
    return mips


def test_faiss_index(mips: Mips):
    mips.encode_text(2, 32)
    xb = mips.embeddings[mips.embeddings_column]
    xb_tensor = torch.Tensor(xb)
    xb = _layer_norm(xb_tensor).numpy()
    xq = xb[:2]
    d = xb.shape[1]
    print("Dim :", d)

    k = 10
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    Dref, Iref = index.search(xq, k)

    k = 10
    index = faiss.IndexFlatL2(d + 1)
    phi = get_phi(xb)
    xb_augmented = augment_xb(xb)
    xq_augmented = augment_xq(xq)
    index.add(xb_augmented)
    D, I = index.search(xq_augmented, k)

    print("xb", xb_augmented[:2])
    # print("xq", xq[:2])
    print("Phi", phi)
    print("D Index Flat L2", D)
    print("I indices", I)
    print("Dref Index Flat IP", Dref)
    print("Iref indices", Iref)

    print(np.all(I == Iref))


if __name__ == "__main__":
    import pandas as pd

    from model_config import ModelConfig
    from data_modules import RGMultiXScienceDataModule
    from retriever_generator import RetrieverGenerator, RGEncoderModelOutput
    from retriever_lightning import RetrieverLightning, RetrieverConfig

    metric = 1  # 0 -> Inner Product, 1 -> L2

    # retriever_config = RetrieverConfig(
    #     mips_state_dict="/sps/liris/fbaud/sotasum/pretrainedrefabstract/mips-longformer-large-4096/pytorch_model.bin",
    #     query_state_dict="/sps/liris/fbaud/sotasum/pretrainedrefabstract/query-longformer-large-4096/pytorch_model.bin",
    #     test_full_data=True,
    #     batch_size=32,
    #     validation_batch_size=32,
    #     mips_tok_max_length=512,
    #     dry_run=True,
    #     inner_product=metric == 0,
    #     top_k=2,
    # )
    # retriever = RetrieverLightning(model_config=retriever_config)
    # retriever.setup(None)
    # retriever = retriever.to("cuda:0")
    # retriever.on_validation_start()

    model_config = ModelConfig(
        mips_db_max_size=4096,
        mips_batch_size=16,
        validation_batch_size=2,
        batch_size=2,
        # query_encoder_path="allenai/longformer-large-4096",
        # query_state_dict="/sps/liris/fbaud/sotasum/pretrainedrefabstract/query-longformer-large-4096/pytorch_model.bin",
        # mips_model_name="allenai/longformer-large-4096",
        # mips_state_dict="/sps/liris/fbaud/sotasum/pretrainedrefabstract/mips-longformer-large-4096/pytorch_model.bin",
        query_encoder_path="allenai/specter2_base",
        mips_model_name="allenai/specter2_base",
        # mips_dataset="multi_x_science",
        mips_dataset="arxiv2",
        mips_string_factory="IVF16,Flat",
        mips_arxiv_data_path="/sps/liris/fbaud/data/arxiv_parquet/arxiv.parquet",
        mips_tmp_folder="/sps/liris/fbaud/sotasum/tmp/test",
        source_memory=True,
        mips_tok_max_length=512,
        mips_metric_type=metric,
        mips_normalize=False,
        mips_topk=10,
    )

    data_modules = RGMultiXScienceDataModule(
        model_config=model_config,
        num_proc_tokenization=4,
        query_max_length=512,
        save_path="/pbs/throng/liris/fbaud/sotasum/multixscienceRG.arrow",
    )
    data_modules.prepare_data()
    data_modules.setup(None)

    for batch in data_modules.val_dataloader():
        break

    # retriever_out = retriever.validation_step(
    #     {
    #         "query_input_ids": batch["query_input_ids"].to("cuda:0"),
    #         "query_attention_mask": batch["query_attention_mask"].to("cuda:0"),
    #     },
    #     0,
    # )

    rg = RetrieverGenerator(model_config).to("cuda:0")
    rg.encoder.mips.encode_text2(rank=0, num_rank=1)
    rg.encoder.mips.build_index()

    rg.encoder.mips.embeddings.to_json(f"./kb_from_Mips_{metric}.json")
    # retriever._full_data.to_json(f"./kb_from_Retriever_{metric}.json")

    with torch.inference_mode():
        # rg.encoder.eval()
        out: RGEncoderModelOutput = rg.encoder(
            input_ids=batch["input_ids"].to("cuda:0"),
            attention_mask=batch["attention_mask"].to("cuda:0"),
            query_input_ids=batch["query_input_ids"].to("cuda:0"),
            query_attention_mask=batch["query_attention_mask"].to("cuda:0"),
        )

    print(out)
    # print(retriever_out)

    # out_retriever_df = pd.DataFrame(
    #     {
    #         "query": batch["query_input"],
    #         # "examples": retriever_out["examples"],
    #         "examples": [d["mips_column"] for d in retriever_out["examples"]],
    #         "scores": retriever_out["scores"].tolist(),
    #         "query_cls": retriever_out["query_cls"].tolist(),
    #     }
    # )

    out_mips_df = pd.DataFrame(
        {
            "query": batch["query_input"],
            "examples": out.examples,
            "scores": out.faiss_scores.tolist(),
            "query_cls": out.query_cls.tolist(),
        }
    )

    print(out_mips_df)
    # print(out_retriever_df)

    out_mips_df.to_json(path_or_buf=f"./output_mips_{metric}.json")
    # out_retriever_df.to_json(path_or_buf=f"./output_retriever_{metric}.json")
