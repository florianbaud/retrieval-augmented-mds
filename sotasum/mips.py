import faiss
import torch
import torch.nn as nn
import numpy as np
import multiprocess
import time
import cloudpickle
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd
import rich

from random import random
from data_loaders import load_mips_multi_x_science
from datasets.arrow_dataset import Dataset
from transformers import LongformerTokenizer, LongformerConfig, LongformerModel
from transformers.models.longformer.modeling_longformer import LongformerBaseModelOutputWithPooling


def _layer_norm(x: torch.Tensor, variance_epsilon: float = 1e-12) -> torch.Tensor:
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    x = (x - u) / torch.sqrt(s + variance_epsilon)
    return x

# see http://ulrichpaquet.com/Papers/SpeedUp.pdf theorem 5


def get_phi(xb: np.ndarray):
    return (xb ** 2).sum(1).max()


def augment_xb(xb: np.ndarray, phi=None):
    norms = (xb ** 2).sum(1)
    if phi is None:
        phi = norms.max()
    extracol = np.sqrt(phi - norms)
    extracol = np.where(~np.isnan(extracol), extracol, 0)
    return np.hstack((xb, extracol.reshape(-1, 1)))


def augment_xq(xq: np.ndarray):
    extracol = np.zeros(len(xq), dtype='float32')
    return np.hstack((xq, extracol.reshape(-1, 1)))


def timer(name: str = ''):
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

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model_name = self.args.mips_model_name
        tokenizer_kwargs = {"max_length": self.args.mips_tok_max_length}

        self.tokenizer = LongformerTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
        )

        config = LongformerConfig.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            gradient_checkpointing=self.args.gradient_checkpointing,
        )
        self.model = LongformerModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            config=config,
            # add_pooling_layer=False,
        )

        if self.args.mips_state_dict is not None:
            mips_state_dict = torch.load(self.args.mips_state_dict)
            self.model.load_state_dict(mips_state_dict)

        self.tokenizer_opt = tokenizer_kwargs

    def forward(self, tokens) -> LongformerBaseModelOutputWithPooling:
        input_ids = tokens['input_ids']

        # put global attention on <s> token
        global_attention_mask = torch.zeros_like(input_ids).cuda()
        global_attention_mask[:, 0] = 1

        outputs = self.model(
            input_ids=input_ids,
            global_attention_mask=global_attention_mask
        )
        return outputs

    @torch.inference_mode()
    def encode(self, tokens) -> LongformerBaseModelOutputWithPooling:
        return self(tokens)

    def tokenize(self, text: list) -> dict:
        tokens = self.tokenizer(
            text=text,
            padding="max_length",
            return_tensors='pt',
            truncation=True,
            **self.tokenizer_opt,
        ).to(self.model.device)
        tokens = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
        return tokens


class Mips(nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        if args.mips_dataset == "multi_x_science":
            self.data = load_mips_multi_x_science(
                data_path=args.data_path,
                script_path=self.args.mips_data_script_path,
            )
        else:
            assert False, f"{args.mips_dataset} not found."

        if isinstance(args.mips_db_max_size, int):
            self.data = self.data.select(range(0, args.mips_db_max_size))

        self.encoder = MipsEncoder(self.args)

        self.memory_tokenizer_kwargs = {
            "max_length": self.args.memory_tok_max_length,
        }
        self.memory_tokenizer = LongformerTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.args.memory_model_name,
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

        self.string_factory = self.args.mips_string_factory
        self.train_size = self.args.mips_train_size
        self.metric_type = self.args.mips_metric_type
        self.normalize = self.args.mips_no_normalize

        self.max_norm = None
        self.has_context = False
        self.rebuilt_steps = [0]
        self.load_from_cache_file = True
        self.text_column = "mips_column"
        self.index_column = "aid"
        self.index_name = "mips_embeddings"
        self.embeddings_column = "embeddings"
        self.scale_topk = 16
        self.examples = None

    def encode_text(self, num_proc: int = 1, batch_size: int = 32) -> None:
        self.new_fingerprint = f"{self.args.mips_cache_prefix}-{self.rebuilt_steps[-1]}-{self.args.mips_db_max_size}"
        desc = "Encoding"
        self.eval()
        if num_proc > 1:
            self._init_context()
            self.embeddings = self.data.map(
                self._map_encode,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                with_rank=True,
                new_fingerprint=self.new_fingerprint,
                load_from_cache_file=self.load_from_cache_file,
                desc=desc,
            )
        elif num_proc == 1:
            fn_kwargs = {"rank": 0}
            self.embeddings = self.data.map(
                self._map_encode,
                batched=True,
                batch_size=batch_size,
                fn_kwargs=fn_kwargs,
                new_fingerprint=self.new_fingerprint,
                load_from_cache_file=self.load_from_cache_file,
                desc=desc,
            )
        self.embeddings.set_format(
            type='numpy',
            columns=[self.embeddings_column],
            output_all_columns=True,
        )
        self.train()

    def build_index(self) -> None:
        max_norm_fingerprint = self.new_fingerprint + "-max_norm"
        self.max_norm = self.embeddings.map(
            self._map_norm,
            batched=True,
            desc="Calculating max norm",
            load_from_cache_file=self.load_from_cache_file,
            new_fingerprint=max_norm_fingerprint,
        )['norm'].max()

        if self.normalize and self.metric_type == faiss.METRIC_INNER_PRODUCT:
            self.new_fingerprint += "-normalized"
            self.embeddings = self.embeddings.map(
                self._map_normalize,
                batched=True,
                desc="Normalization",
                load_from_cache_file=self.load_from_cache_file,
                new_fingerprint=self.new_fingerprint,
            )

        if self.metric_type == faiss.METRIC_L2:
            self.new_fingerprint += "-augmented"
            self.embeddings = self.embeddings.map(
                self._map_augment_xb,
                batched=True,
                desc="Augmenting data",
                load_from_cache_file=self.load_from_cache_file,
                new_fingerprint=self.new_fingerprint,
            )

        self.embeddings.add_faiss_index(
            column=self.embeddings_column,
            index_name=self.index_name,
            string_factory=self.string_factory,
            train_size=self.train_size,
            metric_type=self.metric_type,
        )

    def _map_norm(self, x: dict) -> dict:
        norm = np.linalg.norm(x[self.embeddings_column], axis=1, keepdims=True)
        return {"norm": norm}

    def _map_encode(self, x: dict, rank: int) -> dict:
        self.encoder.to(rank)
        tokens = self.encoder.tokenize(x[self.text_column])
        output = self.encoder.encode(tokens)
        output = output.last_hidden_state[:, 0, :].cpu().numpy()
        x[self.embeddings_column] = output.astype(np.float32)
        return x

    def _map_normalize(self, x: dict) -> dict:
        x[self.embeddings_column] = self.l2_normalization(
            x[self.embeddings_column])
        return x

    def _map_augment_xb(self, x: dict) -> dict:
        x[self.embeddings_column] = augment_xb(
            x[self.embeddings_column], phi=self.max_norm**2)
        return x

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
            multiprocess.set_start_method('spawn')
            self.has_context = True

    def search(self, queries: np.ndarray, ignore_indexes: list = None, k: int = 10):
        queries = self._prepare_query(query=queries)

        # sel = None
        # if ignore_indexes is not None:
        #     sel = pc.index_in(
        #         self.data.data.table['aid'],
        #         pa.array(ignore_indexes),
        #     ).drop_null().to_pylist()
        #     sel = faiss.IDSelectorNot(faiss.IDSelectorBatch(sel))

        scores, indices = self.embeddings.get_index(self.index_name).faiss_index.search(
            queries,
            k + 1 if ignore_indexes is not None else k,
            # params=faiss.SearchParameters(
            #     sel=sel,
            # ),
        )

        if ignore_indexes is not None:
            scores = [[s for i, s in enumerate(
                score) if ignore_indexes[j] != indices[j][i]][:k] for j, score in enumerate(scores)]
            indices = [[i for i in index if ignore_indexes[j] != i][:k]
                       for j, index in enumerate(indices)]

        return scores, indices

    def forward(
            self,
            queries: np.ndarray,
            target_str: list = None,
            ignore_indexes: list = None,
            k: int = 10
    ):

        if self.args.memory_forcing == "target_only":
            flat_texts = target_str
            k = 1
            scores = None
        else:
            scores, indices = self.search(
                queries=queries,
                ignore_indexes=ignore_indexes,
                k=k,
            )

            self.examples = [self.embeddings[i][self.text_column]
                             for i in indices]

            if self.args.memory_forcing == "target_in":
                if self.args.copy_forcing > random() and isinstance(target_str, list):
                    flat_texts = [t for i, df in enumerate(
                        self.examples) for t in ([target_str[i]] + df)]
                    k += 1
                else:
                    flat_texts = [t for df in self.examples for t in df]
            elif self.args.memory_forcing == "no_forcing":
                flat_texts = [t for df in self.examples for t in df]

        tokens = self.encoder.tokenize(flat_texts)
        # TODO: Implement DataLoader ?
        mips_outputs = self.encoder(tokens)

        memory_tokens = self.memory_tokenizer(
            flat_texts,
            padding="max_length",
            return_tensors='pt',
            truncation=True,
            **self.memory_tokenizer_kwargs,
        ).to(self.memory_encoder.device)
        memory_input_ids = memory_tokens['input_ids']

        # put global attention on <s> token
        global_attention_mask = torch.zeros_like(memory_input_ids).cuda()
        global_attention_mask[:, 0] = 1

        memory_outputs = self.memory_encoder(
            input_ids=memory_input_ids,
            global_attention_mask=global_attention_mask
        )

        mips_last_hidden_state: torch.Tensor = mips_outputs[0]
        mips_sequence_len = mips_last_hidden_state.shape[1]
        mips_last_hidden_state = mips_last_hidden_state.reshape(
            queries.shape[0],
            k,
            mips_sequence_len,
            -1,
        )

        return scores, mips_last_hidden_state, memory_outputs, memory_tokens

    def l2_normalization(self, x: np.ndarray) -> np.ndarray:
        if not x.flags.c_contiguous:
            x = np.asarray(x, order="C")
        faiss.normalize_L2(x)
        return x

    def np_search(self, x, k: int = 2) -> tuple:
        y = self.embeddings[self.embeddings_column]
        return inner_product(x, y, k, normalize=self.normalize)

    def save(self, index_file: str, embeddings_file: str, max_norm_file: str) -> None:
        self.save_index(index_file=index_file)
        self.save_embeddings(embeddings_file=embeddings_file)
        with open(max_norm_file, "wb") as f:
            cloudpickle.dump(self.max_norm, f)

    def load(self, index_file: str, embeddings_file: str, max_norm_file: str) -> None:
        self.load_embeddings(embeddings_file=embeddings_file)
        self.load_index(index_file=index_file)
        with open(max_norm_file, 'rb') as f:
            self.max_norm = cloudpickle.load(f)

    def save_embeddings(self, embeddings_file: str) -> None:
        self.embeddings.drop_index(self.index_name)
        self.embeddings.save_to_disk(embeddings_file)

    def save_index(self, index_file: str) -> None:
        self.embeddings.save_faiss_index(self.index_name, index_file)

    def load_embeddings(self, embeddings_file: str) -> None:
        self.embeddings = Dataset.load_from_disk(embeddings_file)

    def load_index(self, index_file: str) -> None:
        self.embeddings.load_faiss_index(self.index_name, index_file)


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
    scores, examples = mips.get_nearest_exemples(
        text=text, k=k, return_index=True)

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

    index_file, embeddings_file, max_norm_file = 'test.index', 'test.embeddings', "max_norm.pkl"
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

    mips = Mips(model_name=model_name, data=data, string_factory=string_factory, train_size=train_size,
                metric_type=metric_type, normalize=normalize, tokenizer_kwargs=tokenizer_kwargs)

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
    from main import get_args
    from lightning_model import LongformerLightning
    from retriever_generator import RetrieverGenerator
    # model_name = 'bert-base-uncased'
    # model_name = "allenai/longformer-base-4096"
    # from data import load_mips_multi_x_science
    # data_path = "../data_hf"
    # multixsci = load_mips_multi_x_science(data_path=data_path)
    # multixsci = multixsci.select(range(1024*2))
    # print("Num examples : ", multixsci.shape[0])

    # mips = test_encode(model_name=model_name, data=multixsci)
    # test_mips(mips, data=multixsci)
    # test_faiss_index(mips)
    # test_mips(model_name=model_name, data=multixsci)
    # test_load_from_new('test.index', 'test.embeddings',
    #                    model_name=model_name, data=multixsci)

    args = get_args()
    rich.print(args)

    pl_model = LongformerLightning(
        args=args,
        model=RetrieverGenerator(args=args),
    )
    mips = pl_model.retriever_generator.encoder.mips
    query_encoder = pl_model.retriever_generator.encoder.query_encoder

    mips.encode_text()
    mips.build_index()

    d = pl_model._get_data_loader('train', batch_size=2)

    for batch in d:
        query_global_attention_mask = torch.zeros_like(
            batch['query_input_ids']).cuda()
        query_global_attention_mask[:, 0] = 1

        query_outputs = query_encoder(
            input_ids=batch['query_input_ids'],
            attention_mask=batch['query_attention_mask'],
            global_attention_mask=query_global_attention_mask,
        )

        break

    query = query_outputs.last_hidden_state[:, 0, :].detach().numpy()
    rich.print(query)
    result = mips.search(
        queries=query,
    )

    rich.print(result)
