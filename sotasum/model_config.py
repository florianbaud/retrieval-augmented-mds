from dataclasses import dataclass


@dataclass
class ModelConfig:
    log_retriever_metrics: bool = False
    log_copy_metrics: bool = False
    validation_batch_size: int = 16
    validation_outputs_dir: str = './outputs/'
    num_beams: int = 2
    use_cache: bool = False
    rouge_path: str = 'rouge'
    generate_max_length: int = 1024
    generate_min_length: int = 0
    generate_no_repeat_ngram_size: int = 0
    generate_length_penalty: float = 1.0  # Training args
    lr: float = 3e-5
    batch_size: int = 16
    # LongFormer pretraining settings : polynomial decay, lr=3e_5, warmup=500
    # PRIMERA pretraining settings : linear decay, lr=3e_5, warmup=10 000, total_step=100 000 # Models args
    warmup_steps: int = 1000
    total_steps: int = 5000
    model_name: str = None
    copy_decoder_layers: int = 8
    model_cache_dir: str = None
    join_method: str = "concat_start_wdoc_global"
    attention_mode: str = "sliding_chunks"
    query_encoder_path: str = "allenai/longformer_large_4096"
    query_state_dict: str = None
    model_tok_max_length: int = None
    decoder_max_length: int = None
    query_tok_max_length: int = None
    label_smoothing_eps: float = 0.1
    use_own_decoder: bool = False
    gradient_checkpointing: bool = False
    output_copy_probs: bool = False
    gates_mode: str = "nmt"  # choices = ['nmt', 'onlycopy', 'both']
    skip_residual: bool = False
    # choices=['target_only', 'target_in', "no_forcing", "retrieved_forcing"]
    memory_forcing: str = "no_forcing"
    use_attention_mask: bool = False  # Memory args
    memory_model_name: str = "allenai/longformer_large_4096"
    memory_tok_max_length: int = None  # MIPS args
    mips_disabled: bool = False
    mips_freezed: bool = False
    mips_encoder_freezed: bool = False
    mips_batch_size: int = 32
    mips_num_gpus: int = 0
    mips_topk: int = 2
    mips_string_factory: str = "IVF256,SQ8"
    mips_nprobe: int = None
    mips_rebuild_every: int = 10000
    mips_train_size: int = -1
    mips_metric_type: int = 0  # help = "Choose between : 0 _> INNER_PRODUCT ; 1 _> L2"
    # mips_normalize:bool =True
    mips_no_normalize = True
    # choices=["multi_x_science", "arxiv"]
    mips_dataset: str = "multi_x_science"
    mips_arxiv_data_path: str = None
    mips_data_script_path: str = "multi_x_science_sum"
    mips_model_name: str = "allenai/longformer_large_4096"
    mips_state_dict: str = None
    # mips_save_path:str =None
    # mips_load_path:str =None
    mips_no_init_build: bool = False
    mips_db_max_size: int = None
    mips_tok_max_length: int = None
    mips_tmp_max_norm_file: str = "max_norm.pkl"
    mips_tmp_index_file: str = "index.faiss"
    mips_tmp_embeddings_folder: str = "embeddings"
    mips_tmp_folder: str = './tmp'
    mips_cache_prefix: str = ""  # Data args
    dataset_name: str = "multi_x_science"  # choices = ['multi_x_science']
    # choices = ['dual', 'original', 'aggregated']
    multi_x_science_dataset_mode: str = 'original'
    data_script_path: str = "multi_x_science_sum"
    data_path: str = '../data_hf'
    doc_sep: str = "<DOC_SEP>"
    data_workers: int = 8
    clean_cache: bool = False
    copy_forcing: float = 0.0
    source_memory: bool = False
