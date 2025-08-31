from sacred import Experiment

ex = Experiment("MM-ViLT",save_git_info=False)


def _loss_names(d):
    ret = {
        "itm": 1,
        "iam":1,
        "tam":1,
        "mlm": 1,
        "mpp": 1,
        "mppd": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "mmimdb": 0,
        "hatememes": 0,
        "food101": 0,
        "zhongshanyi":0
    }
    ret.update(d)
    return ret


# 规定：
# 考虑kaggle中的布局 项目与模型参数和数据集平级
# 例如 多数据集在本地是在datasets/xxx下 在kaggle中则是以datasets_xxx下平级展现
# 对于其它如模型参数等输入具有相似的规定
datasets_dir = r"C:\Users\ZXM\PycharmProjects_hospital\pythonProject3_hgnn+prompt+boarding\pythonProject\datasets_zhongshanyi\cmu-mosi"
datasets_name = "data"  # "modmis"
dataset_type = "arrows"  # "arrows"
pretrained_dir = "../pretrained"  # "../pretrained"
result_dir = "./results"  # "../results"


@ex.config
def config():
    exp_name = "mm-vilt"
    seed = 0
    # datasets = ["modmis"]  # 硬规定 只能有一个数据集 即不支持多任务并行
    _loss_names = {"itm": 1, "iam":1,"tam":1,"mlm": 1,"mpp": 1}#这个地方不能动  一动val的auc就变nan
    loss_names = {"modmis_bin": 0, "modmis_cls": 1, "modmis_reg": 0}
    loss_weight = {"modmis_bin": 0.0, "modmis_cls": 0.0, "modmis_reg": 0.0}
    # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    batch_size = 4096

    # eval config (for bash execution)
    test_ratio = None
    test_type = None
    test_exp_name = None

    # fix backbone model (ViLT) weights
    fix_model = True

    # cache dir
    cache_root = f"{result_dir}/{datasets_name}_cache/"  # 缓存路径

    # missing modality config
    missing_ratio = {'train': 0.7, 'val': 0.7, 'test': 0.7}  # 0.7 0.7 0.7
    missing_type = {'train': 'text', 'val': 'image', 'test': 'none'}  # ['text', 'image', 'both'] in VL taskss  # both both both
    both_ratio = 0.5   # missing both ratio  # 0.5
    missing_table_root = f"{datasets_dir}/{datasets_name}/missing_tables/"
    simulate_missing = False  # False # 模拟缺失，按1:1:1的几率随机设定缺失类型 与missing_table使用互斥
    
    # Modmis task data using
    modmis_label_used = (0)
    modmis_field_used = ['text', 'vision',"audio",'f_ending']

    # k fold config
    fold_id = -1
    fold_count = -1

    # missing_aware_prompts config
    prompt_type = 'input'
    prompt_length = 16
    learnt_p = True
    prompt_layers = [0, 1, 2, 3, 4, 5]
    multi_layer_prompt = True

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 224  # 384 224 224
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # #Text Setting
    # vqav2_label_size = 3129
    # max_text_len = 40
    # tokenizer = "bert-base-uncased"
    # vocab_size = 30522
    # whole_word_masking = False
    # mlm_prob = 0.15
    # draw_false_text = 0

    # Field Setting
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_field = 0

    input_channels = 6  # 输入通道数，对应于6个特征
    num_filters = 64  # 卷积层的过滤器数量
    kernel_size = 1  # 卷积核大小
    output_size = 738  # 您希望的输出特征维度


    # Transformer Setting
    vit = "vit_base_patch16_224"  # patch32_384 patch16_224 patch32_224_in21k
    hidden_size = 768
    hidden_size_for_boarding=420
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    
    # Loss function Setting
    used_loss = "bcewl"  # "bcewl", "focal"
    bcewl_pos_weight = [1.0,]
    focal_gamma = 2

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 8  # 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False
    label_class_count = [2,]

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    finetune_first = False

    # below params varies with the environment
    # 命令行中的参数设置只有这里
    data_root = f"{datasets_dir}/{datasets_name}/"
    log_dir = result_dir
    pl_log_name = f"{exp_name}_{seed}"
    per_gpu_batchsize = 3  # 每个gpu的batch
    num_gpus = 1  # 每个主机上gpu的数量
    num_nodes = 1  # 主机数量
    vit_model= "vit_base_patch16_224"
    load_path =r"C:\Users\ZXM\PycharmProjects_hospital\pythonProject3_hgnn+prompt+boarding\pythonProject\datasets_zhongshanyi\pretrained\vit\vit_base_p16_224.pth"
    vit_load_path = f"{pretrained_dir}/vit/vit_base_p16_224.pth"  # p32_384 p16_224 p32_224_in21k
    # load_path = "pretrain/vit_base_p32_384.pth"
    num_workers = 0  # multiprocessing多进程设置 包括并发准备数据 建议0或1 大于0都是启用多进程 进程多处理快 但是存占用越大 8个就爆16g
    precision = 16
    save_roc_every_n_epoch = 1  # 记录roc的间隔


# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def env_dandelin():
    data_root = "../data2/dsets/dataset"
    log_dir = "../data2/vilt/result"
    num_gpus = 8
    num_nodes = 1


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_finetune_mmimdb():
    exp_name = "finetune_mmimdb"
    datasets = ["mmimdb"]
    loss_names = _loss_names({"mmimdb": 1})
#     loss_names = _loss_names({"mmimdb": 1, "prompt": -0.5})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
#     optim_type = "adam"
    max_text_len = 1024


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end


@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000


@ex.named_config
def vit32_base():
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12
