import random

import torch
import os
import pickle
import json
import zlib


class MODMISDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: r'C:\Users\ZXM\PycharmProjects_hospital\pythonProject3_hgnn+prompt+boarding\datasets_zhongshanyi\data_mosi',
            transform_keys: list,
            image_size: int,
            field_column_name_list: ["text","vision","audio"],
            draw_false_image=0,
            draw_false_field=0,
            draw_false_audio=0,
            image_only=False,
            split="",
            missing_info={},
            used_labels=(0)
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        # assert split in ["train", "val", "test"]
        assert split in ["train", "test"]
        self.split = split

        if split == "train":
            names = ["modmis_train"]
        elif split == "test":
            names = ["modmis_test"]
        # else:
        #     names = ["modmis_val"]

        assert len(transform_keys) >= 1
        super().__init__()

        self.image_size = image_size
        self.field_column_name_list = field_column_name_list
        self.names = names
        self.draw_false_image = draw_false_image
        self.draw_false_field = draw_false_field
        self.image_only = image_only

        self.data_dir = data_dir
        self.used_id = []
        self.phase_data_dict = []  # f_ending modal_1 modal_2

        self.simulate_missing = missing_info['simulate_missing']  # False
        self.missing_info = dict(
            missing_ratio=missing_info['ratio'][split],  # 0.7
            mratio=str(missing_info['ratio'][split]).replace('.', ''),  # 0.7 -> 07
            missing_type=missing_info['type'][split],  # both
            both_ratio=missing_info['both_ratio'],  # 0.5
            missing_table_root=missing_info['missing_table_root'],
        )

        # 读取文件
        with open(f"{data_dir}/split.json", "r", encoding='utf-8') as fp:
            used_id = json.load(fp)[split]
        file_kstr = ["text", "vision","audio", "f_ending"]
        phase_data_dict = {}
        for fname in os.listdir(data_dir):  # modal_1 modal_2
            if split in fname:
                for kstr in file_kstr:
                    if kstr in fname:
                        with open(f"{data_dir}\{fname}", 'rb') as handle:
                            phase_data_dict[kstr] = pickle.load(handle)
        # with open(f"{data_dir}/mean_std.pkl", 'rb') as handle:
        #     mean_std_dict = pickle.load(handle)
        #     for kstr, vdic in mean_std_dict.items():
        #         mean_std_dict[kstr] = vdic["normalizer"]
        # # 去除图像的label 节省空间（当前尚未用到）
        # for sid in used_id:
        #     for kstr in ("T1_image", "T1C_image", "T2_image"):
        #         phase_data_dict[kstr][sid] = (phase_data_dict[kstr][sid][0], None)


        # for kstr in field_column_name_list:
        #     phase_data_dict.pop(kstr, None)

        # # 调整ending字段
        # for sid in used_id:
        #     phase_data_dict["f_ending"][sid] = phase_data_dict["f_ending"][sid][used_labels]

        self.used_id = used_id  # [sid, ...]
        self.phase_data_dict = phase_data_dict  # {kstr: {sid: tensor, ...}, ...}  #f_ending modal_1 modal_2
        # self.mean_std_dict = mean_std_dict  # {kstr: {ck: transforms.Normalize, ...}, ...}
        self.missing_table = self.get_missing_table()  # tensor(len,)

    def get_data(self, index_list: list):
        ret_sids = [self.used_id[ind] for ind in index_list]
        ret_sdata = {}
        for kstr, vdic in self.phase_data_dict.items():
            ret_sdata[kstr] = {sid: vdic[sid] for sid in ret_sids}
        return ret_sids, ret_sdata

    def set_data(self, new_data: tuple[list, dict]):
        self.used_id, self.phase_data_dict = new_data
        self.missing_table = self.get_missing_table()

    @classmethod
    def merge_data(cls, data_list: list[tuple[list, dict]]):
        sids_list = [item[0] for item in data_list]
        sdata_list = [item[1] for item in data_list]
        ret_sid = []
        for sid_item in sids_list:
            ret_sid += sid_item
        ret_sdata = {}
        for kstr in sdata_list[0].keys():
            ret_sdata[kstr] = {}
            for sdata_item in sdata_list:
                ret_sdata[kstr].update(sdata_item[kstr])
        return ret_sid, ret_sdata

    @classmethod
    def split_data(cls, data: tuple[list, dict], split_index_list: list[list]):
        sids, sdata = data
        ret_data_list = []
        for split_index in split_index_list:
            sids_item = [sids[ind] for ind in split_index]
            sdata_item = {}
            for kstr, vdic in sdata.items():
                sdata_item[kstr] = {sid: vdic[sid] for sid in sids_item}
            ret_data_list.append((sids_item, sdata_item))
        return ret_data_list

    def get_missing_table(self):
        missing_table_root = self.missing_info["missing_table_root"]
        missing_type = self.missing_info["missing_type"]
        missing_ratio = self.missing_info["missing_ratio"]

        # 设置缺失比率
        missing_table_name, missing_table_path = "", ""
        if missing_table_root is not None:
            missing_table_name = f'{self.names[0]}_missing_{missing_type}_{self.missing_info["mratio"]}.pt'
            # "./datasets/missing_tables/modmis_missing_both_07.pt"
            missing_table_path = os.path.join(missing_table_root, missing_table_name)

        # 通过缺失的比率设置 missing_table
        # missing_table是缺失状态tensor 0代表无缺失 1代表缺字段 2代表缺图像
        total_num = len(self.used_id)

        if missing_table_root is not None and os.path.exists(missing_table_path):
            missing_table = torch.load(missing_table_path)
            if len(missing_table) != total_num:
                print('missing table mismatched!')
                exit()
        else:
            missing_table = torch.zeros(total_num, dtype=torch.int8)

            if missing_ratio > 0:
                missing_index = random.sample(range(total_num), int(total_num * missing_ratio))
                # 下面应该是对四种情况，标记确实状态
                if missing_type == 'none':
                    missing_table[missing_index] = 0
                elif missing_type == 'text':
                    missing_table[missing_index] = 1
                elif missing_type == 'vision':
                    missing_table[missing_index] = 2
                elif missing_type == 'audio':
                    missing_table[missing_index] = 3

                elif missing_type == 'all':

                    missing_table[missing_index] = 1
                    missing_index_both = random.sample(missing_index,
                                                       int(len(missing_index) * self.missing_info["both_ratio"]))
                    missing_table[missing_index_both] = 2

                if missing_table_root is not None:
                    cache_root = self.missing_info['cache_root']
                    missing_table_path = os.path.join(cache_root, missing_table_name)
                    os.makedirs(cache_root, exist_ok=True)
                    torch.save(missing_table, missing_table_path)
        return missing_table

    def get_info(self):
        data_info = {
            "missing_type": ["none", "text", "vision", "audio"],  # 对应missing_table中的设置
            "total_type_keys":["text","vision","audio"],
            "text_keys": ["text"],  #
            "vision_keys": ["vision"],  # 图像模态
            "audio_keys":["audio"]

        }
        data_info["texts_len"] = {  # 2
            k: len(self.phase_data_dict[k][self.used_id[0]]) for k in data_info["text_keys"]}if self.used_id else{}
        data_info["visions_len"]={
            k: len(self.phase_data_dict[k][self.used_id[0]]) for k in data_info["vision_keys"]} if self.used_id else {}
        data_info["audios_len"] = {
            k: len(self.phase_data_dict[k][self.used_id[0]]) for k in data_info["audio_keys"]} if self.used_id else {}

        # 不知道哪里要用这个东西
        data_info["modal_count"] =3  # 2
        return data_info

    def __len__(self):
        return len(self.used_id)


    def get_image(self, index, image_key="vision"):
        # return {
        #     image_key: self.phase_data_dict[image_key][self.used_id[index]],
        #     "raw_index": index,
        # }
        return self.phase_data_dict[image_key][self.used_id[index]]
    def get_false_image(self, rep, image_key="vision"):
        return {
            f"false_{image_key}_{rep}": self.get_raw_image_label(random.randint(0, len(self) - 1), image_key)[0]
        }
    # def get_raw_field(self, index, text_key="text"):
    #     # return self.phase_data_dict[field_key][self.used_id[index]]
    #     # 从字典中获取原始值
    #     value = self.phase_data_dict[text_key][self.used_id[index]]
    #
    #     # 检查值是否为字符串，并尝试转换为浮点数
    #     if isinstance(value, str):
    #         try:
    #             # 尝试将字符串转换为浮点数
    #             value = float(value)
    #         except ValueError as e:
    #             # 如果转换失败，打印错误并返回一个默认张量
    #             print(f"Cannot convert '{value}' to a real number. Error: {e}")
    #             return torch.tensor([], dtype=torch.float32)
    #
    #     # 检查值是否为列表或元组，如果是，则将每个元素转换为张量
    #     if isinstance(value, (list, tuple)):
    #         value = [float(x) if isinstance(x, str) else x for x in value]
    #         tensor_value = torch.tensor(value, dtype=torch.float32)
    #     else:
    #         # 值已经是数值类型，直接转换为张量
    #         tensor_value = torch.tensor(value, dtype=torch.float32)
    #
    #     return tensor_value

    def get_field(self, index, text_key="text"):
        # return {
        #     text_key: self.phase_data_dict[text_key][self.used_id[index]],
        #     "raw_index": index,
        # }
        return self.phase_data_dict[text_key][self.used_id[index]]
    def get_false_field(self, rep, text_key="text"):
        return {
            f"false_{text_key}_{rep}": self.phase_data_dict[text_key][self.used_id[random.randint(0, len(self) - 1)]]}




    def get_label(self, index, label_key="f_ending"):
        # print(self.phase_data_dict['audio'].shape)
        # print(self.phase_data_dict['vision'].shape)
        # print(self.phase_data_dict['text'].shape)
        # print(self.phase_data_dict['f_ending'].shape)
        return self.phase_data_dict[label_key][self.used_id[index]]
    def get_audio(self, index, audio_key="audio"):

        # return {
        #     audio_key: self.phase_data_dict[audio_key][self.used_id[index]],
        #
        #     "raw_index": index,
        #}
        return self.phase_data_dict[audio_key][self.used_id[index]]
    def get_false_audio(self, rep, audio_key="audio"):
        import random
        return {
            f"false_{audio_key}_{rep}": self.get_raw_audio_label(random.randint(0, len(self) - 1), audio_key)
        }

    def get_suite(self, index):
        result = False
        while result is not True:
            try:
                ret = dict()
                ret.update(self.get_image(index))
                if not self.image_only:
                    for key in self.field_column_name_list:
                        if key=="audio":
                            ret.update(self.get_audio(index,audio_key=key))
                        else:
                            ret.update(self.get_field(index,key))

                # for i in range(self.draw_false_image):
                #     ret.update(self.get_false_image(i,key))
                # for i in range(self.draw_false_field):
                #     ret.update(self.get_false_field(i, key))
                # for i in range(self.draw_false_audio):
                #     ret.update(self.get_false_audio(i,key))
                # result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.names[0]} -> {e}")
                index = random.randint(0, len(self.table) - 1)
        return ret

    # 模拟源程序中的行为，主要获取input_ids和attention_mask
    # input_ids使用
    # https://blog.csdn.net/weixin_44219178/article/details/121991171
    def collate(self, batch):
        batch_size = len(batch)
        dict_batch = {k: [dic[k] for dic in batch] for k in batch[0].keys()}  # 按样本行排改成按关键字列排
        # print(f'dict_batch:',dict_batch)
        # ===== image处理部分
        # 含有'image'字符子串的所有关键字（原本只有'image'一个），原意为取出所有image样本
        # img_keys = [k for k in dict_batch.keys() if "vision" in k]
        # images=dict_batch['vision']
        # image_labels = dict_batch['f_ending']
        # for image in images:
        #
        #     new_images = torch.zeros(batch_size, *[max([img.shape[i] for img in images]) for i in range(3)])  # 改为全1
        #     new_image_labels = None if image_labels is None else torch.zeros(new_images.shape, dtype=torch.uint8)
        #     for bi in range(batch_size):
        #         orig = images[bi]
        #         new_images[bi, :orig.shape[0], :orig.shape[1], :orig.shape[2]] = orig
        #         if new_image_labels is not None:
        #             orig = image_labels[bi]
        #             new_image_labels[bi, :orig.shape[0], :orig.shape[1], :orig.shape[2]] = orig
        #     dict_batch[img_key] = (new_images, new_image_labels,
        #                            self.mean_std_dict[img_key][new_images.shape[1]])


        # # ===== audio处理部分
        # audio_keys = [k for k in dict_batch.keys() if "audio" in k]
        # for audio_key in audio_keys:
        #     audio_tensor = dict_batch[audio_key]
        #
        #
        #     # 原逻辑可能只处理 2 维，现在改 3 维适配
        #     max_len = max([a[0].shape for a in audio_tensor])  # 取时间步维度的最大值
        #     feat_dim = audio_tensor[0].shape[2]  # 特征维度
        #     # padded_audio 改为 4 维：[batch_size, 分块数, max_time, feat_dim]
        #     padded_audio = torch.zeros(batch_size, audio_tensor[0].shape[0], max_len, feat_dim)
        #
        #     for i, audio in enumerate(audio_tensor):
        #         # 填充时间步维度，分块数保持不变
        #         padded_audio[i, :, :audio.shape[1], :] = audio
        #
        #     dict_batch[audio_key] = padded_audio

         # ===== text处理部分
        # text_key = ["text"]
        #     # encodings to mlms ... (need to be implemented)
        #     # fields = [[d[0] for d in dict_batch[f_key]] for f_key in field_keys]
        #     # encodings = [[d[1] for d in dict_batch[f_key]] for f_key in field_keys]
        # for f_key in text_key:
        mlm_ids, mlm_lables = (None, None)  # need to be implemented according to situation
        #     texts = dict_batch[f_key]
        #
        #     max_len = max(len(t[0]) for t in texts)  # 找到最长序列长度
        #     padded_texts = []
        #     for t in texts:
        #         pad_length = max_len - len(t[0])  # 注意这里也是 len(t[0])
        #         padded = torch.cat([t[0], torch.zeros(pad_length, dtype=t[0].dtype)])
        #         padded_texts.append(padded)
        #
        #     # 转换为批量张量（shape: [batch_size, max_len]）
        #     texts_tensor = torch.stack(padded_texts)
        f_key='text'

        # 假设 text 是你的文本序列（可以是列表或张量）

        # 从每个样本中提取第一个tuple元素
        text_tensor= [sample[0] for sample in dict_batch['text']]
        # 转换为张量（确保是张量格式）
        # 提取所有样本的第一个tuple
        # 遍历列表的索引和元素
        for idx, i in enumerate(text_tensor):
            # 检查并修复维度异常的元素
            if len(i.shape) == 1:  # 如果是一维(50,)
                # 扩展为(50, 300)，用0填充
                fixed_tensor = torch.zeros(50, 300, dtype=i.dtype)
                # 通过索引修改原列表中的元素
                text_tensor[idx] = fixed_tensor
        # 遍历所有样本，打印形状异常的索引和形状
        # for i, sample in enumerate(text_tensor):
        #     if sample.shape != (50, 300):
        #         print(f"样本 {i} 形状异常: {sample.shape}")


        # 修复所有形状异常的样本


        # 现在可以安全堆叠了
        stacked_tensor = torch.stack(text_tensor)
        attention_mask = (stacked_tensor != 0).float()
        # 生成掩码：非0位置为1.0，0位置为0.0
        # attention_mask = torch.where(text_tensor != 0, torch.tensor(1.0), torch.tensor(0.0))
        # attention_mask = (torch.stack(text_tensor) != 0).float()
       # attention_mask = (texts != 0).float()  # 转换为 float 类型（模型通常期望浮点型）


        dict_batch[f_key] = 'texts'
        dict_batch[f"{f_key}_labels"] = torch.full_like(stacked_tensor, -100)
        dict_batch[f"{f_key}_masks"] = attention_mask
        dict_batch[f"{f_key}_ids_mlm"] = mlm_ids
        dict_batch[f"{f_key}_labels_mlm"] =mlm_lables




        return dict_batch

    def __getitem__(self, index):
        # print('modmis_dataset.__getitem__ index: ', index)
        # For the case of training with modality-complete data
        # Simulate missing modality with random assign the missing type of samples
        # 当前是模拟缺失的情况，且当前的index样本设置的无缺失，则模拟缺失类型随机挑选0/1/2
        simulate_missing_type = 0
        if self.split == 'train' and self.simulate_missing and self.missing_table[index] == 0:
            simulate_missing_type = random.choice([0, 1, 2, 3])

        # image no missing
        vision = self.get_image(index, "vision")
        # t1c_image_label = self.get_raw_image_label(index, "T1C_image")
        # t2_image_label = self.get_raw_image_label(index, "T2_image")
        audio = self.get_audio(index, "audio")

        # missing field, dummy field is all-zero array
        text = self.get_field(index, "text")
        # field_2 = self.get_raw_field(index, "field_2")
        # field_3 = self.get_raw_field(index, "field_3")
        # field_4 = self.get_raw_field(index, "field_4")
        # field_5 = self.get_raw_field(index, "field_5")
        # field_6 = self.get_raw_field(index, "field_6")
        labels= self.get_label(index, "f_ending")

        if self.missing_table[index] == 1 or simulate_missing_type == 1:
            text = torch.zeros(len(text))
        if self.missing_table[index] == 2 or simulate_missing_type == 2:
            vision= torch.zeros(len(vision))
        if self.missing_table[index] == 3 or simulate_missing_type == 3:
            audio = torch.zeros(len(audio))

        # 解压缩二进制序列 时间换空间 大概每个epoch增加1分钟 空间是原来的1/4
        return {

            "text": (torch.tensor(text), torch.ones_like(torch.tensor(text))),
            "vision": (torch.tensor(vision), torch.ones_like(torch.tensor(vision))),
            "audio":(torch.tensor(audio), torch.ones_like(torch.tensor(audio))),
            # "field_3": (field_3, torch.ones_like(field_3)),
            # "field_4": (field_4, torch.ones_like(field_4)),
            # "field_5": (field_5, torch.ones_like(field_5)),
            # "field_6": (field_6, torch.ones_like(field_6)),
            "labels": labels,
            "missing_type": self.missing_table[index].item() + simulate_missing_type,
        }
