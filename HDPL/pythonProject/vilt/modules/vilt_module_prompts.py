# -*- coding = utf-8 -*-
# @Time：2024/3/13 20:19
# @Author：Bin
# @File：vilt_mapm2.py
# @Software：PyCharm

import torch
import datetime
import pytorch_lightning as pl
import pythonProject.vilt.modules.vision_transformer_prompts as vit
from pythonProject.vilt.modules import heads, objectives, vilt_utils
import numpy as np

from models import HGNN
from torch import nn
from transformers import BertConfig

bert_config = BertConfig()
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config, data_info):
        super().__init__()
        self.save_hyperparameters()  # self.hparams.config == config
        self.data_info = data_info

        # ======== 嵌入层、池化层、注意力头
        self.field_embeddings = nn.ModuleDict()
        self.image_embeddings = nn.ModuleDict()
        self.audio_embeddings = nn.ModuleDict()
        # text模态特征提取器
        for f_key, f_len in data_info["texts_len"].items():


            self.field_embeddings[f_key] = heads.FeatureProcessor(in_ch=f_len, hidden_size=config["hidden_size"]).to(device)
            # self.field_embeddings[f_key] = nn.Linear(in_features=f_len, out_features=config["hidden_size"])
            # #Linear(in_features=2, out_features=768, bias=True)
            # self.field_embeddings[f_key].apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(data_info["modal_count"], config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)
        # image模态特征提取器
        for f_key,f_len in data_info["visions_len"].items():
            self.image_embeddings[f_key]=heads.FeatureProcessor(in_ch=f_len, hidden_size=config["hidden_size"]).to(device)
        # audio模态特征提取器
        for f_key,f_len in data_info["audios_len"].items():
            self.audio_embeddings[f_key]=heads.FeatureProcessor(in_ch=f_len, hidden_size=config["hidden_size"]).to(device)


        #vit  vision transformer vilt包括vision  and  language transformer
        # VILT使用了VIT(v_t文件) 而原模态缺失论文直接在VILT上改的(v_t_p文件替代v_t文件 v_m_a_p_m文件替代v_m文件)
        # 这里的loadpath是指VILT参数的loadpath 与VIT参数的loadpath不一样 所以逻辑没问题
        if self.hparams.config["load_path"] != "":
            # 没有VILT的参数 需要载入VIT参数从头训练（pretrained=True）
            self.transformer = getattr(vit, self.hparams.config["vit_model"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            # VILT的参数包括了VIT的参数 只要VIT的结构 无需载入VIT参数
            self.transformer = getattr(vit, self.hparams.config["vit_model"])(
                pretrained=False, config=self.hparams.config
            )
        self.broadnet = heads.broadNet(map_num=10,  # 初始时多少组mapping nodes  调整2或者3都可以  目前1出现最高准确率77% 但不稳定
                                enhance_num=18,  # 初始时多少enhancement nodes 调整2或者3都可以  目前1出现最高准确率77% 但不稳定
                                EPOCH=4,  # 训练多少轮
                                map_function='relu',
                                enhance_function='relu',
                                batchsize=15,  # 每一组的神经元个数
                                DESIRED_ACC=0.81,  # 期望达到的准确率
                                STEP=10  # 一次增加多少组enhancement nodes
                           )

        # self.pooler = heads.Pooler(config["hidden_size"])
        # self.pooler.apply(objectives.init_weights)

        if config["_loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["_loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)
        if config["_loss_names"]["iam"] > 0:
            self.iam_score = heads.IAMHead(config["hidden_size"])
            self.iam_score.apply(objectives.init_weights)
        if config["_loss_names"]["tam"] > 0:
            self.tam_score = heads.TAMHead(config["hidden_size"])
            self.tam_score.apply(objectives.init_weights)
        if config["_loss_names"]["mpp"] > 0:

            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        # ====================针对不同数据集建立分类器（目前只有一个数据集）
        # ======== 二分类头
        # self.hs_timer = 2  # 隐藏层规模倍数 因为vit只能处理3个通道 对于更多通道的影像 需要重复使用vit 该参数本质上是使用vit的次数  # <--
        self.hs_timer = 1  # <-
        if config["loss_names"]["modmis_bin"] > 0:  # 配置文件 config 里面设置是否启用
            self.modmis_binarizer = heads.ModmisBinHead(config['hidden_size_for_boarding'] * self.hs_timer,
                                                         len(config["label_class_count"]))
            self.modmis_binarizer.apply(objectives.init_weights)
        # ======== 多分类头
        if config["loss_names"]["modmis_cls"] > 0:  # 配置文件 config 里面设置是否启用
            self.modmis_classifier = heads.ModmisClsHead(config["hidden_size"] * self.hs_timer,
                                                         config["label_class_count"])
            self.modmis_classifier.apply(objectives.init_weights)
        # ======== 回归头
        if config["loss_names"]["modmis_reg"] > 0:  # 配置文件 config 里面设置是否启用
            self.modmis_regresser = heads.ModmisRegHead(config["hidden_size"] * self.hs_timer,
                                                         len(config["label_class_count"]))
            self.modmis_regresser.apply(objectives.init_weights)

        # ======== 读取VIT的 checkpoint
        if self.hparams.config["load_path"] != "":
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
           # state_dict = ckpt["state_dict"]
            self.load_state_dict(ckpt, strict=False)
            if self.hparams.config["finetune_first"]:
                print("use pre-finetune model")

        # ======== 配置的参数
        self.prompt_type = self.hparams.config["prompt_type"]
        prompt_length = self.hparams.config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams.config["hidden_size"]
        self.learnt_p = self.hparams.config["learnt_p"]
        self.prompt_layers = self.hparams.config["prompt_layers"]
        self.multi_layer_prompt = self.hparams.config["multi_layer_prompt"]
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1

        # ======== 各种缺失情况下的提示
        sit_ct = -1
        missing_aware_prompt = nn.ParameterDict()
        for f_key in ['none']+ data_info["total_type_keys"]:
            sit_ct += 1
            missing_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
            missing_prompt[:, sit_ct:sit_ct + 1, :].fill_(1)
            if self.learnt_p and self.prompt_type == 'attention':
                missing_prompt[:, prompt_length // 2 + sit_ct:prompt_length // 2 + sit_ct + 1, :].fill_(1)
            missing_aware_prompt[f_key] = nn.Parameter(missing_prompt)
        self.missing_field_prompt = missing_aware_prompt
        # ===== 是否进行提示学习
        if not self.learnt_p:
            for missing_prompt in self.missing_field_prompt.values():
                missing_prompt.requires_grad = False


        #1.迁移学习(不更新预训练模型，仅训练下游任务)
        #2.微调部分模块，节省显存和训练时间
        #3.保持某些模块稳定
        for param in self.transformer.parameters():
                param.requires_grad = False
        for f_embed in self.field_embeddings.values():
                for param in f_embed.parameters():
                 param.requires_grad = False
        for param in self.token_type_embeddings.parameters():
                param.requires_grad = False

        vilt_utils.set_metrics(self)  # 设置评估的指标
        self.current_tasks = list()
        self.records = {}

       # 其它属性设置
        self.best_train_roc = []  # 训练时最好的roc信息
        self.best_val_roc = []  # 验证时最好的roc信息
        self.bcewl_pos_weight = torch.tensor(config["bcewl_pos_weight"], device="cuda")  # 目标函数中采用posweight时的权重设置

    def infer(  # 重点修改的地方
            self,
            batch,
            mask_text=True,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
            is_train=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            img_keys = f"image_{image_token_type_idx - 1}"
        else:
            img_keys = self.data_info["vision_keys"]

        token_type = -1
        #  text掩码处理
        do_mlm = "_mlm" if mask_text else ""
        fields_list, field_masks_list, field_embeds_list = [], [], []
        for f_key in self.data_info["text_keys"]:
            token_type += 1

            fields_list.append(batch[f"{f_key}{do_mlm}"])
            field_masks_list.append(batch[f"{f_key}_masks{do_mlm}"])
            field_embeds_list.append(self.field_embeddings[f_key](fields_list[-1]).unsqueeze(1) + \
                       self.token_type_embeddings(torch.full_like(field_masks_list[-1], token_type, dtype=torch.int32)))

        # prompt产生
        # instance wise missing aware prompts
        prompts = torch.tensor([], device=self.device)
        for missing_tid in batch["missing_type"]:  # 这里是transforms.utils中得到的[image_tensor]?
            prompt = self.missing_field_prompt[self.data_info["missing_type"][missing_tid]]
           #"missing_type": ["none", "modal_1", "modal_2"],
            if prompt.size(0) != 1:
                prompt = prompt.unsqueeze(0)
            prompts = torch.cat([prompts, prompt], dim=0)
        if self.learnt_p:
            if self.prompt_type == 'attention':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length // 2, dtype=prompts.dtype,
                                          device=prompts.device).long()
            elif self.prompt_type == 'input':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length * len(self.prompt_layers),
                                          dtype=prompts.dtype, device=prompts.device).long()
        else:
            prompt_masks = torch.ones(prompts.shape[0], self.prompt_length, dtype=prompts.dtype,
                                      device=prompts.device).long()
        if self.prompt_type == 'input':
            total_prompt_len = len(self.prompt_layers) * prompts.shape[-2]
        elif self.prompt_type == 'attention':
            total_prompt_len = prompts.shape[-2]

        # image模态处理  vit推断 获取特征和掩码
        out_feats, image_labels, patch_index = None, None, None
        image_masks_list = []
        image_embeds_list = []
        for img_key in self.data_info["vision_keys"]:
            token_type += 1
            image_embeds_cat = []
            image_mask_cat = []
            m_image, m_mask, m_norm = batch[img_key]
            m_normed_image = None if m_norm is None else m_norm(m_image)
            # eff_cn = torch.where(m_mask.sum(dim=(0, 2, 3)) > 0)[0][:8] - 3
            # eff_cn = torch.where(m_mask.sum(dim=(0, 2, 3)) > 0)[0][:6]
            # eff_cn = range(15, m_image.shape[1]-3, 3)
            eff_cn = torch.where(m_mask.sum(dim=(0, 2, 3)) > 0)[0][0]
            eff_cn = range(eff_cn, eff_cn+6, 3)
            for cni in eff_cn:
                (image_embeds, image_masks, patch_index, image_labels,) = \
                    self.transformer.visual_embed(
                        # m_image[:, cni, :, :].unsqueeze(1).repeat(1, 3, 1, 1),
                        m_image[:, cni:cni+3, :, :],
                        max_image_len=self.hparams.config["max_image_len"],
                        mask_it=mask_image,
                        # _mask=None if m_mask is None else m_mask[:, cni, :, :].unsqueeze(1),
                        # normalized_x=None if m_norm is None else m_normed_image[:, cni, :, :].unsqueeze(1).repeat(1, 3, 1, 1),
                        normalized_x=None if m_norm is None else m_normed_image[:, cni:cni+3, :, :],
                    )
                image_embeds_cat.append(image_embeds)
                image_mask_cat.append(image_masks)
            image_mask_cat = torch.cat(image_mask_cat, dim=1)
            image_embeds_cat = torch.cat(image_embeds_cat, dim=1) + self.token_type_embeddings(torch.full_like(image_mask_cat, token_type))
            image_masks_list.append(image_mask_cat)
            image_embeds_list.append(image_embeds_cat)



        audio_masks_list, audio_embeds_list = [], []
        for f_key in self.data_info["audio_keys"]:
          token_type += 1
          max_len = self.audio_embeddings[f_key].size(1)
          audio_masks_list = torch.arange(max_len).expand(self.audio_embeddings[f_key].shape[0],
                                                          max_len) < torch.tensor(self.audio_embeddings[f_key].shape[0]).unsqueeze(1)
          audio_embeds_list.append(self.audio_embeddings[f_key](fields_list[-1]).unsqueeze(1) + \
                                   self.token_type_embeddings(
                                       torch.full_like(audio_masks_list[-1], token_type, dtype=torch.int32)))

        criterion = heads.InfoNCE_ThreeModal(temperature=0.07).cuda()
        loss, t_en, v_en, i_en = criterion(field_embeds_list, audio_embeds_list, image_embeds_list)
        # vit推断
        # co_masks = torch.cat([prompt_masks] + field_masks_list , dim=1)
        # x = torch.cat(field_embeds_list , dim=1).detach()
        co_masks = torch.cat([prompt_masks] + field_masks_list + image_masks_list + audio_masks_list, dim=1)
        x = torch.cat(t_en + i_en + v_en, dim=1).detach()
        for i, blk in enumerate(self.transformer.blocks):
            if i in self.prompt_layers:
                if self.multi_layer_prompt:
                    x, _attn = blk(x, mask=co_masks,
                                    prompts=prompts[:, self.prompt_layers.index(i)],
                                    learnt_p=self.learnt_p,
                                    prompt_type=self.prompt_type)
                else:
                    x, _attn = blk(x, mask=co_masks, prompts=prompts, learnt_p=self.learnt_p)
            else:
                x, _attn = blk(x, mask=co_masks)
        x = self.transformer.norm(x)

        # # 获取特征
        # # 以下这两个东西太大了(指内存) 而且目前在训练的时候用不上 先注释掉
        # field_feats, image_feats = (
        #     x[:, total_prompt_len: total_prompt_len + field_embeds.shape[1]],
        #     x[:, total_prompt_len + field_embeds.shape[1]:],
        # )
        if self.prompt_type == 'input':
            # cls_feats = self.pooler(x[:,:prompts.size(1)].mean(dim=1,keepdim=True))
            # out_feats=self.pooler(x[:, total_prompt_len:total_prompt_len + 1])
            print(f"data shape into pooler:",(x[:, total_prompt_len:total_prompt_len + 1]).shape)  #88,1,768



            # starttime = datetime.datetime.now()
            out_feats=self.broadnet.fit(x[:, total_prompt_len:total_prompt_len + 1], torch.stack(batch['label']))
            # endtime = datetime.datetime.now()

        #total_prompt_len 96
        elif self.prompt_type == 'attention':
            # out_feats.append(self.pooler(x))
            out_feats = self.pooler(x)

        # out_feats = torch.cat(out_feats, dim=-1)
        # out_feats = torch.stack(out_feats).mean(dim=0)
        ret = {
            # "field_feats": field_feats,  # text_features替换成字段特征，并以此类推多种字段特征
            # "image_feats": image_feats,
            "out_feats": (out_feats.to('cuda'))
,
            # "raw_cls_feats": x[:, 0],
            # "image_labels": image_labels,
            # "image_masks": image_masks,
            # "field_labels": field_labels,
            # "field_ids": field_ids,
            # "field_masks": field_masks,
            # "patch_index": patch_index,
        }

        return ret

    def forward(self, batch):
        if len(self.current_tasks) == 0:  # 没有设定任务，只推断，推断不包含任何loss计算
            return self.infer(batch)

        ret = dict()
        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Multi-label classification for MODMIS
        if "modmis_bin" in self.current_tasks:  # 训练并更新结果指标
            ret.update(objectives.compute_modmis_bin(self, batch))

        # Multi-label-multi-class classification for MODMIS
        if "modmis_cls" in self.current_tasks:

            ret.update(objectives.compute_modmis_cls(self, batch))

        # Multi-label-multi-class regression for MODMIS
        if "modmis_reg" in self.current_tasks:
            ret.update(objectives.compute_modmis_reg(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        torch.cuda.empty_cache()

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        torch.cuda.empty_cache()

        return total_loss

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)
        torch.cuda.empty_cache()
        # print('missing_img:', self.missing_img_prompt[0, 0:3, 0:8])
        # print('missing_text:', self.missing_field_prompt[0, 0:3, 0:8])
        # print('complete:', self.complete_prompt[0, 0:3, 0:8])

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict(
            total_loss = sum([v for k, v in output.items() if "loss" in k])
        )

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
