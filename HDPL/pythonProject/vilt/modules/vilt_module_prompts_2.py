# -*- coding = utf-8 -*-
# @Time：2024/3/13 20:19
# @Author：Bin
# @File：vilt_mapm2.py
# @Software：PyCharm

import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer_prompts as vit
from vilt.modules import heads, objectives, vilt_utils
import torch.nn.functional as F


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config, data_info):
        super().__init__()
        self.save_hyperparameters()  # self.hparams.config == config
        self.data_info = data_info

        # ======== 嵌入层、池化层、注意力头
        self.field_embeddings = nn.ModuleDict()


        for f_key, f_len in data_info["fields_len"].items():

            # self.field_embeddings[f_key] = nn.Linear(in_features=f_len, out_features=config["hidden_size"])
            self.field_embeddings[f_key] = heads.CNN1D(input_channels=f_len, num_filters=config["num_filters"],kernel_size=config["kernel_size"]\
                                                       ,output_size=config["hidden_size"])

            self.field_embeddings[f_key].apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(data_info["modal_count"], config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)
        #vit  vision transformer vilt包括vision  and  language transformer
        # VILT使用了VIT(v_t文件) 而原模态缺失论文直接在VILT上改的(v_t_p文件替代v_t文件 v_m_a_p_m文件替代v_m文件)
        # 这里的loadpath是指VILT参数的loadpath 与VIT参数的loadpath不一样 所以逻辑没问题
        if self.hparams.config["load_path"] == "":
            # 没有VILT的参数 需要载入VIT参数从头训练（pretrained=True）
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )
        else:
            # VILT的参数包括了VIT的参数 只要VIT的结构 无需载入VIT参数
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        # if config["loss_names"]["mlm"] > 0:
        #     self.mlm_score = heads.MLMHead(bert_config)
        #     self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        # if config["loss_names"]["mpp"] > 0:
        #     self.mpp_score = heads.MPPHead(bert_config)
        #     self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        # ====================针对不同数据集建立分类器（目前只有一个数据集）
        # ======== 二分类头
        # self.hs_timer = 2  # 隐藏层规模倍数 因为vit只能处理3个通道 对于更多通道的影像 需要重复使用vit 该参数本质上是使用vit的次数  # <--
        self.hs_timer = 1  # <-
        if config["loss_names"]["modmis_bin"] > 0:  # 配置文件 config 里面设置是否启用
            self.modmis_binarizer = heads.ModmisBinHead(config["hidden_size"] * self.hs_timer,
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

        # ======== 读取VILT的 checkpoint
        if self.hparams.config["load_path"] != "":
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
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

        # ======== 各种缺失情况下的提示（需要补充剩下的22种缺失情况）
        sit_ct = -1
        missing_aware_prompt = nn.ParameterDict()
        for f_key in ['none']+ data_info["fields_keys"]:
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



    # def lortho_loss(self, P_s, epsilon=1e-8):
    #         total_loss = 0
    #         num_prompts = len(P_s)
    #         for i in range(num_prompts):
    #             for j in range(i + 1, num_prompts):
    #                 Pis = P_s[i]
    #                 Pts = P_s[j]
    #                 flat_Pis = torch.flatten(Pis, start_dim=1)
    #                 flat_Pts = torch.flatten(Pts, start_dim=1)
    #                 dot_product = torch.sum(flat_Pis * flat_Pts, dim=1)
    #                 norm_Pis = torch.norm(flat_Pis, p=2, dim=1)
    #                 norm_Pts = torch.norm(flat_Pts, p=2, dim=1)
    #                 norm_Pis = torch.clamp(norm_Pis, min=epsilon)
    #                 norm_Pts = torch.clamp(norm_Pts, min=epsilon)
    #                 neg_cosine_similarity = -dot_product / (norm_Pis * norm_Pts)
    #                 Lortho = torch.max(neg_cosine_similarity, torch.tensor(0.0))
    #                 total_loss += Lortho.mean()
    #         return total_loss / (num_prompts * (num_prompts - 1) / 2)
    #
    # def train_step(self, batch):
    #         # 获取提示参数
    #         P_1_s = self.missing_field_prompt["field_1"]
    #         P_2_s = self.missing_field_prompt["field_2"]
    #         P_3_s = self.missing_field_prompt["field_3"]
    #         P_4_s = self.missing_field_prompt["field_4"]
    #         P_5_s = self.missing_field_prompt["field_5"]
    #         P_6_s = self.missing_field_prompt["field_6"]
    #         P_s = [P_1_s, P_2_s, P_3_s, P_4_s, P_5_s, P_6_s]
    #
    #         # 计算正交性损失
    #         loss_ortho = self.lortho_loss(P_s)
    #
    #         # 清除之前的梯度
    #         if self.learnt_p:
    #             self.zero_grad()
    #             loss_ortho.backward()
    #             self.missing_field_prompt["none"] = P_1_s + P_2_s + P_3_s + P_4_s + P_5_s + P_6_s
    #             self.missing_field_prompt["none"].detach_()  # 确保none_prompt不参与梯度计算
    #             self.step()
    #         else:
    #             loss_ortho = torch.tensor(0.)
    #         print(f"已经降低了正交损失")
    #         return {"loss": loss_ortho}
    #         self.optimizer=self.configure_optimizers
    #         #评估指标设置
    #         vilt_utils.set_metrics(self)  # 设置评估的指标
    #         self.current_tasks = list()
    #         self.records = {}
    #
    #          # 其它属性设置
    #         self.best_train_roc = []  # 训练时最好的roc信息
    #         self.best_val_roc = []  # 验证时最好的roc信息
    #         self.bcewl_pos_weight = torch.tensor(config["bcewl_pos_weight"], device="cuda")  # 目标函数中采用posweight时的权重设置
    #


        # ===== 是否冻结 Transformer （原本是冻结） vit 的
        #     if not self.hparams.config.get("finetune_vit",False):
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

    # def configure_optimizers(self):
    #         if self.learnt_p:
    #             optimizer = torch.optim.SGD(list(self.missing_field_prompt.values()), lr=0.01)
    #             return optimizer
    #         else:
    #             return None
    def infer(  # 重点修改的地方
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
            is_train=None,
    ):
        # if f"image_{image_token_type_idx - 1}" in batch:
        #     img_keys = f"image_{image_token_type_idx - 1}"  # 这步不会执行
        # else:
        #     img_keys = self.data_info["image_keys"]

        token_type = -1
        # 指标模态处理
        do_mlm = "_mlm" if mask_text else ""
        fields_list, field_masks_list, field_embeds_list = [], [], []
        for f_key in self.data_info["fields_keys"]:
            token_type += 1
            # field_labels = batch[f"field_labels{do_mlm}"]  # 注意 这里不是真正的field_lable(f_ending) 只是值全为-100的张量
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



        # vit推断
        co_masks = torch.cat([prompt_masks] + field_masks_list , dim=1)
        x = torch.cat(field_embeds_list , dim=1).detach()
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
            # out_feats.append(self.pooler(x[:, total_prompt_len:total_prompt_len + 1]))
            out_feats = self.pooler(x[:, total_prompt_len:total_prompt_len + 1])
        #total_prompt_len 96
        elif self.prompt_type == 'attention':
            # out_feats.append(self.pooler(x))
            out_feats = self.pooler(x)

        # out_feats = torch.cat(out_feats, dim=-1)
        # out_feats = torch.stack(out_feats).mean(dim=0)
        ret = {
            # "field_feats": field_feats,  # text_features替换成字段特征，并以此类推多种字段特征
            # "image_feats": image_feats,
            "out_feats": out_feats,
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
