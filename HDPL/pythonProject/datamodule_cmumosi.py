

from pythonProject.dataset_cmumosi import MODMISDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import time


class MODMISDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self.data_dir = _config["data_root"]

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.image_size = _config["image_size"]
        # self.draw_false_image = _config["draw_false_image"]
        self.draw_false_field = _config["draw_false_field"]
        self.image_only = _config["image_only"]

        # construct missing modality info
        self.missing_info = {
            'ratio': _config["missing_ratio"],
            'type': _config["missing_type"],
            'both_ratio': _config["both_ratio"],
            'missing_table_root': _config["missing_table_root"],
            'cache_root': _config["cache_root"],
            'simulate_missing': _config["simulate_missing"]
        }

        # for bash execution
        if _config["test_ratio"] is not None:  # None
            self.missing_info['ratio']['val'] = _config["test_ratio"]
            self.missing_info['ratio']['test'] = _config["test_ratio"]
        if _config["test_type"] is not None:  # None
            self.missing_info['type']['val'] = _config["test_type"]
            self.missing_info['type']['test'] = _config["test_type"]

        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )

        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        self.setup_flag = False
        self.used_labels = _config["modmis_label_used"]
        self.used_fields = _config["modmis_field_used"]

        self.fold_id = _config["fold_id"]
        self.fold_count = _config["fold_count"]

    def get_dataset_info(self):
        tmp_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="train",
            image_size=self.image_size,
            # draw_false_image=self.draw_false_image,
            draw_false_field=self.draw_false_field,
            image_only=self.image_only,
            missing_info=self.missing_info,
            used_labels=self.used_labels,
            field_column_name_list=self.used_fields,
        )
        return tmp_dataset.get_info()

    @property
    def dataset_cls(self):
        return MODMISDataset

    @property
    def dataset_name(self):
        return "modmis"

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,  # "datasets/modmis/"
            self.train_transform_keys,  # ["pixelbert"]
            split="train",
            image_size=self.image_size,  # 384
            # draw_false_image=self.draw_false_image,  # 1
            draw_false_field=self.draw_false_field,  # 0
            image_only=self.image_only,  # False
            # {ration, type, both_ration, missing_table_root, simulate_missing}
            # 0.7, both, 0.5, "./datasets/missing_tables/", False
            missing_info=self.missing_info,
            used_labels=self.used_labels,
            field_column_name_list=self.used_fields,
        )
        # return  tmp_dataset.get_info()

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,  # "datasets/modmis/"
            self.val_transform_keys,  # ["pixelbert"]
            split="val",
            #split="test",
            image_size=self.image_size,  # 384
            # draw_false_image=self.draw_false_image,  # 1
            draw_false_field=self.draw_false_field,  # 0
            image_only=self.image_only,  # False
            # {ration, type, both_ration, missing_table_root, simulate_missing}
            # 0.7, both, 0.5, "./datasets/missing_tables/", False
            missing_info=self.missing_info,
            used_labels=self.used_labels,
            field_column_name_list=self.used_fields,
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.make_no_false_val_dset(self.image_only)
        # return tmp_dataset.get_info()

    def make_no_false_val_dset(self, image_only=False):  # ？
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            #split="test",
            image_size=self.image_size,
            # draw_false_image=0,
            draw_false_field=0,
            image_only=image_only,
            used_labels=self.used_labels,
            field_column_name_list=self.used_fields,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            # draw_false_image=self.draw_false_image,
            draw_false_field=self.draw_false_field,
            # image_only=self.image_only,
            missing_info=self.missing_info,
            used_labels=self.used_labels,
            field_column_name_list=self.used_fields,
        )

    # def fold_exchange(self, fold_id, fold_count=5):
    #     train_val_count = len(self.train_dataset) + len(self.val_dataset)
    #     val_count = train_val_count // fold_count
    #     train_data, val_data = MODMISDataset.split_data(
    #         MODMISDataset.merge_data([
    #             self.train_dataset.get_data(list(range(0, len(self.train_dataset)))),
    #             self.val_dataset.get_data(list(range(0, len(self.val_dataset)))),
    #         ]),
    #         [
    #             list(range(0, val_count * fold_id)) + list(range(val_count * fold_id + val_count, train_val_count)),
    #             list(range(val_count * fold_id, val_count * fold_id + val_count)),
    #         ]
    #     )
    #     self.train_dataset.set_data(train_data)
    #     self.val_dataset.set_data(val_data)
    # def fold_exchange(self, fold_id, fold_count=5):
    #     # 1. 获取训练集的原始数据（而非数据集对象）
    #     total_count = len(self.train_dataset)
    #     # 假设 get_data() 接收索引列表，返回对应的数据
    #     total_data = self.train_dataset.get_data(list(range(total_count)))
    #
    #     # 2. 计算每折大小（处理不能整除的情况）
    #     val_count = total_count // fold_count
    #     start = val_count * fold_id
    #     end = min(start + val_count, total_count)  # 避免最后一折越界
    #
    #     # 3. 划分索引
    #     val_indices = list(range(start, end))
    #     train_indices = list(range(0, start)) + list(range(end, total_count))
    #
    #     # 4. 分割数据（基于实际数据内容，而非数据集对象）
    #     train_data, val_data = MODMISDataset.split_data(total_data, [train_indices, val_indices])
    #
    #     # 5. 更新数据集（如果必须修改原对象，确保后续逻辑兼容）
    #     self.train_dataset.set_data(train_data)
    #     self.val_dataset.set_data(val_data)
    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            # self.set_val_dataset()
            # if self.fold_id >= 0 and self.fold_count >= 2:
            #     self.fold_exchange(self.fold_id, self.fold_count)
            self.set_test_dataset()
            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    # def val_dataloader(self):
    #     loader = DataLoader(
    #         self.val_dataset,
    #         batch_size=self.eval_batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=True,
    #         collate_fn=self.val_dataset.collate,
    #     )
    #     return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
