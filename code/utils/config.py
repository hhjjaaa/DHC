
class Config:
    def __init__(self,task):
        if task == "synapse":
            self.base_dir = './Datasets/Synapse'
            self.save_dir = './synapse_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 14
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
        elif task == "amos": # amos
            self.base_dir = './Datasets/amos22'
            self.save_dir = './amos_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 16
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
        else:
            self.base_dir = './Datasets/amos22'
            self.save_dir = '/home/hutianjiao/Project/DHC/mri_data/YX_data5/'
            self.patch_size = (448, 448)
            self.num_cls = 2
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100