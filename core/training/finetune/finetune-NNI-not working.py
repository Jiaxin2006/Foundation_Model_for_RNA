import sys
sys.path.append("../")
from torch.utils.data import random_split
from core.data_utils.dataset import FTDNADataset
import torch 
from nni.nas.strategy import RandomOneShot, GridSearch
import nni.nas.evaluator.pytorch.lightning as pl
from nni.nas.experiment import NasExperiment
from torch.utils.data import DataLoader
import socket
from core.models.cnn import CNN_ContrastiveLearning_PretrainModel, ModelConfig, CNN_ContrastiveLearning_ModelSpace
from nni.nas.evaluator import FunctionalEvaluator
import time
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from nni.nas.evaluator.pytorch import Classification

class CNNCLSModelSpace(ModelSpace):
    def __init__(self, config, num_classes):
        super().__init__()
        self.pretrained_model = CNN_ContrastiveLearning_ModelSpace(config)
        '''
        state_dict = torch.load(pretrained_model_path)['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("training_module._model.", "")  # 根据实际前缀修改
            new_state_dict[new_key] = v
        # 加载到你的模型中
        self.pretrained_model.load_state_dict(new_state_dict)
        '''        
        hidden_dim_list = config.hidden_dim_list
        max_channel = max(hidden_dim_list)
        self.cls_head = nn.Linear(max_channel, num_classes)

    def forward(self, x):
        features = self.pretrained_model(x)
        logits = self.cls_head(features)
        return logits
    
def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # 绑定端口0，让操作系统分配空闲端口
        return s.getsockname()[1]

def my_collate_fn(batch):
    xs, ys = zip(*batch)  # 解开成两个元组
    return list(xs), torch.tensor(ys)  # 返回 list[str], tensor


if __name__ == "__main__":
    for task_index in range(1):
        #train_dataset, val_dataset, test_dataset = FTDNADataset(task_index,'train'), FTDNADataset(task_index,'dev'), FTDNADataset(task_index,'test')

        #train_loader = DataLoader(train_dataset, batch_size=128, collate_fn = my_collate_fn)
        #val_loader = DataLoader(val_dataset, batch_size=128, collate_fn = my_collate_fn)
        #test_loader = DataLoader(test_dataset, batch_size=128, collate_fn = my_collate_fn)

        if task_index == 11:
            num_classes = 3
        else:
            num_classes = 2

        pretrained_path = f"/projects/slmreasoning/yifang/nni_pre_logs/Pretrain-step=12000.ckpt"
        config = ModelConfig.from_json("/projects/slmreasoning/yifang/configs/test_run/searchSpace_configs.json")
        config.channel_config_list_done_flag = True
        CNN_Finetune_Model = CNNCLSModelSpace(config=config,num_classes=num_classes)
        state_dict = torch.load(pretrained_path)['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("training_module._model.", "") 
            new_state_dict[new_key] = v

        CNN_Finetune_Model.pretrained_model.load_state_dict(new_state_dict)
        '''
        dummy_input = torch.tensor([[1, 3, 0, 1, 0, 2, 1, 3, 0, 1, 0, 2, 1, 3, 0, 1, 0, 2, 1, 3, 0, 1, 0, 2],[0, 0, 2, 1, 3, 1, 0, 0, 2, 1, 3, 1, 0, 0, 2, 1, 3, 1, 0, 0, 2, 1, 3, 1]])
        feat, output = CNN_Finetune_Model(dummy_input)
        logger.info(f"output is {output}")
        logger.info(f"feat is {feat}")
        '''
        #####
        evaluator = lambda: Classification(
            train_dataloaders=DataLoader(FTDNADataset(task_index,'train'), batch_size=128, collate_fn = my_collate_fn),
            val_dataloaders=DataLoader(FTDNADataset(task_index,'val'), batch_size=128, collate_fn = my_collate_fn),
            max_epochs=10,
            num_classes=num_classes
        )

        #search_strategy = GridSearch(shuffle=True, seed=42)
        search_strategy = RandomOneShot()

        exp = NasExperiment(CNN_Finetune_Model, evaluator, strategy=search_strategy)

        exp.config.trial_concurrency = 1
        exp.config.max_trial_number = 10
        exp.config.trial_gpu_number = 1
        exp.config.training_service.use_active_gpu = False
        exp.run(port=10086)

        best_model = exp.get_best_model()
        print("Best model:", best_model)

        # Optionally, evaluate on test set using best model
        model.load_state_dict(best_model.model_state_dict())
        model.eval()

        from sklearn.metrics import accuracy_score

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy: {test_acc:.4f}")
        '''
        ########
        #evaluator = FunctionalEvaluator(evaluate_model(model, task_index))
        
        exp = NasExperiment(CNN_Finetune_Model, evaluator, search_strategy)

        exp.config.max_trial_number = 100   # spawn 3 trials at most
        exp.config.trial_concurrency = 1  # will run 1 trial concurrently
        exp.config.trial_gpu_number = 1
        exp.config.experiment_working_directory = "/projects/slmreasoning/yifang/nni_ft_logs/"
        exp.config.debug = True
        exp.config.training_service.debug = True
        exp.config.training_service.use_active_gpu = False


        free_port = find_free_port()
        exp.run(port=10086)
        logger.debug("Search process is done. Press Ctrl+C to stop.")
        time.sleep(1e9)  # 模拟 `sleep forever`
        '''