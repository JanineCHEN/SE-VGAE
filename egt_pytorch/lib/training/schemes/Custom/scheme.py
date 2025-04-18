import torch
import numpy as np
import torch.nn.functional as F

from lib.training.training import cached_property
from ..egt_graph_training import EGT_GRAPH_Training

from lib.models.Custom import EGT_Custom
from lib.data.custom_from_txts import CustomStructuralSVDGraphDataset

class Custom_Training(EGT_GRAPH_Training):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name = 'ENZYMES',
            dataset_path = 'cache_data/ENZYMES',
            predict_on   = ['val'],
            evaluate_on  = ['val'],
            state_file   = None,
        )
        return config_dict
    
    def get_dataset_config(self):
        dataset_config, _ = super().get_dataset_config()
        return dataset_config, CustomStructuralSVDGraphDataset
    
    def get_model_config(self):
        model_config, _ = super().get_model_config()
        return model_config, EGT_Custom
    
    def calculate_loss(self, outputs, inputs):
        return F.l1_loss(outputs, inputs['target'])
    
    # @cached_property
    # def evaluator(self):
    #     # from ogb.lsc.pcqm4m import PCQM4MEvaluator
    #     # evaluator = PCQM4MEvaluator()
    #     return evaluator
    
    def prediction_step(self, batch):
        return dict(
            predictions = self.model(batch),
            targets     = batch['target'],
        )
    
    def evaluate_on(self, dataset_name, dataset, predictions):
        print(f'Evaluating on {dataset_name}')
        input_dict = {"y_true": predictions['targets'], 
                      "y_pred": predictions['predictions']}
        results = self.evaluator.eval(input_dict)
        for k, v in results.items():
            if hasattr(v, 'tolist'):
                results[k] = v.tolist()
        return results

SCHEME = Custom_Training
