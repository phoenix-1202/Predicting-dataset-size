from statistics import median
import torch
from pickle import load
import numpy as np
from back.implementation_skip_connection.FullyConnectedNet import FullyConnectedNet
from back.implementation_skip_connection.features1065 import Features1065
from back.main.TrainingSetSizePredictor import TrainingSetSizePredictor


class FullyConnectedSizePredictor(TrainingSetSizePredictor):
    def __init__(self, by):
        super().__init__()
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._feature_extractor = Features1065(by, self._device)
        self._net = FullyConnectedNet()
        self._net.load_state_dict(
            torch.load('back/implementation_skip_connection/best-nn-with-skip-connection_1066.pt', map_location=torch.device('cpu')))
        self._net.to(self._device)
        self._scaler = load(open('back/implementation_skip_connection/scaler.pkl', 'rb'))
        
    def _make_prediction(self, images_and_labels):
        features = self._feature_extractor.get_features(images_and_labels)
        features = [np.append(f, self.map50) for f in features]
        features = self._scaler.transform(features)
        features = torch.stack([torch.from_numpy(f).to(torch.float32).to(self._device) for f in features])
        sizes, _ = self._net(features)
        sizes = sizes.flatten().tolist()
        return round(median(sizes))
