import itertools
from statistics import median
import torch
from torchvision.transforms import functional
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from PIL import Image
from back.implementation_inceptionv3.InceptionNet import InceptionV3Head, InceptionV3Tail
from back.main.TrainingSetSizePredictor import TrainingSetSizePredictor
from back.main.utils import get_voc_bboxes


class InceptionSizePredictor(TrainingSetSizePredictor):
    def __init__(self):
        super().__init__()
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._features = InceptionV3Head().eval().to(self._device)
        self._model = InceptionV3Tail(self._by)
        self._model.load_state_dict(torch.load(self._weights, map_location=torch.device('cpu')))
        self._model.to(self._device)
        self._pil_to_tensor = transforms.PILToTensor()

    _by = 2
    _weights = 'back/implementation_inceptionv3/best-separate_test_class_1_frozen_no_reliability_inceptionv3_by2.pt'
    _transforms = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _make_prediction(self, images_and_labels):
        image_features = []
        for image_path, label_path in images_and_labels:
            image = Image.open(image_path).convert('RGB')
            bboxes = get_voc_bboxes(label_path, image.size)
            print(bboxes)
            bboxes = [torch.tensor([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]) for bbox in bboxes]
            print(bboxes)
            bboxes = torch.stack(bboxes)
            image = self._pil_to_tensor(image)
            image_with_boxes = draw_bounding_boxes(image, bboxes, width=5)
            image_with_boxes = image_with_boxes.detach()
            image_with_boxes = functional.to_pil_image(image_with_boxes)
            image_tensor = self._transforms(image_with_boxes).to(self._device).unsqueeze(dim=0)
            image_tensor = self._features(image_tensor)
            image_features.append(image_tensor)
        positions = [p for p in range(0, len(image_features))]
        all_combinations = [i for i in itertools.combinations(positions, self._by)]
        sizes = []
        map50_tensor = torch.tensor([self.map50]).to(self._device)
        for comb in all_combinations:
            features_i = torch.stack([image_features[c] for c in comb])
            size_i = self._model(features_i, map50_tensor).detach().cpu()[0][0].item()
            sizes.append(size_i)
        return round(median(sizes))
