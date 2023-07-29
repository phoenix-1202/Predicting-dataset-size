from abc import ABC, abstractmethod
from back.main.utils import get_all_files_from


def _contains(path: str, names: set):
    the_name = path.split('/')[-1]
    for s in names:
        if s in the_name:
            return True
    return False


def _get_common_names(first_paths: list, second_paths: list):
    image_names = set([s.split('/')[-1].split('.')[0] for s in first_paths])
    label_names = set([s.split('/')[-1].split('.')[0] for s in second_paths])
    return image_names.intersection(label_names)


class TrainingSetSizePredictor(ABC):
    def __init__(self):
        self.image_paths = []
        self.label_paths = []
        self.map50 = None
    
    number_of_images = 5
    image_formats = ['.jpeg', '.jpg', '.png']
    labels_formats = ['.txt']
    
    def add_images(self, folder_path: str):
        self.image_paths = get_all_files_from(folder_path, self.image_formats)
    
    def add_labels(self, folder_path: str):
        self.label_paths = get_all_files_from(folder_path, self.labels_formats)
    
    def set_map50(self, map50: float):
        self.map50 = map50
    
    def is_ready_to_predict(self):
        no_images = len(_get_common_names(self.image_paths, self.label_paths))
        return no_images >= self.number_of_images and self.map50 is not None
    
    def predict(self):
        common_names = _get_common_names(self.image_paths, self.label_paths)
        images = sorted([path for path in self.image_paths if _contains(path, common_names)])
        labels = sorted([path for path in self.label_paths if _contains(path, common_names)])
        return self._make_prediction(zip(images, labels))
    
    @abstractmethod
    def _make_prediction(self, images_and_labels):
        pass
