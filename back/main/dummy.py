from back.main.TrainingSetSizePredictor import TrainingSetSizePredictor


class Dummy(TrainingSetSizePredictor):
    def __init__(self):
        super().__init__()

    def _make_prediction(self, images_and_labels):
        return 324
