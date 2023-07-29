from back.implementation_inceptionv3.InceptionSizePredictor import InceptionSizePredictor
from front.gui import GUI


if __name__ == '__main__':
    predictor = InceptionSizePredictor()
    gui = GUI(predictor)
    gui.draw()

