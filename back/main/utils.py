from os import listdir
from os.path import isfile, join
import pybboxes as pbx


def get_all_files_from(folder_path: str, formats: list = None):
    files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    if formats is not None:
        files = [f for f in files if f.lower().endswith(tuple(formats))]
    files = [join(folder_path, x) for x in files]
    return files


def get_all_folders_from(mypath):
    folders = [join(mypath, f) for f in listdir(mypath) if not isfile(join(mypath, f))]
    return folders


def get_voc_bboxes(label_file: str, image_size: tuple):
    boxes = []
    with open(label_file, 'r') as lf:
        lines = [line.rstrip() for line in lf]
        for line in lines:
            split = line.split(' ')
            if split[0] != '0':
                print('Other class %s' % split[0])
                continue
            points = [float(c) for c in split[-4:]]
            bbox = (points[0], points[1], points[2], points[3])
            xmin, ymin, xmax, ymax = pbx.convert_bbox(bbox, from_type="yolo", to_type="voc", image_size=image_size)
            boxes.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
    return boxes
