# import basic libraries
import os
import json

# import advance libraries
import skimage.draw

# import mask rcnn libraries
from mrcnn.utils import Dataset

# import numpy libraries
import numpy as np

class SignalPlatesDataset(Dataset):

    def load_dataset(self, dataset_dir, subset):

        print(f"Loading dataset '{subset}'...")

        # we use add_class for each class in our dataset and assign numbers to them. 0 is background
        self.add_class("code", 1, "r-1")
        self.add_class("code", 2, "r-2")
        self.add_class("code", 3, "r-3")
        self.add_class("code", 4, "r-4a")
        self.add_class("code", 5, "r-4b")
        self.add_class("code", 6, "r-5a")
        self.add_class("code", 7, "r-5b")
        self.add_class("code", 8, "r-6a")
        self.add_class("code", 9, "r-6b")
        self.add_class("code", 10, "r-6c")
        self.add_class("code", 11, "r-7")
        self.add_class("code", 12, "r-8a")
        self.add_class("code", 13, "r-8b")
        self.add_class("code", 14, "r-9")
        self.add_class("code", 15, "r-10")
        self.add_class("code", 16, "r-11")
        self.add_class("code", 17, "r-12")
        self.add_class("code", 18, "r-13")
        self.add_class("code", 19, "r-14")
        self.add_class("code", 20, "r-15")
        self.add_class("code", 21, "r-16")
        self.add_class("code", 22, "r-17")
        self.add_class("code", 23, "r-18")
        self.add_class("code", 24, "r-19")
        self.add_class("code", 25, "r-20")
        self.add_class("code", 26, "r-21")
        self.add_class("code", 27, "r-22")
        self.add_class("code", 28, "r-23")
        self.add_class("code", 29, "r-24a")
        self.add_class("code", 30, "r-24b")
        self.add_class("code", 31, "r-25a")
        self.add_class("code", 32, "r-25b")
        self.add_class("code", 33, "r-25c")
        self.add_class("code", 34, "r-25d")
        self.add_class("code", 35, "r-26")
        self.add_class("code", 36, "r-27")
        self.add_class("code", 37, "r-28")
        self.add_class("code", 38, "r-29")
        self.add_class("code", 39, "r-30")
        self.add_class("code", 40, "r-31")
        self.add_class("code", 41, "r-32")
        self.add_class("code", 42, "r-33")
        self.add_class("code", 43, "r-34")
        self.add_class("code", 44, "r-35a")
        self.add_class("code", 45, "r-35b")
        self.add_class("code", 46, "r-36a")
        self.add_class("code", 47, "r-36b")
        self.add_class("code", 48, "r-37")
        self.add_class("code", 49, "r-38")
        self.add_class("code", 50, "r-39")
        self.add_class("code", 51, "r-40")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # load annotations using json.load()
        annotations = json.load(open(os.path.join(dataset_dir, "signal-plates-identifier-vgg-project.json")))

        # convert annotations into a list
        annotations_list = list(annotations.values())

        # we only require the regions in the annotations list
        annotations_list = [anno for anno in annotations_list if anno['regions']]

        # Add images
        for anno in annotations_list:
            # extracting shape attributes and region attributes
            polygons = [r['shape_attributes'] for r in anno['regions']]
            codes = [s['region_attributes']['code'] for s in anno['regions']]

            # create a dictionary {name_of_class: class_id} remember background has id 0
            name_dict = {'r-1': 1, 'r-2': 2, 'r-3': 3, 'r-4a': 4, 'r-4b': 5, 'r-5a': 6, 'r-5b': 7, 'r-6a': 8, 'r-6b': 9, 'r-6c': 10, 'r-7': 11, 'r-8a': 12, 'r-8b': 13, 'r-9': 14, 'r-10': 15, 'r-11': 16, 'r-12': 17, 'r-13': 18, 'r-14': 19, 'r-15': 20, 'r-16': 21, 'r-17': 22, 'r-18': 23, 'r-19': 24, 'r-20': 25, 'r-21': 26, 'r-22': 27, 'r-23': 28, 'r-24a': 29, 'r-24b': 30, 'r-25a': 31, 'r-25b': 32, 'r-25c': 33, 'r-25d': 34, 'r-26': 35, 'r-27': 36, 'r-28': 37, 'r-29': 38, 'r-30': 39, 'r-31': 40, 'r-32': 41, 'r-33': 42, 'r-34': 43, 'r-35a': 44, 'r-35b': 45, 'r-36a': 46, 'r-36b': 47, 'r-37': 48, 'r-38': 49, 'r-39': 50, 'r-40': 51}

            # all the ids/classes in a image
            num_ids = [name_dict[code] for code in codes]

            # you can print these ids
            # print("numids",num_ids)

            # read image and get height and width
            image_path = os.path.join(dataset_dir, anno['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            # add image to the dataset
            self.add_image(
                "code",
                image_id=anno['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

        print(f"Dataset '{subset}' successfully loaded.")

    # this function calls on the extract_boxes method and is used to load a mask for each instance in an image
    # returns a boolean mask with following dimensions width * height * instances
    def load_mask(self, image_id):

        # info points to the current image_id
        info = self.image_info[image_id]

        # for cases when source is not code
        if info["source"] != "code":
            return super(self.__class__, self).load_mask(image_id)

        # get the class ids in an image
        num_ids = info['num_ids']

        # we create len(info["polygons"])(total number of polygons) number of masks of height 'h' and width 'w'
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        # we loop over all polygons and generate masks (polygon mask) and class id for each instance
        # masks can have any shape as we have used polygon for annotations
        # for example: if 2.jpg have four objects we will have following masks and class_ids
        # 000001100 000111000 000001110
        # 000111100 011100000 000001110
        # 000011111 011111000 000001110
        # 000000000 111100000 000000000
        #    1         2          3    <- class_ids
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'], mask.shape)

            mask[rr, cc, i] = 1

        # return masks and class_ids as array
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    # this functions takes the image_id and returns the path of the image
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "code":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
    