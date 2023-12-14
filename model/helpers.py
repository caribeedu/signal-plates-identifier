# import mask rcnn libraries
from mrcnn.model import MaskRCNN

# import dataset
from dataset import SignalPlatesDataset

def build_dataset(name, sets_path):
    # create dataset
    set = SignalPlatesDataset()
    # load the dataset
    set.load_dataset(sets_path, name)
    # prepare dataset
    set.prepare()

    return set

def create_model(mode, config, weights_path):
    print(f"Creating model for '{mode}'...")

    # define the model
    model = MaskRCNN(mode=mode, model_dir='./', config=config)

    print('Loading model weights...')

    # load the model weights
    if mode == 'inference':
        model.load_weights(weights_path, by_name=True)
    elif mode == 'training':
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    return model