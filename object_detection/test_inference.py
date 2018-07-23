# Copyright 2018 DarkRabbit. All Rights Reserved.
# Author BLOG: https://blog.csdn.net/DarkRabbit
# Licensed under the Apache License, Version 2.0.
# ==============================================================================

import od_path
od_path.add_path()

import os
import re
import base64
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils

flags = tf.app.flags

flags.DEFINE_string("input_path", None, "The input image.")
flags.DEFINE_string("input_base64str", None, "The input image base64str")
flags.DEFINE_string("frozen_path", None, "The frozen inference graph path.")
flags.DEFINE_string("labels_items_path", None, "The labels items path.")
flags.DEFINE_string("output_path", "./output.png", "The output inference path.")
flags.DEFINE_integer("num_classes", 5, "The number of classes.")

FLAGS = flags.FLAGS


def _load_frozen_model_into_memory(frozen_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()

        with tf.gfile.GFile(frozen_path, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")
            return detection_graph


def _loading_label_map(labels_items_path, max_num_classes):
    label_map = label_map_util.load_labelmap(labels_items_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=max_num_classes)
    category_index = label_map_util.create_category_index(categories)
    return label_map, categories, category_index

def load_image(image_path = None, image_base64str = None):
    if not image_path and not image_base64str:
        raise Exception("`image_path` or `image_base64str` can not be None.")
    path = image_path
    if not path:
        base64_data = re.sub("^data:image/.+;base64,", "", image_base64str)
        bytes_data = base64.b64decode(base64_data)
        path = BytesIO(bytes_data)

    image = Image.open(path)
    return image

def load_image_into_numpy_array(image):
    """
    parameters:
        image: PIL.Image(image_path)

    return: numpy array
    """
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image_np, graph):
    """
    parameters:
        image_np: load_image_into_numpy_array(image)
        graph: a graph of tensorflow
    
    return: dict
        num_detections,
        detection_classes,
        detection_boxes,
        detection_scores,
        detection_masks, if in tensor_names
    """
    with graph.as_default():
        with tf.Session() as sess:

            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                "num_detections",
                "detection_boxes",
                "detection_scores",
                "detection_classes",
                "detection_masks"
            ]:
                tensor_name = key + ":0"
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            if "detection_masks" in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict["detection_boxes"], [0])
                detection_masks = tf.squeeze(
                    tensor_dict["detection_masks"], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict["num_detections"][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks,
                    detection_boxes,
                    image_np.shape[0],
                    image_np.shape[1]
                )
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict["detection_masks"] = tf.expand_dims(
                    detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name("image_tensor:0")

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict = {image_tensor: np.expand_dims(image_np, 0)})

            # All outputs are float32 numpy arrays, so convert types as appropriate
            output_dict["num_detections"] = int(output_dict["num_detections"][0])
            output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(np.uint8)
            output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
            output_dict["detection_scores"] = output_dict["detection_scores"][0]
            if "detection_masks" in output_dict:
                output_dict["detection_masks"] = output_dict["detection_masks"][0]
    return output_dict

def main(unused_argv):
    assert FLAGS.input_path or FLAGS.input_base64str, "`input_path` or `input_base64str` is missing."
    assert FLAGS.frozen_path, "`frozen_path` is missing."
    assert FLAGS.labels_items_path, "`labels_items_path` is missing."
    assert FLAGS.output_path, "`output_path` is missing."
    assert FLAGS.num_classes > 0, "`num_classes` is less than or equal to zero."

    detection_graph = _load_frozen_model_into_memory(FLAGS.frozen_path)
    label_map, categories, category_index = _loading_label_map(FLAGS.labels_items_path, max_num_classes=FLAGS.num_classes)

    image = load_image(FLAGS.input_path, FLAGS.input_base64str)
    image_np = load_image_into_numpy_array(image)

    output_dict = run_inference_for_single_image(image_np, detection_graph)

    vis_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict["detection_boxes"], # np.squeeze(boxes),
        output_dict["detection_classes"], # np.squeeze(classes).astype(np.int32),
        output_dict["detection_scores"], # np.squeeze(scores),
        category_index,
        instance_masks = output_dict.get("detection_masks"),
        use_normalized_coordinates=True,
        min_score_thresh=.3,
        line_thickness=8
    )
    
    print(output_dict["detection_boxes"])
    print(output_dict["detection_classes"])
    print(output_dict["detection_scores"])

    output_dir = os.path.dirname(FLAGS.output_path)
    tf.gfile.MakeDirs(output_dir)
    plt.imsave(FLAGS.output_path, image_np)


if __name__ == "__main__":
    tf.app.run()
