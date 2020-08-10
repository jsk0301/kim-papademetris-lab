"""
A mock server that uses gateway.py to establish a web server. Depending on the command line options provided,
"-s2D", "-s3D" or "-b", the server is capable of returning either a sample 2D segmentation, 3D segmentation or
bounding box correspondingly when an inference reuqest is sent to the "/" route.

"""

import argparse
import functools
import json
import logging
import logging.config
import os
import tempfile
import yaml

import numpy
import pydicom
from utils import tagged_logger

import prostate_model

# ensure logging is configured before flask is initialized

with open('logging.yaml', 'r') as f:
    conf = yaml.safe_load(f.read())
    logging.config.dictConfig(conf)

logger = logging.getLogger('inference')

# pylint: disable=import-error,no-name-in-module
from gateway import Gateway
from flask import make_response

def handle_exception(e):
    logger.exception('internal server error %s', e)
    return 'internal server error', 500

def get_empty_response():
    response_json = {
        'protocol_version': '1.0',
        'parts': []
    }
    return response_json, []

def healthcheck_handler():
    # Return if the model is ready to receive inference requests

    return make_response('READY', 200)

def get_probability_mask_3D_response(json_input, dicom_instances):
    # Assuming that all files have the same size
    dcm = pydicom.read_file(dicom_instances[0])
    depth = len(dicom_instances)
    image_width = dcm.Columns
    image_height = dcm.Rows
    response_json = {
        'protocol_version': '1.0',
        'parts': [
            {
                'label': 'Mock seg',
                'binary_type': 'probability_mask',
                'binary_data_shape': {
                    'timepoints': 1,
                    'depth': depth,
                    'width': image_width,
                    'height': image_height
                }
            }
        ]
    }

    array_shape = (depth, image_height, image_width)

    # This code produces a mask that grows from the center of the image outwards as the image slices advance
    mask = numpy.zeros(array_shape, dtype=numpy.uint8)
    mid_x = int(image_width / 2)
    mid_y = int(image_height / 2)
    for s in range(depth):
        offset_x = int(s / depth * mid_x)
        offset_y = int(s / depth * mid_y)
        indices = numpy.ogrid[mid_y - offset_y : mid_y + offset_y, mid_x - offset_x : mid_x + offset_x]
        mask[s][tuple(indices)] = 255

    return response_json, [mask]

def request_handler_3D_segmentation(json_input, dicom_instances, input_digest):
    """
    A mock inference model that returns a mask array of ones of size (height * depth, width)
    """
    transaction_logger = tagged_logger.TaggedLogger(logger)
    transaction_logger.add_tags({ 'input_hash': input_digest })
    transaction_logger.info('mock_model received json_input={}'.format(json_input))
    return get_probability_mask_3D_response(json_input, dicom_instances)

if __name__ == '__main__':
    args = parse_args()
    app = Gateway(__name__)
    app.register_error_handler(Exception, handle_exception)
    app.add_inference_route('/', request_handler_3D_segmentation)
    app.add_healthcheck_route(healthcheck_handler)
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=True)
