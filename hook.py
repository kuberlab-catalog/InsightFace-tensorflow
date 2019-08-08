import cv2
from ml_serving.utils import helpers
import numpy as np


def process(inputs, ctx):
    img, _ = helpers.load_image(inputs, 'input')
    img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, axis=0)

    input_name = list(ctx.driver.inputs.keys())[0]
    outputs = ctx.driver.predict({input_name: img})

    output = list(outputs.values())[0]
    print(output)
    return {'output': output}

