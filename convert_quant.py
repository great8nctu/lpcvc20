from functools import partial

import argparse
import imagenet
import numpy as np
import tensorflow as tf

def representative_data_gen(args):
    dataset = imagenet.get_val_dataset(args.imagenet_path, batch_size=1, normalize_method=imagenet.NormalizeMethod.TF, image_size=args.image_size, include_background=not args.no_background)
    dataset = dataset.take(100)
    for image, label in dataset:
        yield [image]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keras_model_file', dest="keras_model_file", required=True)
    parser.add_argument('--output_file', dest="output_file", required=True)
    parser.add_argument('--imagenet_path', dest="imagenet_path", required=True)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--no_background', action='store_true')
    args = parser.parse_args()

    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(args.keras_model_file)
    converter.representative_dataset = partial(representative_data_gen, args=args)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()
    open(args.output_file, 'wb').write(tflite_quant_model)

if __name__ == '__main__':
    main()
