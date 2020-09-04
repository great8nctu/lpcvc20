from imagenet import NormalizeMethod
from tensorflow.keras.models import load_model
from tqdm import trange

import argparse
import imagenet
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--keras_model_file',
        dest="keras_model_file",
        required=True)
    parser.add_argument('--imagenet_path', dest="imagenet_path", required=True)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument(
        '--normalize_method',
        dest="normalize_method",
        type=lambda method: NormalizeMethod[method],
        choices=list(NormalizeMethod),
        default=NormalizeMethod.TF)
    parser.add_argument(
        '--batch_size',
        dest="batch_size",
        type=int,
        default=256)
    args = parser.parse_args()

    val_size = 50000
    model = load_model(args.keras_model_file)
    dataset = imagenet.get_val_dataset(
        args.imagenet_path,
        args.batch_size,
        args.normalize_method,
        image_size=args.image_size)
    steps_per_epoch = val_size // args.batch_size + \
        int(val_size % args.batch_size != 0)
    iterator = iter(dataset)
    correct = 0

    with trange(steps_per_epoch) as t:
        for _ in t:
            images, labels = next(iterator)
            preds = tf.math.argmax(model.predict(images), -1)
            correct += tf.equal(preds, labels).numpy().sum()
            t.set_postfix(correct=correct)

    print(f"{correct}/{val_size}")


if __name__ == '__main__':
    main()
