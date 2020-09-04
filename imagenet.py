from enum import Enum
from functools import partial

import tensorflow as tf
import os
import preprocess


class NormalizeMethod(Enum):
    DEFAULT = 1
    TF = 2
    PYTORCH = 3
    NONE = 4
    POSTQUANT = 5

    def __str__(self):
        return self.name


def parser(record):
    img_features = tf.io.parse_single_example(
        record,
        features={
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
            'image/encoded': tf.io.FixedLenFeature([], tf.string)})

    label = tf.cast(img_features['image/class/label'], tf.int64)
    byte = tf.cast(img_features['image/encoded'], tf.string)

    return byte, label


def normalize(image, method):
    if method == NormalizeMethod.DEFAULT:
        image = tf.divide(image, 255)
    elif method == NormalizeMethod.TF:
        image = tf.subtract(image, [0.5 * 255, 0.5 * 255, 0.5 * 255])
        image = tf.divide(image, [0.5 * 255, 0.5 * 255, 0.5 * 255])
    elif method == NormalizeMethod.PYTORCH:
        image = tf.subtract(image, [0.485 * 255, 0.456 * 255, 0.406 * 255])
        image = tf.divide(image, [0.229 * 255, 0.224 * 255, 0.225 * 255])
    elif method == NormalizeMethod.NONE:
        image = tf.cast(image, dtype=tf.uint8)
    elif method == NormalizeMethod.POSTQUANT:
        image = tf.subtract(image, [127, 127, 127])
        image = tf.divide(image, [128, 128, 128])
    return image


def one_hot(label, num_label):
    return tf.one_hot(label, num_label)


def get_dataset(
        files,
        batch_size,
        normalize_method,
        is_training=False,
        use_color_jitter=False,
        use_one_hot=False,
        image_size=224,
        use_cache=False,
        use_randaug=False,
        include_background=True):
    def _preprocess_image(byte, label):
        image = preprocess.preprocess_image(
            image_size=image_size,
            image_bytes=byte,
            is_training=is_training,
            use_color_jitter=use_color_jitter,
            use_randaug=use_randaug)
        image = normalize(image, method=normalize_method)
        if include_background:
            num_label = 1001
        else:
            num_label = 1000
            label -= 1
        if use_one_hot:
            label = one_hot(label, num_label)
        return image, label

    if is_training:
        shards = tf.data.Dataset.from_tensor_slices(files)
        shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
        dataset = shards.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.experimental.AUTOTUNE)
    else:
        dataset = tf.data.TFRecordDataset(filenames=files)
    dataset = (dataset
               .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
               .map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE))
    if use_cache:
        dataset = dataset.cache()
    if is_training:
        dataset = dataset.shuffle(50000)
    dataset = (dataset
               .map(_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .batch(batch_size, drop_remainder=is_training))
    return dataset


def get_train_dataset(
        tfrecords_dir,
        batch_size=256,
        normalize_method=NormalizeMethod.TF,
        use_color_jitter=False,
        use_one_hot=False,
        only_train=True,
        image_size=224,
        use_cache=False,
        use_randaug=False,
        include_background=True):
    if only_train:
        subset = 'train'
    else:
        subset = '*'
    files = tf.io.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
    dataset = get_dataset(
        files,
        batch_size,
        normalize_method,
        is_training=True,
        use_color_jitter=use_color_jitter,
        use_one_hot=use_one_hot,
        image_size=image_size,
        use_cache=use_cache,
        use_randaug=use_randaug,
        include_background=include_background)
    return dataset


def get_val_dataset(
        tfrecords_dir,
        batch_size=256,
        normalize_method=NormalizeMethod.TF,
        use_one_hot=False,
        image_size=224,
        use_cache=False,
        include_background=True):
    subset = 'validation'
    files = tf.io.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
    dataset = get_dataset(
        files,
        batch_size,
        normalize_method,
        is_training=False,
        use_one_hot=use_one_hot,
        image_size=image_size,
        use_cache=use_cache,
        include_background=include_background)
    return dataset


def test_speed(dataset):
    from timeit import default_timer as timer
    iterator = iter(dataset)
    for i in range(500):
        start_time = timer()
        _ = next(iterator)
        lap_time = timer() - start_time
        print("%s Time - %fsec " % (i, lap_time))
