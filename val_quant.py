from imagenet import NormalizeMethod
from multiprocessing import Pool
from tensorflow.keras.models import load_model
import tensorflow.lite as tflite
from tqdm import tqdm

import argparse
import imagenet


global interpreter, input_index, output_index


def init(tflite_model_file):
    global interpreter, input_index, output_index
    interpreter = tflite.Interpreter(tflite_model_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]


def eval(data):
    image, label = data
    global interpreter, input_index, output_index
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
    return int(prediction.argmax() == label[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tflite_model_file',
        dest="tflite_model_file",
        required=True)
    parser.add_argument('--imagenet_path', dest="imagenet_path", required=True)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--cpus', dest="cpus", type=int, default=24)
    parser.add_argument('--no_background', action='store_true')
    args = parser.parse_args()

    val_size = 50000
    dataset = imagenet.get_val_dataset(
        args.imagenet_path,
        1,
        imagenet.NormalizeMethod.NONE,
        image_size=args.image_size,
        include_background=not args.no_background)
    dataset = dataset.take(val_size).as_numpy_iterator()
    correct = 0

    with Pool(processes=args.cpus, initializer=init, initargs=(args.tflite_model_file,)) as pool:
        with tqdm(pool.imap_unordered(eval, dataset), total=val_size) as t:
            for idx, result in enumerate(t, start=1):
                correct += result
                t.set_postfix(correct=correct, accuracy=correct / idx)

    print(f"{correct}/{val_size}")


if __name__ == '__main__':
    main()
