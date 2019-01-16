# Overview

This repo contains code for the "TensorFlow for poets 2" series of codelabs.

There are multiple versions of this codelab depending on which version 
of the tensorflow libraries you plan on using:

* For [Retrainning a Model](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/), you can use this guide
* For [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) the new, ground up rewrite targeted at mobile devices
  use [this version of the codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite) 
* For the more mature [TensorFlow Mobile](https://www.tensorflow.org/mobile/mobile_intro) use 
  [this version of the codealab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2).


This repo contains simplified and trimmed down version of tensorflow's example image classification apps.

* The TensorFlow Lite version, in `android/tflite`, comes from [tensorflow/contrib/lite/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite).
* The Tensorflow Mobile version, in `android/tfmobile`, comes from [tensorflow/examples/android/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android).

The `scripts` directory contains helpers for the codelab. Some of these come from the main TensorFlow repository, and are included here so you can use them without also downloading the main TensorFlow repo (they are not part of the TensorFlow `pip` installation).

# Steps

## Clone the git repository

git clone https://github.com/googlecodelabs/tensorflow-for-poets-2

cd tensorflow-for-poets-2

## Prepare data

Prepare a set of images to teach the model about the new classes you want to recognize.

## (Re)training the network

The retrain script can retrain either Inception V3 model or a MobileNet. 

Inception V3 has a first-choice accuracy of 78% on ImageNet, but is the model is 85MB, and requires many times more processing than even the largest MobileNet configuration, which achieves 70.5% accuracy, with just a 19MB download.

ou will pass the settings inside Linux shell variables. Set those shell variables as follows:

IMAGE_SIZE=224

ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

CUDA_VISIBLE_DEVICES="/gpu:2"

python -m scripts.retrain \

  --bottleneck_dir=tf_files/bottlenecks \

  --how_many_training_steps=100000 \

  --model_dir=tf_files/model/ \

  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \

  --output_graph=tf_files/retrained_graph.pb \

  --output_labels=tf_files/retrained_labels.txt \

  --architecture="${ARCHITECTURE}" \

  --image_dir=tf_files/flower_photos \

  --learning_rate=0.001 \

  --train_batch_size=128

## Start TensorBoard

Before starting the training, launch tensorboard in the background. TensorBoard is a monitoring and inspection tool included with tensorflow. You will use it to monitor the training progress.

tensorboard --logdir tf_files/training_summaries &

## Create a TfLite Mobile App

### Optimize the model and convert to TFLite format

Mobile devices have significant limitations, so any pre-processing that can be done to reduce an app's footprint is worth considering. With TFLite a new graph converter is now included with the TensorFlow installation. This program is called the "TensorFlow Lite Optimizing Converter" or TOCO.

TFLite uses a different serialization format from regular TensorFlow. TensorFlow uses Protocol Buffers, while TFLite uses FlatBuffers because.

git clone https://github.com/tensorflow/tensorflow.git

cd tensorflow

IMAGE_SIZE=224

bazel run --config=opt \

  //tensorflow/contrib/lite/toco:toco -- \

  --input_file=tf_files/retrained_graph.pb \

  --output_file=tf_files/optimized_graph.lite \

  --input_format=TENSORFLOW_GRAPHDEF \

  --output_format=TFLITE \

  --input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3 \

  --input_array=input \

  --output_array=final_result \

  --inference_type=FLOAT \

  --input_type=FLOAT

### Run the customized app

Add your model files to the project

The demo project is configured to search for a graph.pb, and a labels.txt files in the android/tflite/app/src/main/assets/ directory.

## Create a Mobile App

### Optimize for inference

python -m tensorflow.python.tools.optimize_for_inference \

  --input=tf_files/retrained_graph.pb \

  --output=tf_files/optimized_graph.pb \

  --input_names="input" \

  --output_names="final_result"


### Quantize the network weights

python -m scripts.quantize_graph \

  --input=tf_files/optimized_graph.pb \

  --output=tf_files/rounded_graph.pb \

  --output_node_names=final_result \

  --mode=weights_rounded

### Change the output_name in ClassifierActivity.java

private static final String INPUT_NAME = "input";

private static final String OUTPUT_NAME = "final_result";