
�?root"_tf_keras_network*�?{"name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_1", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_1", "inbound_nodes": [["input_1", 0, 0, {"y": ["input_2", 0, 0], "name": null}]]}, {"class_name": "Dense", "config": {"name": "HiddenLayer1", "trainable": true, "dtype": "float32", "units": 1800, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "HiddenLayer1", "inbound_nodes": [[["tf.math.subtract_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Dropout1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "Dropout1", "inbound_nodes": [[["HiddenLayer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "HiddenLayer2", "trainable": true, "dtype": "float32", "units": 600, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "HiddenLayer2", "inbound_nodes": [[["Dropout1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "Dropout2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "Dropout2", "inbound_nodes": [[["HiddenLayer2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["Dropout2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "shared_object_id": 16, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1200]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1200]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1200]}, {"class_name": "TensorShape", "items": [null, 1200]}], "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1200]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1200]}, "float32", "input_2"]}]], {}]}, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1200]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1200]}, "float32", "input_2"]}], "keras_version": "2.8.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_1", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "name": "tf.math.subtract_1", "inbound_nodes": [["input_1", 0, 0, {"y": ["input_2", 0, 0], "name": null}]], "shared_object_id": 2}, {"class_name": "Dense", "config": {"name": "HiddenLayer1", "trainable": true, "dtype": "float32", "units": 1800, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "HiddenLayer1", "inbound_nodes": [[["tf.math.subtract_1", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dropout", "config": {"name": "Dropout1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "Dropout1", "inbound_nodes": [[["HiddenLayer1", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "HiddenLayer2", "trainable": true, "dtype": "float32", "units": 600, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 10}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "HiddenLayer2", "inbound_nodes": [[["Dropout1", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dropout", "config": {"name": "Dropout2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "name": "Dropout2", "inbound_nodes": [[["HiddenLayer2", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["Dropout2", 0, 0, {}]]], "shared_object_id": 15}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["output_layer", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "binary_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 19}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1200]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}2
�root.layer-1"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1200]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1200]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}2
�root.layer-2"_tf_keras_layer*�{"name": "tf.math.subtract_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "TFOpLambda", "config": {"name": "tf.math.subtract_1", "trainable": true, "dtype": "float32", "function": "math.subtract"}, "inbound_nodes": [["input_1", 0, 0, {"y": ["input_2", 0, 0], "name": null}]], "shared_object_id": 2}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "HiddenLayer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "HiddenLayer1", "trainable": true, "dtype": "float32", "units": 1800, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 5}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["tf.math.subtract_1", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1200}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1200]}}2
�root.layer-4"_tf_keras_layer*�{"name": "Dropout1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "Dropout1", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "inbound_nodes": [[["HiddenLayer1", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1800]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "HiddenLayer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "HiddenLayer2", "trainable": true, "dtype": "float32", "units": 600, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.009999999776482582}, "shared_object_id": 10}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Dropout1", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1800}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1800]}}2
�root.layer-6"_tf_keras_layer*�{"name": "Dropout2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "Dropout2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}, "inbound_nodes": [[["HiddenLayer2", 0, 0, {}]]], "shared_object_id": 12, "build_input_shape": {"class_name": "TensorShape", "items": [null, 600]}}2
�root.layer_with_weights-2"_tf_keras_layer*�{"name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Dropout2", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 600}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 600]}}2
�_root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 23}2
�`root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "binary_accuracy", "dtype": "float32", "config": {"name": "binary_accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 19}2