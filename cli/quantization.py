import tensorflow as tf

model_path = "./models/explosion_model"

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()
print(len(tflite_model))
print("hellomoto")

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quant_model = converter.convert()
print(f"{len(tflite_model)} vs {len(tflite_quant_model)}")

with open("./models/explosion_quant_model.tflite", "wb") as file:
    file.write(tflite_quant_model)
