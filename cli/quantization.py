import tensorflow as tf

# models = ["Arson", "Explosion", "RoadAccidents", "Shooting", "Vandalism", "Fighting"]
models = ["Fighting"]
prefix = "./models/"

def transformer(model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    
    tflite_model = converter.convert()

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_quant_model = converter.convert()
    print(f"{len(tflite_model)} vs {len(tflite_quant_model)}")
    
    with open(f"{model_path}.tflite", "wb") as file:
        file.write(tflite_quant_model)

if __name__ == "__main__":
    for model in models:
        model_path = prefix + model
        transformer(model_path)
