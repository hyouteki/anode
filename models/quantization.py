import tensorflow as tf

models = ["Arson", "Explosion", "RoadAccidents", "Shooting", "Vandalism"]
prefix = "./OF-CNNLSTM/"

def transformer(modelPath):
    converter = tf.lite.TFLiteConverter.from_saved_model(modelPath)    
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]    
    tfliteQuantModel = converter.convert()
    
    with open(f"{modelPath}.tflite", "wb") as file:
        file.write(tfliteQuantModel)

if __name__ == "__main__":
    for model in models:
        modelPath = prefix + model
        transformer(modelPath)
