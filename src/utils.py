import tensorflow as tf

# Check for GPU availability
def check_gpu():
    gpu_available = tf.config.list_physical_devices('GPU')
    print("GPU Available:", gpu_available)
    return gpu_available
