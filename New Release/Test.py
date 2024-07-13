import tensorflow as tf

# Check if GPU is available
if tf.test.is_gpu_available():
    print("GPU is available")
else:
    print("GPU is NOT available")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate memory on demand
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Physical GPUs Available: ", len(gpus))
else:
    print("No GPU detected")

# Example TensorFlow code
with tf.device('/GPU:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

print("Result of matrix multiplication:")
print(c)
