import tensorflow as tf

a = tf.keras.utils.to_categorical([1, 2, 3], num_classes=3)
print(a)