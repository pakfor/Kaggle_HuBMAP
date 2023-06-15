import tensorflow as tf
from tensorflow.keras import layers

class UNet():
    def __init__(self, output_dim):
        self.output_dim = output_dim

    def custom_u_net_conv_block(self, x, unit=64, ker_size=3, stride=1, padding='same', batch_norm=True, repeat=2):
        for i in range(0, repeat):
            x = layers.Conv2D(unit, ker_size, stride, padding=padding)(x)
            x = layers.ReLU()(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
        return x

    def build(self):
        input = layers.Input(shape=(512,512,3))

        x = layers.Conv2D(64,3,1,padding='same')(input)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = self.custom_u_net_conv_block(x, unit=64, repeat=1)
        a1 = x
        x = layers.MaxPooling2D(2,2)(a1)

        x = self.custom_u_net_conv_block(x, unit=128, repeat=2)
        a2 = x
        x = layers.MaxPooling2D(2,2)(a2)

        x = self.custom_u_net_conv_block(x, unit=256, repeat=2)
        a3 = x
        x = layers.MaxPooling2D(2,2)(a3)

        x = self.custom_u_net_conv_block(x, unit=512, repeat=2)
        a4 = x
        x = layers.MaxPooling2D(2,2)(a4)

        x = self.custom_u_net_conv_block(x, unit=1024, repeat=2)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2DTranspose(128,1,2,padding='same')(x)

        x = layers.concatenate([x,a4],axis=-1)
        x = self.custom_u_net_conv_block(x, unit=512, repeat=2)

        x = layers.Conv2DTranspose(64,1,2,padding='same')(x)

        x = layers.concatenate([x,a3],axis=-1)
        x = self.custom_u_net_conv_block(x, unit=256, repeat=2)

        x = layers.Conv2DTranspose(32,1,2,padding='same')(x)

        x = layers.concatenate([x,a2],axis=-1)
        x = self.custom_u_net_conv_block(x, unit=128, repeat=2)

        x = layers.Conv2DTranspose(16,1,2,padding='same')(x)

        x = layers.concatenate([x,a1],axis=-1)
        x = self.custom_u_net_conv_block(x, unit=64, repeat=2)

        x = layers.Conv2D(self.output_dim,1,1,padding='same', activation = 'sigmoid', dtype='float32')(x)

        model = tf.keras.Model(input,x)
        model.summary()
        return model