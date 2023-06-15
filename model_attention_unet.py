import tensorflow as tf
from tensorflow.keras import layers

class AttentionUNet():
    def conv_block(self, x, num_filters):
        x = layers.Conv2D(num_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(num_filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    def encoder_block(self, x, num_filters):
        x = self.conv_block(x, num_filters)
        p = layers.MaxPool2D((2, 2))(x)
        return x, p

    def attention_gate(self, g, s, num_filters):
        Wg = layers.Conv2D(num_filters, 1, padding="same")(g)
        Wg = layers.BatchNormalization()(Wg)

        Ws = layers.Conv2D(num_filters, 1, padding="same")(s)
        Ws = layers.BatchNormalization()(Ws)

        out = layers.Activation("relu")(Wg + Ws)
        out = layers.Conv2D(num_filters, 1, padding="same")(out)
        out = layers.Activation("sigmoid")(out)
        return out * s

    def decoder_block(self, x, s, num_filters):
        x = layers.UpSampling2D(interpolation="bilinear")(x)
        s = self.attention_gate(x, s, num_filters)
        x = layers.Concatenate()([x, s])
        x = self.conv_block(x, num_filters)
        return x

    def __init__(self, input_shape, output_dim):
        self.input_shape = input_shape
        self.output_dim = output_dim

    def build(self):
        """ Inputs """
        inputs = layers.Input(self.input_shape)

        """ Encoder """
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)

        b1 = self.conv_block(p3, 512)

        """ Decoder """
        d1 = self.decoder_block(b1, s3, 256)
        d2 = self.decoder_block(d1, s2, 128)
        d3 = self.decoder_block(d2, s1, 64)

        """ Outputs """
        outputs = layers.Conv2D(self.output_dim, 1, padding="same", activation="sigmoid", dtype='float32')(d3)

        """ Model """
        model = tf.keras.Model(inputs, outputs, name="Attention-UNET")
        model.summary()
        return model