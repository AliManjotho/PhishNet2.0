from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, ReLU, Softmax, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.initializers import he_normal

class PhishingModel(Model):

    def __init__(self, masked_outputs=True):
        super(PhishingModel, self).__init__()

        self.alpha_layer_1 = Dense(1024, kernel_initializer=he_normal)
        self.alpha_layer_2 = ReLU()
        self.alpha_layer_3 = Dropout(120)
        self.alpha_layer_4 = BatchNormalization()
        self.alpha_layer_5 = Dense(256, kernel_initializer=he_normal)
        self.alpha_layer_6 = ReLU()
        self.alpha_layer_7 = BatchNormalization()
        self.alpha_layer_8 = Dense(128, kernel_initializer=he_normal)
        self.alpha_layer_9 = ReLU()
        self.alpha_layer_10 = Dropout(32)
        self.alpha_layer_11 = Dense(64, kernel_initializer=he_normal)
        self.alpha_layer_12 = ReLU()
        self.alpha_layer_13 = Dropout(16)
        self.alpha_layer_14 = BatchNormalization()
        self.alpha_layer_15 = Dense(2, kernel_initializer=he_normal)
        self.alpha_layer_16 = Softmax()

        self.beta_layer_1 = Dense(256, kernel_initializer=he_normal)
        self.beta_layer_2 = ReLU()
        self.beta_layer_3 = Dropout(64)
        self.beta_layer_4 = BatchNormalization()
        self.beta_layer_5 = Dense(64, kernel_initializer=he_normal)
        self.beta_layer_6 = ReLU()
        self.beta_layer_7 = Dropout(16)
        self.beta_layer_8 = BatchNormalization()
        self.beta_layer_9 = Dense(64, kernel_initializer=he_normal)
        self.beta_layer_10 = ReLU()
        self.beta_layer_11 = Dense(16, kernel_initializer=he_normal)
        self.beta_layer_12 = ReLU()
        self.beta_layer_13 = Dropout(4)
        self.beta_layer_14 = BatchNormalization()
        self.beta_layer_15 = Dense(2, kernel_initializer=he_normal)
        self.beta_layer_16 = Softmax()

        self.gamma_layer_1 = Dense(256, kernel_initializer=he_normal)
        self.gamma_layer_2 = ReLU()
        self.gamma_layer_3 = Dropout(64)
        self.gamma_layer_4 = Dense(64, kernel_initializer=he_normal)
        self.gamma_layer_5 = ReLU()
        self.gamma_layer_6 = Dense(64, kernel_initializer=he_normal)
        self.gamma_layer_7 = ReLU()
        self.gamma_layer_8 = Dense(16, kernel_initializer=he_normal)
        self.gamma_layer_9 = ReLU()
        self.gamma_layer_10 = Dense(2, kernel_initializer=he_normal)
        self.gamma_layer_11 = Softmax()

        self.alpha_beta_concat = Concatenate(axis=1)


    def call(self, inputs):
        x = inputs[0:55]
        x = self.alpha_layer_1(x)
        x = self.alpha_layer_2(x)
        x = self.alpha_layer_3(x)
        x = self.alpha_layer_4(x)
        x = self.alpha_layer_5(x)
        x = self.alpha_layer_6(x)
        x = self.alpha_layer_7(x)
        x = self.alpha_layer_8(x)
        x = self.alpha_layer_9(x)
        x = self.alpha_layer_10(x)
        x = self.alpha_layer_11(x)
        x = self.alpha_layer_12(x)
        x = self.alpha_layer_13(x)
        x = self.alpha_layer_14(x)
        x = self.alpha_layer_15(x)
        alpha_out = self.alpha_layer_16(x)

        x = inputs[56:87]
        x = self.beta_layer_1(x)
        x = self.beta_layer_2(x)
        x = self.beta_layer_3(x)
        x = self.beta_layer_4(x)
        x = self.beta_layer_5(x)
        x = self.beta_layer_6(x)
        x = self.beta_layer_7(x)
        x = self.beta_layer_8(x)
        x = self.beta_layer_9(x)
        x = self.beta_layer_10(x)
        x = self.beta_layer_11(x)
        x = self.beta_layer_12(x)
        x = self.beta_layer_13(x)
        x = self.beta_layer_14(x)
        x = self.beta_layer_15(x)
        beta_out = self.beta_layer_16(x)

        alpha_beta_out = self.alpha_beta_concat([alpha_out, beta_out, inputs])

        x = alpha_beta_out
        x = self.gamma_layer_1(x)
        x = self.gamma_layer_2(x)
        x = self.gamma_layer_3(x)
        x = self.gamma_layer_4(x)
        x = self.gamma_layer_5(x)
        x = self.gamma_layer_6(x)
        x = self.gamma_layer_7(x)
        x = self.gamma_layer_8(x)
        x = self.gamma_layer_9(x)
        x = self.gamma_layer_10(x)
        gamma_out = self.gamma_layer_11(x)

        return alpha_out, beta_out, gamma_out





