# mf
from surprise import Dataset
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, multiply
from keras import optimizers, losses, metrics
import keras.backend as K
import numpy as np


def get_data():
    data = Dataset.load_builtin('ml-1m')
    data_set = data.build_full_trainset()
    # print(data_set.n_items)
    # print(data_set.n_users)
    x = []
    y = []
    for u, i, r in data_set.all_ratings():
        x.append((u, i))
        y.append(r)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


class Mlp:
    def __init__(self):
        self.items = 3706
        self.users = 6040

    def data(self):
        X_train, X_test, self.y_train, self.y_test = get_data()
        self.X_train_u = []
        self.X_train_i = []
        self.X_test_u = []
        self.X_test_i = []
        for u, i in X_train:
            self.X_train_u.append(u)
            self.X_train_i.append(i)

        for u, i in X_test:
            self.X_test_u.append(u)
            self.X_test_i.append(i)

    def inference(self):
        u_inputs = Input(shape=(1,), name='u_input_op')
        i_inputs = Input(shape=(1,), name='i_input_op')
        u_embedding = Embedding(input_dim=self.users, output_dim=8, input_length=1, name='u_embedding_op')(u_inputs)
        i_embedding = Embedding(input_dim=self.items, output_dim=8, input_length=1, name='i_embedding_op')(i_inputs)
        u_flatten = Flatten()(u_embedding)
        i_flatten = Flatten()(i_embedding)
        u_i = multiply([u_flatten, i_flatten])
        output = Dense(1)(u_i)
        self.model = Model(inputs=[u_inputs, i_inputs], outputs=output)

    def compile(self):
        self.model.summary()
        self.model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.mean_squared_error,
            metrics=[metrics.mean_squared_error]
        )

    def train(self):
        self.his = self.model.fit(
            x=[np.asarray(self.X_train_u), np.asarray(self.X_train_i)],
            y=np.asarray(self.y_train),
            validation_data=([np.asarray(self.X_test_u), np.asarray(self.X_test_i)], np.asarray(self.y_test)),
            epochs=20,
            batch_size=256
        )

    def predict(self):
        self.model.predict()

    def build(self):
        self.data()
        self.inference()
        self.compile()
        self.train()


if __name__ == '__main__':
    mlp_model = Mlp()
    mlp_model.build()
    print(mlp_model.his.history)
    print(mlp_model.model.predict([np.asarray([0]), np.asarray([0])]))




