# mlp
# {'val_loss': [
# 0.8274254393281284, 0.8053291361300785, 0.7859281158888486, 0.7825269175061362, 0.7822666343532938,
# 0.7825471893008191, 0.7806398462219931, 0.7833374913989448, 0.7835199509313114, 0.7958879216652841,
# 0.794692136500967, 0.8017291041461059, 0.8098521981832999, 0.8141202649758785, 0.8198717381744882,
# 0.8279802773999505, 0.8414359575271597, 0.8396208376656866, 0.8445005035683095, 0.8511163710873269
# ],
# 'val_mean_squared_error': [
# 0.8274254393281284, 0.8053291361300785, 0.7859281158888486, 0.7825269175061362, 0.7822666343532938,
# 0.7825471893008191, 0.7806398462219931, 0.7833374913989448, 0.7835199509313114, 0.7958879216652841,
# 0.794692136500967, 0.8017291041461059, 0.8098521981832999, 0.8141202649758785, 0.8198717381744882,
# 0.8279802773999505, 0.8414359575271597, 0.8396208376656866, 0.8445005035683095, 0.8511163710873269
# ],
# 'loss': [
# 1.1144618829389996, 0.8009950357983264, 0.7715282297317705, 0.7488418974234977, 0.7319394611718272,
# 0.7145572536125903, 0.6952577738607248, 0.6761519667335364, 0.6582879601022921, 0.6415560948076215,
# 0.6254914589686456, 0.6108845931735103, 0.5969660604467011, 0.5842290976573724, 0.5730708022449436,
# 0.5623907744139961, 0.5526787808392529, 0.5439159440058099, 0.535404206367364, 0.5279838359120821
# ],
# 'mean_squared_error': [
# 1.1144618829389996, 0.8009950357983264, 0.7715282297317705, 0.7488418974234977, 0.7319394611718272,
# 0.7145572536125903, 0.6952577738607248, 0.6761519667335364, 0.6582879601022921, 0.6415560948076215,
# 0.6254914589686456, 0.6108845931735103, 0.5969660604467011, 0.5842290976573724, 0.5730708022449436,
# 0.5623907744139961, 0.5526787808392529, 0.5439159440058099, 0.535404206367364, 0.5279838359120821
# ]}
# 0.7806398462219931
# 0.8835

from surprise import Dataset
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten
from keras import optimizers, losses, metrics
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
        u_embedding = Embedding(input_dim=self.users, output_dim=32, input_length=1, name='u_embedding_op')(u_inputs)
        i_embedding = Embedding(input_dim=self.items, output_dim=32, input_length=1, name='i_embedding_op')(i_inputs)
        u_flatten = Flatten(name='u_flatten_op')(u_embedding)
        i_flatten = Flatten(name='i_flatten_op')(i_embedding)
        u_i_concat = concatenate([u_flatten, i_flatten], name='concat_op')
        dense_1 = Dense(32, activation='relu', name='dense_1_op')(u_i_concat)
        dense_2 = Dense(16, activation='relu', name='dense_2_op')(dense_1)
        dense_3 = Dense(8, activation='relu', name='dense_3_op')(dense_2)
        output = Dense(1, name='output_op')(dense_3)
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
            epochs=5,
            batch_size=256
        )

    def predict(self):
        pred = self.model.predict([np.asarray([0]), np.asarray([0])])
        return pred

    def build(self):
        self.data()
        self.inference()
        self.compile()
        self.train()


if __name__ == '__main__':
    mlp_model = Mlp()
    mlp_model.build()
    pred = mlp_model.predict()
    print(pred)




