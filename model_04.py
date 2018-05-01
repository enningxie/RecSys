# mlp-attention_03
# {'val_loss': [
# 0.8321127831733865, 0.8149616334763244, 0.8080661035819032, 0.8043228179784596, 0.8002183685173205,
# 0.7972857836676045, 0.793001569427853, 0.7924909000240787, 0.7921270519020359, 0.7874714549603654,
# 0.7854632683025837, 0.7836462063538413, 0.7816860973027518, 0.7811260843474584, 0.7826188791772042,
# 0.7800090469637393, 0.7793066135596607, 0.7784056678202643, 0.7783930039180803, 0.7800617894507442
# ],
# 'val_mean_squared_error': [
# 0.8321127831733865, 0.8149616334763244, 0.8080661035819032, 0.8043228179784596, 0.8002183685173205,
# 0.7972857836676045, 0.793001569427853, 0.7924909000240787, 0.7921270519020359, 0.7874714549603654,
# 0.7854632683025837, 0.7836462063538413, 0.7816860973027518, 0.7811260843474584, 0.7826188791772042,
# 0.7800090469637393, 0.7793066135596607, 0.7784056678202643, 0.7783930039180803, 0.7800617894507442
# ],
# 'loss': [
# 1.2129996511072807, 0.8089207498588716, 0.7925854784378688, 0.7837273693483895, 0.7770178057682471,
# 0.7712283459233497, 0.7657942494154589, 0.7617266936164807, 0.7575358347528671, 0.7536763259070509,
# 0.7502327760970651, 0.746288952394418, 0.7429369581147989, 0.7393621096899098, 0.7362319128615927,
# 0.73318466221791, 0.7307932809421275, 0.7283218859352288, 0.726398458615269, 0.7244027269489683
# ],
# 'mean_squared_error': [
# 1.2129996511072807, 0.8089207498588716, 0.7925854784378688, 0.7837273693483895, 0.7770178057682471,
# 0.7712283459233497, 0.7657942494154589, 0.7617266936164807, 0.7575358347528671, 0.7536763259070509,
# 0.7502327760970651, 0.746288952394418, 0.7429369581147989, 0.7393621096899098, 0.7362319128615927,
# 0.73318466221791, 0.7307932809421275, 0.7283218859352288, 0.726398458615269, 0.7244027269489683
# ]}
# 0.7800090469637393
# 0.8832



from surprise import Dataset
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, multiply
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
        attention_u = Dense(32, activation='softmax', name='attention_vec_u')(u_flatten)
        attention_u_mul = multiply([u_flatten, attention_u], name='attention_mul_u')
        attention_i = Dense(32, activation='softmax', name='attention_vec_i')(i_flatten)
        attention_i_mul = multiply([i_flatten, attention_i], name='attention_mul_i')
        u_i_concat = concatenate([attention_u_mul, attention_i_mul], name='concat_op')
        attention_u_i = Dense(64, activation='softmax', name='attention_vec_u_i')(u_i_concat)
        attention_u_i_mul = multiply([u_i_concat, attention_u_i], name='attention_mul_u_i')
        dense_1 = Dense(32, activation='relu', name='dense_1_op')(attention_u_i_mul)
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




