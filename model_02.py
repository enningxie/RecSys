# mlp_attention
# {'val_loss': [
# 0.8163300458284443, 0.7987896355853377, 0.7839253415701084, 0.7732363951680441,
# 0.7681095451553074, 0.7664300101587679, 0.7643328860262513, 0.7635605529368878,
# 0.7653752291444846, 0.7689169669360832, 0.7732669168484914, 0.7873278110231666,
# 0.7886072771500431, 0.7983797422110516, 0.805395925250368, 0.8150209715744392,
# 0.8221210376574007, 0.8306424452357457, 0.8347012218066645, 0.8472284756017524
# ],
# 'val_mean_squared_error': [
# 0.8163300458284443, 0.7987896355853377, 0.7839253415701084, 0.7732363951680441,
# 0.7681095451553074, 0.7664300101587679, 0.7643328860262513, 0.7635605529368878,
# 0.7653752291444846, 0.7689169669360832, 0.7732669168484914, 0.7873278110231666,
# 0.7886072771500431, 0.7983797422110516, 0.805395925250368, 0.8150209715744392,
# 0.8221210376574007, 0.8306424452357457, 0.8347012218066645, 0.8472284756017524
# ],
# 'loss': [
# 1.1665671023080197, 0.7919197071753454, 0.763886582989932, 0.7402164054533085,
# 0.720628759692561, 0.7016330183313703, 0.6817957864113541, 0.6599540514646366,
# 0.6386242118971979, 0.6188564913261181, 0.6003975062373864, 0.5840127102886384,
# 0.5681901097745802, 0.5540921113833704, 0.5414115025206144, 0.5296459047524005,
# 0.5191614477442308, 0.5094999734682331, 0.5005742947450886, 0.4923522201838196
# ],
# 'mean_squared_error': [
# 1.1665671023080197, 0.7919197071753454, 0.763886582989932, 0.7402164054533085,
# 0.720628759692561, 0.7016330183313703, 0.6817957864113541, 0.6599540514646366,
# 0.6386242118971979, 0.6188564913261181, 0.6003975062373864, 0.5840127102886384,
# 0.5681901097745802, 0.5540921113833704, 0.5414115025206144, 0.5296459047524005,
# 0.5191614477442308, 0.5094999734682331, 0.5005742947450886, 0.4923522201838196
# ]}
# 0.7635605529368878
# 0.8738

from surprise import Dataset
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, Flatten, multiply, regularizers
from keras import optimizers, losses, metrics
import numpy as np


def get_data():
    data = Dataset.load_builtin('ml-100k')
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
        attention_probs = Dense(64, activation='softmax', name='attention_vec')(u_i_concat)
        attention_mul = multiply([u_i_concat, attention_probs], name='attention_mul')
        dense_1 = Dense(64 , activation='relu', name='dense_1_op')(attention_mul)
        dense_2 = Dense(32, activation='relu', name='dense_2_op')(dense_1)
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
            epochs=30,
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
    best_index = 0
    best_result = 1.0
    for i, result in enumerate(mlp_model.his.history['val_mean_squared_error']):
        if best_result > result:
            best_result = result
            best_index = i
    print("best_index: {0}, best_result: {1}.".format(best_index, np.sqrt(best_result)))
    pred = mlp_model.model.predict([np.asarray([0]), np.asarray([0])])
    print(pred)




