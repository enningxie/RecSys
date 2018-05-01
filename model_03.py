# mlp-attention_02
# {'val_loss': [
# 0.8408573621920321, 0.8113642657271882, 0.8025012318207997, 0.7935497788274474, 0.7879389109668815,
# 0.7814277308461638, 0.775492575915156, 0.7743384593365328, 0.775747516258671, 0.7749475723695656,
# 0.783225849018001, 0.7771770018869348, 0.7760210549038458, 0.7781160926205287, 0.779945488108774,
# 0.7801610001767221, 0.7822167160553423, 0.7815052622447096, 0.7841386477906546, 0.7827288544009602
# ],
# 'val_mean_squared_error': [
# 0.8408573621920321, 0.8113642657271882, 0.8025012318207997, 0.7935497788274474, 0.7879389109668815,
# 0.7814277308461638, 0.775492575915156, 0.7743384593365328, 0.775747516258671, 0.7749475723695656,
# 0.783225849018001, 0.7771770018869348, 0.7760210549038458, 0.7781160926205287, 0.779945488108774,
# 0.7801610001767221, 0.7822167160553423, 0.7815052622447096, 0.7841386477906546, 0.7827288544009602
# ],
# 'loss': [
# 1.1535629132283833, 0.8126167149279035, 0.7889490043907195, 0.7750283503554996, 0.7654959220210497,
# 0.754737268504087, 0.7444609114043512, 0.7372140922256852, 0.7314677306859819, 0.7261646949399454,
# 0.7214236349499802, 0.7168029387634678, 0.7122004911905905, 0.7079394369758819, 0.7038796790663087,
# 0.6996286518199817, 0.6954886699031319, 0.6911364870675247, 0.6871858599231481, 0.6829559556120018
# ],
# 'mean_squared_error': [
# 1.1535629132283833, 0.8126167149279035, 0.7889490043907195, 0.7750283503554996, 0.7654959220210497,
# 0.754737268504087, 0.7444609114043512, 0.7372140922256852, 0.7314677306859819, 0.7261646949399454,
# 0.7214236349499802, 0.7168029387634678, 0.7122004911905905, 0.7079394369758819, 0.7038796790663087,
# 0.6996286518199817, 0.6954886699031319, 0.6911364870675247, 0.6871858599231481, 0.6829559556120018
# ]}
# 0.7743384593365328
# 0.8799



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




