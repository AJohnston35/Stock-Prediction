from collections import deque
import random
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Flatten, RepeatVector, Permute, Multiply
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import mysql.connector

# Connect to the database
endpoint = 'stockaide-db.cluster-czcooo8241qk.us-east-2.rds.amazonaws.com'
username = 'admin'
password = 'mnmsstar15'
database = 'stockaide'

engine = create_engine(f'mysql+mysqlconnector://{username}:{password}@{endpoint}/{database}', echo=False)
conn = mysql.connector.connect(user=username, password=password,host=endpoint, database=database)
cursor = conn.cursor()

# Retrieve stock data from database
sql_query = '''SELECT * FROM subset_data WHERE Date < '2022-01-01' '''
cursor.execute(sql_query)
result = cursor.fetchall()
train_df = pd.DataFrame(result, columns=cursor.column_names)

sql_query = '''SELECT * FROM subset_data WHERE Date >= '2022-01-01' '''
cursor.execute(sql_query)
result = cursor.fetchall()
test_df = pd.DataFrame(result, columns=cursor.column_names)

# Separate the dataframes by ticker
train_dfs = []
for ticker in train_df['Ticker'].unique():
    train_dfs.append(train_df[train_df['Ticker'] == ticker])
    
test_dfs = []
for ticker in test_df['Ticker'].unique():
    test_dfs.append(test_df[test_df['Ticker'] == ticker])
    
# Filter data to only include necessary columns
columns = ['Open','High','Low','Close', 'Volume', 'Weighted_avg', 
           'DFF', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi',
           'volume_em', 'volume_sma_em', 'volume_vpt', 'volume_vwap', 
           'volume_mfi', 'volume_nvi', 'volatility_bbm', 'volatility_bbh', 
           'volatility_bbl', 'volatility_bbw', 'volatility_bbp', 
           'volatility_bbhi', 'volatility_bbli', 'volatility_kcc', 
           'volatility_kch', 'volatility_kcl', 'volatility_kcw', 
           'volatility_kcp', 'volatility_kchi', 'volatility_kcli', 
           'trend_sma_slow', 'trend_sma_fast', 'trend_ema_slow', 
           'trend_ema_fast', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 
           'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_diff', 
           'trend_trix', 'trend_mass_index', 'trend_cci', 'trend_dpo', 
           'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_a', 
           'trend_ichimoku_b', 'trend_visual_ichimoku_a', 
           'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down', 
           'trend_aroon_ind', 'momentum_rsi', 'momentum_tsi', 'momentum_uo', 
           'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 
           'momentum_ao', 'momentum_kama', 'momentum_roc', 'others_dr', 
           'others_dlr', 'others_cr', 'Adjusted_close_returns', 'Adjusted_close']

train_dfs = [df[columns] for df in train_dfs]
test_dfs = [df[columns] for df in test_dfs]

# Convert data to numpy arrays
train_data = [df.to_numpy() for df in train_dfs]
test_data = [df.to_numpy() for df in test_dfs]

# Scale the data
scalers = [MinMaxScaler() for _ in range(len(train_data))]
train_data_scaled = [scaler.fit_transform(data) for scaler, data in zip(scalers, train_data)]
test_data_scaled = [scaler.transform(data) for scaler, data in zip(scalers, test_data)]

# Create the sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

x_train, y_train = [], []
for data in train_data_scaled:
    X, y = create_sequences(data, 30)
    x_train.append(X)
    y_train.append(y)

x_test, y_test = [], []
for data in test_data_scaled:
    X, y = create_sequences(data, 30)
    x_test.append(X)
    y_test.append(y)

# Create the LSTM model
class Model:
    def __init__ (self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
    
    def build_model(self):
        
        inputs = Input(shape=(self.state_size, 1))
        
        lstm_out = LSTM(64, return_sequences=True)(inputs)
        
        attention = TimeDistributed(Dense(1, activation='tanh'))(lstm_out)
        attention = Flatten()(attention)
        attention = Dense(self.state_size, activation='softmax')(attention)
        attention = RepeatVector(64)(attention)
        attention = Permute([2, 1])(attention)
        
        sent_representation = Multiply()([lstm_out, attention])
        sent_representation = Flatten()(sent_representation)
        
        dense1 = Dense(32, activation='relu')(sent_representation)
        dense2 = Dense(8, activation='relu')(dense1)
        
        output = Dense(self.action_size, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model

# Create the Q-learning agent       
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(next_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)
        
def train_models(x_train, y_train, x_test, y_test):
    models = []
    for i in range(len(x_train)):
        model = Model(30, 1).build_model()
        agent = DQNAgent(30, 1)
        for j in range(100):
            state = x_train[i]
            for k in range(len(state)):
                action = agent.act(state[k].reshape(1, 30, 1))
                next_state = state[k+1].reshape(1, 30, 1)
                reward = y_train[i][k]
                done = True if k == len(state) - 1 else False
                agent.remember(state[k].reshape(1, 30), action, reward, next_state, done)
                state = next_state
                if done:
                    print(f'Training model {i+1}/{len(x_train)}: Episode {j+1}, Reward: {reward}')
                    break
            if len(agent.memory) > 32:
                agent.replay(32)
        models.append(agent)
    return models
