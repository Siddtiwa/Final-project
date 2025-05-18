from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.recurrent import LSTM, GRU
from tensorflow.python.keras.layers import Dense, Dropout

def build_model(model_type, input_shape, units):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units))
    else:
        model.add(GRU(units, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(units))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model