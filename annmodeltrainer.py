import tensorflow as tf
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Input 
from tensorflow.keras.callbacks import EarlyStopping


data = pd.read_csv('corona.csv', low_memory=False)
data = data.drop(['Ind_ID', 'Test_date' ], axis=1)
data = data[data['Corona'] != 'other']
data = data[data['Known_contact'] != 'other']
data = data.dropna(subset=['Age_60_above'])
data = data.dropna(subset=['Sex'])

def pickle_saver(column_name,encoder):
    with open(column_name+'encoder.pkl' , 'wb') as file:
        pickle.dump(encoder ,file)

def labelencoder(column_name , data):
    label_encoder = LabelEncoder()
    data[column_name] = label_encoder.fit_transform(data[column_name])
    pickle_saver(column_name,label_encoder)
    return data
    
data = labelencoder('Cough_symptoms',data)
data = labelencoder('Fever',data)
data = labelencoder('Sore_throat',data)
data = labelencoder('Shortness_of_breath',data)
data = labelencoder('Headache',data)
data = labelencoder('Corona',data)
data = labelencoder('Age_60_above',data)
data = labelencoder('Sex',data)
data = labelencoder('Known_contact',data)

x = data.drop('Corona',axis=1)
y = data['Corona']

x_train ,x_test ,y_train ,y_test = train_test_split(x,y ,test_size=.2 , random_state=42)

scaller = StandardScaler()
x_train = scaller.fit_transform(x_train)
x_test = scaller.transform(x_test)
pickle_saver('scaller',scaller)

model =Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(64 , activation='relu'),
    Dense(8, activation='relu'),
    Dense(1 , activation='sigmoid')
])

opt= tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt ,loss='binary_crossentropy', metrics=['accuracy'])
early_stoping_callback = EarlyStopping(monitor = 'val_loss' , patience = 5, restore_best_weights=True)
history = model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=100,callbacks=[early_stoping_callback])
model.save('covid_prediction.keras')