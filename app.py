import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
data=pd.read_csv("AimilData.csv")
# data.drop(['isFraud', 'isFlaggedFraud'], axis=1, inplace=True)
@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    transid = request.form['steps']
    X = data[data.nameOrig == transid]
    X.drop(['nameOrig', 'nameDest', 'isFraud'], axis=1, inplace=True)
    X = np.array(X)
    # print(X.dtypes)
    # print(X.shape)
    X.reshape
    prediction = model.predict([X])

    return render_template('form.html', prediction_text='Employee Salary should be $ {}'.format(X.dtypes))


if __name__ == "__main__":
    app.run(debug=True)
    *******************************************************************************************************************************************
    import pickle
import pandas as pd
import numpy as np
data=pd.read_csv("AimilData.csv")
humi = request.form['steps']
X = data[data.nameOrig == humi]
X = np.array(X)
a = X[5]
# data.drop(['isFraud','isFlaggedFraud'],axis=1,inplace=True)
# transid = 'C1231006815'
transid = input()
X=data[data.nameOrig==transid]
X.drop(['nameOrig','nameDest','isFraud'],axis=1,inplace=True)
# print(X.dtypes)
# print(X.shape)
print(X[0])
# prediction = model.predict(X)
# print(prediction)
# data.head()
