from flask import Flask
from flask import request

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

app = Flask(__name__)

# Loading Dataset
data = pd.read_csv('data.csv')
Y = data.iloc[:, 0]
X = data.iloc[:, 1:]

# Conditions for Fuzzy Logic
matching = {
   "Left" : ["Up", "Down"],
   "Right" : ["Up", "Down"],
   "Up" : ["Left", "Right"],
   "Down": ["Left", "Right"]
}

# Training Artificial Neural Network
# model = MLPClassifier(solver='adam')
# model.fit(X, Y)


def train_ANN(a, b):
   return MLPClassifier(solver="adam").fit(a, b)

model = train_ANN(X, Y)

### Routes ###

@app.route('/')
def home():
   return 'Decision Making System'

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
   if request.method == 'POST':
      x, y, z = request.json['X'], request.json['Y'], request.json['Distance']
      df = pd.DataFrame([[x, y, z]], columns=['X', 'Y', 'Distance'])

      final = ""

      # Condition for No Action
      if x**2 + y**2 + z**2 > 75**2:
         final = "Do Nothing"
      
      # When Object is too close
      elif x**2 + y**2 + z**2 <= 50**2:
         final = model.predict(df)[0] + " : 100%."
      
      # When object is not too close nor too far (Fuzzy Logic)
      else:
         probabilities = model.predict_proba(df)[0]
         highest = 0
         for i in range(0, len(probabilities)):
            if probabilities[highest] < probabilities[i]:
               highest = i
         
         match = matching[Y.unique()[highest]]

         second = None
         for i in range(0, len(probabilities)):
            if Y.unique()[i] in match:
               if second is None:
                  second = i
               elif probabilities[second] < probabilities[i]:
                  second = i
         
         total = probabilities[highest] + probabilities[second]
         one = (probabilities[highest] * 100)/total
         two = (probabilities[second] * 100)/total

         final = Y.unique()[highest] + " : " + str(one) + "%. " + \
             Y.unique()[second] + " : " + str(two) + "%."
      return final

if __name__ == '__main__':
   app.run(debug=True)
