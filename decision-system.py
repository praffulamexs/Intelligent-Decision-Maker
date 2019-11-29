import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")

# Conditions for Fuzzy Logic
matching = {
   "Left" : ["Up", "Down"],
   "Right" : ["Up", "Down"],
   "Up" : ["Left", "Right"],
   "Down": ["Left", "Right"]
}

class ANN:
   def __init__(self, data):
      self.data = data
      self.X = None
      self.Y = None
      # self.manage_train_data()
      self.model = MLPClassifier(solver="adam")
   
   def manage_train_data(self):
      self.Y = self.data.iloc[:, 0]
      self.X = self.data.iloc[:, 1:]

   def add_data(self, df):
      self.data.append(df)

      # Save Data in File
      file = open('data.csv', 'a')
      df.to_csv(file, header=False, index=False)
      file.close()
      
      self.manage_train_data()
      print("Retraining data...")
      self.train()
      print()

   def train(self):
      self.model.fit(self.X, self.Y)

   def predict(self):
      x = int(input("Enter the X coordinates of the object: "))
      y = int(input("Enter the Y coordinates of the object: "))
      z = int(input("Enter the distance of the object: "))
      df = pd.DataFrame([[x, y, z]], columns=['X', 'Y', 'Distance'])

      probabilities = self.model.predict_proba(df)[0]
      highest = 0
      for i in range(0, len(probabilities)):
         if probabilities[highest] < probabilities[i]:
            highest = i
      
      # Checking if the action should be taken by Human or Machine
      # The below code runs if the machine is capable to handle the situation
      if self.controller(probabilities[highest], x, y, z):
         print()
         print("Decision will be taken by the Machine.")
         # Condition for No Action
         if x**2 + y**2 + z**2 > 75**2:
            final = "Do Nothing"
         
         # When Object is too close
         elif x**2 + y**2 + z**2 <= 50**2:
            final = self.model.predict(df)[0] + " : 100%."
         
         # When object is not too close nor too far (Fuzzy Logic)
         else:
            unique = self.Y.unique()
            match = matching[unique[highest]]

            second = None
            for i in range(0, len(probabilities)):
               if unique[i] in match:
                  if second is None:
                     second = i
                  elif probabilities[second] < probabilities[i]:
                     second = i

            total = probabilities[highest] + probabilities[second]
            one = (probabilities[highest] * 100)/total
            two = (probabilities[second] * 100)/total

            final = unique[highest] + " : " + str(one) + "%. " + \
               unique[second] + " : " + str(two) + "%."
         print("  ", final)
      
   def controller(self, value, x, y, z):
      if value < 0.55:
         correct_input = False
         while not correct_input:
            direction = input("The machine is unable to make a decision, please enter the direction (Up, Left, Right, Down) that you want to go in : ")
            if direction == 'l' or direction == 'L' or direction == 'left' or direction == 'Left':
               direction = 'Left'
               correct_input = True
            elif direction == 'r' or direction == 'R' or direction == 'right' or direction == 'Right':
               direction = 'Right'
               correct_input = True
            elif direction == 'u' or direction == 'U' or direction == 'up' or direction == 'Up':
               direction = 'Up'
               correct_input = True
            elif direction == 'd' or direction == 'D' or direction == 'down' or direction == 'Down':
               direction = 'Down'
               correct_input = True
            else :
               print("Incorrect Input")

         new_df = pd.DataFrame([[direction, x, y, z]], columns=['Direction', 'X', 'Y', 'Distance'])
         self.add_data(new_df)
         return False
      else :
         return True

def main():
   data = pd.read_csv('data.csv')
   ann = ANN(data)
   ann.manage_train_data()
   ann.train()
   ann.predict()

   # while True:
   #    ann.predict()

main()
