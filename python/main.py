import pandas as pd
import numpy as np

print(np.zeros((1,2)))

df = pd.read_csv('./data/train.csv')
print(df.head(5))

## Nullish checks

# PassengerIds
print('PassengerIds')
print(df[df['PassengerId'].isna()])

# Survived
print('Survived')
print(df[df['Survived'].isna()])

# Pclass (1st, 2nd, or 3rd)
print('Pclass')
print(df[df['Pclass'].isna()])

# Name
print('Name')
print(df[df['Name'].isna()])

# Sex
print('Sex')
print(df[df['Sex'].isna()])

# Age
print('Age')
print(df[df['Age'].isna()])

# SibSp (Number of siblings/spouses aboard
print('SibSp')
print(df[df['SibSp'].isna()])

# Parch
print('Parch')
print(df[df['Parch'].isna()])

# Ticket
print('Ticket')
print(df[df['Ticket'].isna()])

# Fare
print('Fare')
print(df[df['Fare'].isna()])

# Cabin
print('Cabin')
print(df[df['Cabin'].isna()])

# Embarked (Port of Embarkation)
print('Embarked')
print(df[df['Embarked'].isna()])

print("~~~~~~~DATASETS~~~~~~~")

x = df.drop(columns=['Survived'])
y = df['Survived']

print("Features")
print(x.head(5))
print("Labels")
print(y.head(5))

np.random.seed(42)

x = x.to_numpy()
y = y.to_numpy()

indices = np.arange(len(x))
np.random.shuffle(indices)

x = x[indices]
y = y[indices]

ratio = 0.8
split_index = int(len(x) * ratio)

x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = x[:split_index], x[split_index:]
