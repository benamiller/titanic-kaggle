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
