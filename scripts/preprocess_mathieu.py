def main():

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings

    # Ignore all warnings
    warnings.filterwarnings("ignore")                                                                                          

    train = pd.read_csv("/mnt/c/Users/mathi/Downloads/titanic/train.csv")
    test  = pd.read_csv("/mnt/c/Users/mathi/Downloads/titanic/test.csv")


    print(train.shape)
    print(test.shape)


    test.info(),train.info()

    train.sample(20)


    train.drop(columns=['Cabin'],inplace=True)
    test.drop(columns=['Cabin'],inplace=True)


    train.isnull().sum()

    test.isnull().sum()


    train['Embarked'].fillna('S',inplace=True)


    test['Fare'].fillna(test['Fare'].mean(), inplace=True)

    df=pd.concat([train,test],sort=True).reset_index(drop=True)

    df.shape

    df.head()


    df.corr(numeric_only=True)['Age'].abs()


    df_Age_mean=df.groupby(['Sex', 'Pclass']).median(numeric_only=True)['Age']
    df_Age_mean


    df['Age']=df.groupby(['Sex','Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))


    df.isnull().sum()

    df['Title']=df['Name'].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]


    df['Title'].value_counts()


    df['Title'] = df['Title'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')


    df['Title'].value_counts()


    df['Family_size']=df['SibSp'] + df['Parch'] + 1


    df.sample(10)


    df.drop(columns=['Name','Parch','SibSp','Ticket'],inplace=True)


    df.sample(10)



    def family_size(number):
        if number==1:
            return "Alone"
        elif number>1 and number <5:
            return "Small"
        else:
            return "Large"
        
        
    df['Family_size']=df['Family_size'].apply(family_size)


    df.sample(10)


    def family_size(number):
        if number==1:
            return "Alone"
        elif number >1 and number <5:
            return "Small"
        else:
            return "Large"
        
        
    df['Family_size']=df['Family_size'].apply(family_size)


    df.info()


    df['Age'] = df['Age'].astype('int64')


    df.info()



if __name__ == "__main__":
     main()