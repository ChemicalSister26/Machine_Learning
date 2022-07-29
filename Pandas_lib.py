import pandas as pd
import numpy as np

students_performance = pd.read_csv('csv_source/StudentsPerformance.csv')
a = students_performance.describe()
b = students_performance.dtypes
c = students_performance.shape
d = students_performance.size
e = students_performance.iloc[0:5, 0:2]                               #index location - gives a part of dataframe
e1 = students_performance.iloc[[0, 3, 5], [0, 2, 5]]

student_performance_with_names = students_performance.iloc[[1, 3, 4, 5]]
student_performance_with_names.index = ['Grace', 'Bob', 'Allen', 'Mary']
e2 = student_performance_with_names.loc[['Grace', 'Allen']]
e3 = pd.Series([1, 2, 3, 4], index = ['Grace', 'Bob', 'Allen', 'Mary'])
e4 = pd.Series([5, 6, 7, 8], index = ['Grace', 'Bob', 'Allen', 'Mary'])
e5 = pd.DataFrame({'col1': e3, 'col2': e4})                              # creates DataFrame from two Series


f = pd.read_csv('dataset/titanic.csv')
e6 = students_performance.gender
e7 = students_performance.gender == 'female'
e8 = students_performance.loc[students_performance.gender == 'female']
e9 = students_performance['writing score'].mean()
e10 = students_performance.loc[students_performance['writing score'] > e9]
e11 = (students_performance['writing score'] > e9) & (students_performance.gender == 'female')
e12 = students_performance.loc[e11]
e13 = students_performance['lunch'] == 'free/reduced'
e14 = students_performance['lunch'] == 'standard'


students_performance = students_performance.rename(columns={
    'parental level of education': 'parental_level_of_education',
    'test preparation course': 'test_preparation_course',
    'math score': 'math_score',
    'reading score': 'reading_score',
    'writing score': 'writing_score'
})
variable = 71
res = students_performance.query('writing_score > 75')  # query is a very cozy instruments
res1 = students_performance.query('gender == "female"')
res2 = students_performance.query('gender == "female" & writing_score > 74')
res3 = students_performance.query('writing_score > @variable')  # when comparing with variable we should to shield the variable

res4 = list(students_performance)
res5 = students_performance.filter(like='9', axis=0)

res6 = students_performance.groupby('gender', as_index=False).aggregate({'reading_score': 'mean',
                                                         'writing_score': 'mean'})

#it is better to point arg as_index=Fals, it makes data more consistent

res7 = students_performance.groupby(['gender', 'race/ethnicity'], as_index=False).aggregate({'reading_score': 'mean'})
res8 = students_performance.sort_values(['gender', 'writing_score']).groupby('gender').head(10)
dota_hero = pd.read_csv('dataset/dota_hero_stats.csv')


# here I am founding the number of heroes with certain number of legs (groupby legs number)
res9 = dota_hero.groupby('legs').aggregate({"legs": 'count'})

#here I am looking for higher average earnings between two contributors
earnings = pd.read_csv('dataset/accountancy.csv')
res10 = earnings.groupby(['Executor', 'Type']).aggregate({'Salary': 'mean'})

res11 = dota_hero.iloc[:, [1,6]]
res12 = dota_hero.groupby(['attack_type', 'primary_attr']).aggregate({'primary_attr': 'count'})

vodorosli = pd.read_csv('dataset/algae.csv')
print(vodorosli)
res13 = vodorosli.groupby('genus').aggregate({'sucrose': 'mean', 'alanin': 'mean', 'citrate': 'mean',
                                               'glucose': 'mean', 'oleic_acid': 'mean'})





