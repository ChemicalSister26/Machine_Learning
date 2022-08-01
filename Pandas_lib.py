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


# creates Pandas Dataframe_________________________________________________________________________
dict = {'type':  ['A', 'A', 'B', 'B'], 'value':  [10, 14, 12, 23]}
my_data = pd.DataFrame(data=dict)


# choose only required columns from my_stat.csv dataset______________________________________________
# variable subset_1 contains 10 str and only one 1 and 3 columns.___________________________________
# variable subset_2 contains all rows except  1 и 5 and only  2 and 4 cols.________________________
df = pd.read_csv('dataset/my_stat (1).csv')
subset_1 = df.iloc[:, [0, 2]].head(10)
subset_2 = df.drop([0, 4], axis=0).iloc[:, [1, 3]]

# subset_3 only those results where V1>0 and variable V3 == 'A'________________________________________
# subset_4  only those results where V2 != 10, or V4 >= 1.

subset_3 = df.query('V1 > 0 and V3 == "A"')
subset_4 = df.query('V2 != 10 or V4 >= 1')


# create new column V5 = V1 + V4____________________________________________________________________
# create new colunm V6 = log(V2)
# then remove them

df['V5'] = df.V1 + df.V4
df['V6'] = np.log(df.V2)
df.drop(['V5', 'V6'], axis=1, inplace=True)
# how to rename columns____________________________________________________________________

df.rename(index=str, columns={"V1": "session_value", "V2": "group", "V3": "time", "V4": "n_users"}, inplace=True)


# filling missed positions in Dataframe_____________________________________________________________________
# replace one meaning by another

my_stat = pd.read_csv('dataset/my_stat_1.csv')
my_stat.fillna(0, inplace=True)
median = my_stat.query('n_users >= 0')['n_users'].median()
my_stat.loc[my_stat.n_users < 0, 'n_users'] = median


# calculate mean() of session_value variable for each group (group)
# rename col session_value to mean_session_value.
mean_session_value = my_stat.groupby('group', as_index=False).agg({'session_value': 'mean'})
mean_session_value.rename(index=str, columns={"session_value": "mean_session_value"}, inplace=True)

print(mean_session_value)