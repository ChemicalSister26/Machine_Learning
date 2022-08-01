import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


students_perf = pd.read_csv('csv_source/StudentsPerformance.csv')
students_perf = students_perf.rename(columns={
    'parental level of education': 'parental_level_of_education',
    'test preparation course': 'test_preparation_course',
    'math score': 'math_score',
    'reading score': 'reading_score',
    'writing score': 'writing_score'})

# students_perf.plot.scatter(x='math_score', y='reading_score')
#
# sns.lmplot(x='math_score', y='reading_score', hue='gender', data=students_perf)
# plt.show()

#
# res = pd.read_csv('dataset/iris.csv')
#
# ax = sns.violinplot(res['petal length'])
# plt.show()



