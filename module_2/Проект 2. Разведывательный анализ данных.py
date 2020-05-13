#!/usr/bin/env python
# coding: utf-8

# # Основная информация

# **Суть проекта** — отследить влияние условий жизни учащихся в возрасте от 15 до 22 лет на их успеваемость по математике, чтобы на ранней стадии выявлять студентов, находящихся в группе риска.
# 
# **Цель проекта:** построить модель, которая предсказывала бы результаты госэкзамена по математике для каждого ученика школы
# 
# **Цель EDA:** первично проверить гипотезу о связи различных переменных с результатом госэкзамена по математике
# 
# **Задачи:**
# * выявить параметры, влияющие на результаты госэкзамена по математике;
# * отобрать наиболее важные переменные;
# * обнаружить отклонения и выбросы.
# 
# **В дата сете представлены следующие данные**:
# 1. **school** — аббревиатура школы, в которой учится ученик
# 2. **sex** — пол ученика ('F' - женский, 'M' - мужской)
# 3. **age** — возраст ученика (от 15 до 22)
# 4. **address** — тип адреса ученика ('U' - городской, 'R' - за городом)
# 5. **famsize** — размер семьи('LE3' <= 3, 'GT3' >3)
# 6. **Pstatus** — статус совместного жилья родителей ('T' - живут вместе 'A' - раздельно)
# 7. **Medu** — образование матери (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)
# 8. **Fedu** — образование отца (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)
# 9. **Mjob** — работа матери ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)
# 10. **Fjob** — работа отца ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)
# 11. **reason** — причина выбора школы ('home' - близость к дому, 'reputation' - репутация школы, 'course' - образовательная программа, 'other' - другое)
# 12. **guardian** — опекун ('mother' - мать, 'father' - отец, 'other' - другое)
# 13. **traveltime** — время в пути до школы (1 - <15 мин., 2 - 15-30 мин., 3 - 30-60 мин., 4 - >60 мин.)
# 14. **studytime** — время на учёбу помимо школы в неделю (1 - <2 часов, 2 - 2-5 часов, 3 - 5-10 часов, 4 - >10 часов)
# 15. **failures** — количество внеучебных неудач (n, если 1<=n<3, иначе 4)
# 16. **schoolsup** — дополнительная образовательная поддержка (yes или no)
# 17. **famsup** — семейная образовательная поддержка (yes или no)
# 18. **paid** — дополнительные платные занятия по математике (yes или no)
# 19. **activities** — дополнительные внеучебные занятия (yes или no)
# 20. **nursery** — посещал детский сад (yes или no)
# 21. **granular_studytime** — 
# 22. **higher** — хочет получить высшее образование (yes или no)
# 23. **internet** — наличие интернета дома (yes или no)
# 24. **romantic** — в романтических отношениях (yes или no)
# 25. **famrel** — семейные отношения (от 1 - очень плохо до 5 - очень хорошо)
# 26. **freetime** — свободное время после школы (от 1 - очень мало до 5 - очень мого)
# 27. **goout** — проведение времени с друзьями (от 1 - очень мало до 5 - очень много)
# 28. **health** — текущее состояние здоровья (от 1 - очень плохо до 5 - очень хорошо)
# 29. **absences** — количество пропущенных занятий
# 20. **score** — баллы по госэкзамену по математике
# 
# **Этапы:**
# 1. Первичный отсмотр данных
# 2. Первичный анализ данных в столбцах
# 3. Анализ номинативных переменных
# 4. Коррелляционный анализ
# 5. Выводы
# 

# ### Используемые функции

# In[135]:


"Определяем кол-во пропущенных значений"
def missing_values(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * mis_val / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("В таблице " + str(df.shape[1]) + " столбцов.\n"      
            "из них " + str(mis_val_table_ren_columns.shape[0]) +
              " колонок имеют пропущенные значения.")
        return mis_val_table_ren_columns


# In[136]:


"Анализ переменных"

def allocation(df_col):
    y = int(pd.DataFrame(stud_math.school).count())- int(pd.DataFrame(df_col).count())
    z = len(pd.DataFrame(df_col.value_counts()))
    return display(pd.DataFrame(df_col.value_counts())), print('В колонке:', 'уникальных значений -', z, 'пропущенных значений -', y)


# In[137]:


"Определяем наличие выбросов"
def outliers(df_col):
    perc25 = df_col.quantile(0.25)
    perc75 = df_col.quantile(0.75)
    IQR = perc75 - perc25
    lower_border = perc25 - 1.5*IQR
    higher_border = perc75 + 1.5*IQR
    for i in df_col:
        if i <= lower_border or i >= higher_border:
            print('В колонке есть значения, которые могут считаться выбросами')
            break


# In[138]:


"Определяем границы выбросов"
def outlier_scope(df_col):
    perc25 = df_col.quantile(0.25)
    perc75 = df_col.quantile(0.75)
    IQR = perc75 - perc25
    lower_border = perc25 - 1.5 * IQR
    higher_border = perc75 + 1.5 * IQR
    return("Границы выбросов:", lower_border, higher_border)


# In[139]:


"Удаляем выбросы"
def remove_outlier(df, df_col):
    perc25 = df_col.quantile(0.25)
    perc75 = df_col.quantile(0.75)
    IQR = perc75 - perc25
    lower_border = perc25 - 1.5 * IQR
    higher_border = perc75 + 1.5 * IQR
    return df.loc[(df_col > lower_border) & (df_col < higher_border)]


# In[140]:


"Определяем столбцы со статистически значимыми отклонениями"
def get_stat_dif(column):
    cols = stud_math.loc[:, column].value_counts().index[:10]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(stud_math.loc[stud_math.loc[:, column] == comb[0], 'score'], 
                        stud_math.loc[stud_math.loc[:, column] == comb[1], 'score']).pvalue \
            <= 0.05/len(combinations_all): # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


# In[141]:


"Строим гистограмму Density Plots"
def Density(df, df_col, x):
    types = df.dropna(subset=['score'])
    types = types[x].value_counts()
    types = list(types[types.values > 100].index)

    for b_type in types:

        subset = stud_math[df_col == b_type]

        sns.kdeplot(subset['score'].dropna(),
                   label = b_type, shade = False, alpha = 0.8);

    plt.xlabel('Score', size = 20); plt.ylabel('Density', size = 20); 
    plt.title('Density Plot Score', size = 28);


# # Импорт библиотек

# In[142]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind
import warnings


# # Очистка и форматирование данных

# ## Загрузка данных

# In[143]:


warnings.filterwarnings('ignore') # запретим оповещения

pd.set_option('display.max_rows', 50) # выведем больше строк
pd.set_option('display.max_columns', 50) # выведем больше колонок

stud_math = pd.read_csv('/Users/kseniahabipova/Downloads/stud_math.csv')


# Выведем названия столбцов:

# In[144]:


print(stud_math.columns)


# Переименуем некоторые из столбцов для удобства использования.

# In[145]:


stud_math = stud_math.rename(columns={'Pstatus': 'parents_status', 'Medu': 'mother_education', 'Fedu': 'father_education',
                          'Mjob': 'mother_job', 'Fjob': 'father_job', 'schoolsup': 'school_support',
                          'famsup': 'family_support', 'paid': 'paid_math_course', 
                          'studytime, granular': 'granular_studytime', 'romantic': 'romantic_relations', 
                          'famrel': 'family_relations', 'goout': 'go_out'})


# Посмотрим получившуюся таблицу:

# In[146]:


stud_math.head()


# ## Типы данных и пропущенные значения

# Таблица содержит 30 столбцов с данными. Посмотрим, какие типы данных присутствуют:

# In[147]:


stud_math.info()


# Для удобства анализа сконвертируем столбцы, имеющие значения "Yes" и "No" в тип "float"

# In[148]:


# заменим значения yes = 0, no = 1
stud_math = stud_math.replace({'yes': 0.0, 'no': 1.0})

# заменим NaN, на "не число", интерпретируемое при чтении как число
stud_math = stud_math.replace({'NaN': np.nan})

# Конвертируем столбцы
for col in list(stud_math.columns):
    if ('school_support' in col or 'family_support' in col or 'paid_math_course' in col or 
        'activities' in col or 'nursery' in col or 'higher' in col or 'internet' in col in col or 
        'romantic_relations' in col):
        stud_math[col] = stud_math[col].astype(float)


# ## Пропущенные значения

# Оценим кол-во пропущенных значений:

# In[149]:


missing_values(stud_math)


# 27 колонок имеют пропущенные значения. Доля пропущенных значений не превышает 11,5% В целом показатель не очень высокий, оставим все строки.

# ## Выбросы

# Проверим наличие выбросов. Сначала посмотрим описание таблицы:

# In[150]:


stud_math.describe()


# В столбце father_education присутствует значение равное 40, скорее всего это опечатка. Заменим его на значение 4:

# In[151]:


stud_math.father_education = stud_math.father_education.apply(lambda x: 4.0 if x == 40.0 else x)


# В остальных столбцах визуально все дланнные в порядке, проверим каждый столбец отдельно:

# ### 1. school

# In[152]:


allocation(stud_math.school)


# ### 2. sex

# In[153]:


allocation(stud_math.sex)


# ### 3. age

# In[154]:


allocation(stud_math.age)


# Посмотрим на распределение возрастов:

# In[155]:


stud_math.age.hist()
stud_math.age.describe()


# Распределение диспропорционально, есть вероятность наличия выбросов. Проверим это предположение:

# In[156]:


outliers(stud_math.age)


# In[157]:


outlier_scope(stud_math.age)


# Как выбросы можно определить значения возрастов 21 год и старше. С точки зрения логики лучше удалить значения возрастов старше 20 лет, так как, судя по значениям, они случайны

# In[158]:


stud_math = stud_math.loc[stud_math.age < 20]


# ### 4. address

# In[159]:


allocation(stud_math.address)


# ### 5. famsize

# In[160]:


allocation(stud_math.famsize)


# ### 6. parents_status

# In[161]:


allocation(stud_math.parents_status)


# ### 7. mother_education

# In[162]:


allocation(stud_math.mother_education)


# Посмотрим на распределение значений:

# In[163]:


stud_math.mother_education.hist()
stud_math.mother_education.describe()


# Значение 0.0 распределено диспропорционально. Для удаления диспропорции отфильтруем данное значение

# In[164]:


stud_math = stud_math.loc[stud_math.mother_education >= 1.0]


# In[165]:


allocation(stud_math.mother_education)


# ### 8. father_education

# In[166]:


allocation(stud_math.father_education)


# Посмотрим на распределение значений:

# In[167]:


stud_math.father_education.hist()
stud_math.father_education.describe()


# Значение 0.0 распределены диспропорционально. Для удаления диспропорции отфильтруем данное значение

# In[168]:


stud_math = stud_math.loc[stud_math.father_education >= 1.0]


# In[169]:


allocation(stud_math.father_education)


# ### 9. mother_job

# In[170]:


allocation(stud_math.mother_job)


# ### 10. father_job

# In[171]:


allocation(stud_math.father_job)


# Отфильтруем значения, встречающиеся меньше 20 раз

# In[172]:


stud_math = stud_math.loc[(stud_math.father_job != 'at_home')&(stud_math.father_job != 'health')]


# In[173]:


allocation(stud_math.father_job)


# ### 11. reason

# In[174]:


allocation(stud_math.reason)


# ### 12. guardian

# In[175]:


allocation(stud_math.guardian)


# ### 13. traveltime

# In[176]:


allocation(stud_math.traveltime)


# Посмотрим на распределение значений:

# In[177]:


stud_math.traveltime.hist()
stud_math.traveltime.describe()


# Значения 3.0 и 4.0 распределены диспропорционально. Для удаления диспропорции изменим значения таким образом, чтобы 2.0 соответствовало значению "путь до школы более 15 мин"

# In[178]:


stud_math.traveltime = stud_math.traveltime.apply(lambda x: 2.0 if x == 3.0 else x)


# In[179]:


stud_math.traveltime = stud_math.traveltime.apply(lambda x: 2.0 if x == 4.0 else x)


# ### 14. studyltime

# In[181]:


allocation(stud_math.studytime)


# Посмотрим на распределение значений:

# In[182]:


stud_math.studytime.hist()
stud_math.studytime.describe()


# Значение 4.0 распределено диспропорционально. Для удаления диспропорции изменим значения таким образом, чтобы 4.0 соответствовало значению "время на учёбу помимо школы в неделю более 5 часов"

# In[183]:


stud_math.studytime = stud_math.studytime.apply(lambda x: 3.0 if x == 4.0 else x)


# ### 15. failures

# In[185]:


allocation(stud_math.failures)


# Посмотрим на распределение значений:

# In[186]:


stud_math.failures.hist()
stud_math.failures.describe()


# Значения, в которых зафиксировано кол-во провалов (1.0, 2.0, 3.0) распределены диспропорционально. Для удаления диспропорции изменим значения таким образом, в таблице фиксировались провалы в формате 1.0 (да) и 0.0 (нет)

# In[187]:


stud_math.failures = stud_math.failures.apply(lambda x: 1.0 if x == 2.0 else x)


# In[188]:


stud_math.failures = stud_math.failures.apply(lambda x: 1.0 if x == 3.0 else x)


# ### 16. school_support

# In[189]:


allocation(stud_math.school_support)


# Посмотрим на распределение значений:

# In[190]:


stud_math.school_support.hist()
stud_math.school_support.describe()


# Видна диспропорция в распределении, но так как в таблице дано всего 2 варианта значения, пока вариантов оптимизации нет

# ### 17. family_support

# In[191]:


allocation(stud_math.family_support)


# Посмотрим на распределение значений:

# In[192]:


stud_math.family_support.hist()
stud_math.family_support.describe()


# Диспропорций нет

# ### 18. paid_math_course

# In[193]:


allocation(stud_math.paid_math_course)


# Посмотрим на распределение значений:

# In[194]:


stud_math.paid_math_course.hist()
stud_math.paid_math_course.describe()


# Диспропорций нет

# ### 19. activities

# In[195]:


allocation(stud_math.activities)


# Посмотрим на распределение значений:

# In[196]:


stud_math.activities.hist()
stud_math.activities.describe()


# Диспропорций нет

# ### 20. nursery

# In[197]:


allocation(stud_math.nursery)


# Посмотрим на распределение значений:

# In[198]:


stud_math.nursery.hist()
stud_math.nursery.describe()


# Есть небольшая диспропорция в распределении, но так как даны всего 2 варианта значений и оба значения достаточно велики, пока оставляем данные в таком виде

# ### 21. granular_studytime

# In[199]:


allocation(stud_math.granular_studytime)


# Посмотрим на распределение значений:

# In[200]:


stud_math.granular_studytime.hist()
stud_math.granular_studytime.describe()


# Распределение диспропорционально, есть вероятность наличия выбросов. Проверим это предположение:

# In[201]:


outliers(stud_math.granular_studytime)


# In[202]:


outlier_scope(stud_math.granular_studytime)


# Отфильтруем значения, определенные как выбросы

# In[203]:


stud_math = remove_outlier(stud_math, stud_math.granular_studytime)


# ### 22. higher

# In[204]:


allocation(stud_math.higher)


# В колонке всего 2 значения и одно из них встречается всего 15 раз. Удалим колонку

# In[205]:


stud_math = stud_math.drop(['higher'], axis=1)


# ### 23. internet

# In[206]:


allocation(stud_math.internet)


# Посмотрим на распределение значений:

# In[207]:


stud_math.internet.hist()
stud_math.internet.describe()


# Есть диспропорция в распределении, но так как даны всего 2 варианта значений и оба значения достаточно велики, пока оставляем данные в таком виде

# ### 24. romantic_relations

# In[208]:


allocation(stud_math.romantic_relations)


# Посмотрим на распределение значений:

# In[209]:


stud_math.romantic_relations.hist()
stud_math.romantic_relations.describe()


# Диспропорции нет

# ### 25. family_relations

# In[210]:


allocation(stud_math.family_relations)


# Посмотрим на распределение значений:

# In[211]:


stud_math.family_relations.hist()
stud_math.family_relations.describe()


# Есть диспропорция в распределении и есть в наличии отрицательные значения, которых не должно быть в данном столбце. Отфильтруем отрицаельные значения:

# In[212]:


stud_math = stud_math.loc[stud_math.family_relations >= 1]


# Есть диспропорция в распределении. Перегруппируем значения таким образом, чтобы шкала изменилась на: от 1.0 до 4.0 (от "плохо" до "отлично")

# In[213]:


stud_math.family_relations = stud_math.family_relations.apply(lambda x: 1.0 if x == 2.0 else x)


# In[214]:


stud_math.family_relations = stud_math.family_relations.apply(lambda x: 2.0 if x == 3.0 else x)


# In[215]:


stud_math.family_relations = stud_math.family_relations.apply(lambda x: 3.0 if x == 4.0 else x)


# In[216]:


stud_math.family_relations = stud_math.family_relations.apply(lambda x: 4.0 if x == 5.0 else x)


# ### 26. freetime

# In[218]:


allocation(stud_math.freetime)


# Посмотрим на распределение значений:

# In[219]:


stud_math.freetime.hist()
stud_math.freetime.describe()


# Есть диспропорция в распределении. Перегруппируем значения таким образом, чтобы шкала изменилась на: от 1.0 до 3.0 (от "мало" до "много")

# In[220]:


stud_math.freetime = stud_math.freetime.apply(lambda x: 1.0 if x == 2.0 else x)


# In[221]:


stud_math.freetime = stud_math.freetime.apply(lambda x: 2.0 if x == 3.0 else x)


# In[222]:


stud_math.freetime = stud_math.freetime.apply(lambda x: 3.0 if x == 4.0 else x)


# In[223]:


stud_math.freetime = stud_math.freetime.apply(lambda x: 3.0 if x == 5.0 else x)


# ### 27. go_out

# In[225]:


allocation(stud_math.go_out)


# Посмотрим на распределение значений:

# In[226]:


stud_math.go_out.hist()
stud_math.go_out.describe()


# Есть диспропорция в распределении. Перегруппируем значения таким образом, чтобы шкала изменилась на: от 1.0 до 3.0 (от "Мало" до "Много)

# In[227]:


stud_math.go_out = stud_math.go_out.apply(lambda x: 1.0 if x == 2.0 else x)


# In[228]:


stud_math.go_out = stud_math.go_out.apply(lambda x: 2.0 if x == 3.0 else x)


# In[229]:


stud_math.go_out = stud_math.go_out.apply(lambda x: 3.0 if x == 4.0 else x)


# In[230]:


stud_math.go_out = stud_math.go_out.apply(lambda x: 3.0 if x == 5.0 else x)


# ### 28. health

# In[232]:


allocation(stud_math.health)


# Посмотрим на распределение значений:

# In[233]:


stud_math.health.hist()
stud_math.health.describe()


# Диспропорции нет

# ### 29. absences

# In[234]:


allocation(stud_math.absences)


# Посмотрим на распределение значений:

# In[235]:


stud_math.absences.hist()
stud_math.absences.describe()


# Распределение диспропорционально, есть вероятность наличия выбросов. Проверим это предположение:

# In[236]:


outliers(stud_math.absences)


# Отфильтруем значения, определенные как выбросы

# In[237]:


outlier_scope(stud_math.absences)


# In[238]:


stud_math = remove_outlier(stud_math, stud_math.absences)


# In[ ]:





# ### 30. score

# In[239]:


allocation(stud_math.score)


# Посмотрим на распределение значений:

# In[240]:


stud_math.score.hist()
stud_math.score.describe()


# Распределение диспропорционально, есть вероятность наличия выбросов. Проверим это предположение:

# In[241]:


outliers(stud_math.score)


# In[97]:


outlier_scope(stud_math.score)


# In[98]:


stud_math = remove_outlier(stud_math, stud_math.score)


# In[99]:


stud_math.score.hist()
stud_math.score.describe()


# # Корреляционный анализ

# Цель — это влияние условий жизни на успеваемость по математике (score в нашем наборе), так что целесообразно для начала понять, какое эта величина имеет распределение. Посмотрим на него, построив гистограмму

# In[100]:


plt.hist(stud_math['score'].dropna(), bins = 100);
plt.xlabel('Score'); plt.ylabel('Number of students');
plt.title('Score Distribution');


# В получившимся распределении видна диспропорция, значительная часть оценок равна 50 баллам.

# ## Анализ номинативных переменных

# In[101]:


Density(stud_math, stud_math['school'], 'school')


# In[102]:


Density(stud_math, stud_math['sex'], 'sex')


# In[103]:


Density(stud_math, stud_math['address'], 'address')


# In[104]:


Density(stud_math, stud_math['famsize'], 'famsize')


# In[105]:


Density(stud_math, stud_math['parents_status'], 'parents_status')


# In[106]:


Density(stud_math, stud_math['mother_job'], 'mother_job')


# In[107]:


Density(stud_math, stud_math['father_job'], 'father_job')


# In[108]:


Density(stud_math, stud_math['reason'], 'reason')


# Судя по получившимся графикам, значимое влияние на параметр "оценка" оказывает только пол студента. Дополнительно проверим данные:

# In[109]:


for col in ['school', 'sex', 'address', 'famsize', 'parents_status', 
            'mother_job', 'father_job', 'reason', 'guardian']:
    get_stat_dif(col)


# Оставим для дальнейшего анализа колонки "Пол" и "Адрес", "Профессия матери"

# ## Анализ числовых переменных

# Проверим корелляцию столбца:

# In[110]:


stud_math.corr()['score']


# Оставим для дальнейшей работы следующие столбцы:

# In[111]:


stud_math_for_model = stud_math.loc[:, ['sex', 'age', 'address', 'mother_job', 'studytime', 'family_support', 
        'paid_math_course', 'activities', 'nursery', 'granular_studytime', 'internet', 'romantic_relations',
       'family_relations', 'freetime', 'health', 'score']]

stud_math_for_model.head()


# # Выводы

# В результате EDA для анализа влияния условий жизни на успеваемость школьников по математике были получены следующие выводы:
# 
# * В данных достаточно мало пустых значений, только столбец bean_type был заполнен в малом количестве случаев.
# * Выбросы найдены только в 4 столбцах, что позволяет сделать вывод о том, что данные достаточно чистые.
# * В части колонок недостаточно данных по некоторым показателям, поэтому для аналитики их лучше сгруппировать в более крупные группы;
# * Положительная корреляция с параметрами профессии родителей может говорить о том, что родители с определенным профессиональным статусом начинают влиять на учебные успехи своих детей.
# * Самые важные параметры, которые предлагается использовать в дальнейшем для построения модели, это 'sex', 'age', 'address', 'mother_job', 'studytime', 'family_support','paid_math_course', 'activities', 'nursery', 'granular_studytime', 'internet', 'romantic_relations','family_relations', 'freetime', 'health'.

# In[ ]:




