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

# # Выводы

# В результате EDA для анализа влияния условий жизни на успеваемость школьников по математике были получены следующие выводы:
# 
# * В данных достаточно мало пустых значений, только столбец bean_type был заполнен в малом количестве случаев.
# * Выбросы найдены только в 4 столбцах, что позволяет сделать вывод о том, что данные достаточно чистые.
# * В части колонок недостаточно данных по некоторым показателям, поэтому для аналитики их лучше сгруппировать более
# * Положительная корреляция с параметрами профессии родителей может говорить о том, что родители с определенным профессиональным статусом начинают влиять на учебные успехи своих детей.
# * Самые важные параметры, которые предлагается использовать в дальнейшем для построения модели, это 'sex', 'age', 'address', 'mother_job', 'studytime', 'family_support','paid_math_course', 'activities', 'nursery', 'granular_studytime', 'internet', 'romantic_relations','family_relations', 'freetime', 'health'.

# In[ ]:




