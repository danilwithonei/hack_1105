
has_answer_prompt = """
## Текст
{context}

## Вопрос
{question}

## Задача
Напиши 1, если точно уверен, что текст содержит ответ на вопрос, иначе напиши 0
Если текст близок по смыслу к вопросу, но не содержит ответа, напиши 0
Ничего не объясняй. В ответе должно быть только 1 или 0
"""

base_prompt = """
## Задание
Ты специалист по работе с данными, ответь на вопрос используя информацию из контекста.
Дай краткий ответ. Не придумывай новую информацию. 

## Контекст
{context}

## Вопрос
{question}

## Ответ
"""


classifier = '''
## Задача
Ты специалист службы поддержки и должен направить пользователя к другому специалисту. 
Укажи только номер специалиста, без объяснений. Не используй точки.

Список специалистов:
0) Специалист по вопросам аналитики 
1) HR по вопросам: об адресе, о директоре, о контактах

Если не знаешь к какому специалисту отправить, пиши 0.

## Задание
Вопрос:
{context}{question}

Результат:
'''

company_extraction = '''
## Задача
Ты специалист по работе с компаниями. Из запроса пользователя извлеки название компании.
Если название не указано, напиши null
Не придумывай название, если его нет в тексте.

## Пример
Вопрос:
сформируй отчет по компании VK видео

Результат:
VK видео

Вопрос:
аналитический отчет по одноклассникам

Результат:
одноклассники

## Задание
Вопрос:
{context}{question}

Результат:
'''

year_extraction = '''
## Задача
Ты специалист по работе с компаниями. Из запроса пользователя извлеки год.
Если год не указан, напиши null 
Не придумывай год, если его нет в тексте.

## Пример
Вопрос:
сформируй отчет по компании VK видео за 2023 год

Результат:
2023

Вопрос:
аналитический отчет за 2005г. по одноклассникам

Результат:
2005

## Задание
Вопрос:
{context}{question}

Результат:
'''

report_header = '''
Отчет по {company} за {year} год.
________________________________________________________________________
'''

report = {
    'Ежедневная аудитория' : 'Какая средняя ежедневная аудитория {company} в {year} году?',
    'DAU' : 'Какое DAU у {company} в {year} году?',
    'Средняя месячная аудитория' : 'Какая средняя месячная аудитория {company} в {year} году?',
    'MAU' : 'Какое MAU у {company} в {year} году?',
    'Среднее количество публикаций за месяц' : 'Какое среднее количество публикаций у {company} в {year} году за месяц?',
    'Бизнес модель' : 'Какая бизнес модель у {company} в {year} году?',
    'Оценка удобства интерфейса' : 'Какая оценка удобства интерфейса {company} в {year} году?',
    'Оценка рекомендательных алгоритмов' : 'Какая оценка рекомендательных алгоритмов {company} в {year} году?',
    'Сколько времени проводят пользователи' : 'Сколько времени пользователи проводят в {company} в {year} году?',
    'Выручка' : 'Какая выручка {company} в {year} год?',
    'Расходы' : 'Какие расходы {company} в {year} год?',
    'Среднесуточный охват' : 'Какой среднесуточный охват у {company} в {year} году?',
    'Рекламные бюджеты' : 'Какие рекламные бюджеты {company} в {year} год?',
    'Количество просмотров/прослушиваний в год' : 'Количество просмотров/прослушиваний на {company} в {year} год?',
    'Количество просмотров/прослушиваний за день' : 'Количество просмотров/прослушиваний на {company} в {year} году в среднем за день?',
    'Количество просмотров/прослушиваний за месяц' : 'Количество просмотров/прослушиваний на {company} в {year} году в среднем за месяц?',

}


about_company_prompt = '''
## Задача
Ты специалист представитель компании. Основываясь на контексте ответь на вопрос нового сотрудника.
Если информации нет, не придумывай ее, скажи, что не знаешь.

## Контекст
 
Media Wise — медийное агентство полного цикла, решающее бизнес-задачи клиента через эффективное присутствие бренда в каналах коммуникации. Агентство специализируется на разработке и реализации медийных стратегий и решений.
115114, Москва,
Дербеневская наб., д. 7, стр. 9
Построить маршрут
отправить на почту
info@themediawise.com
+7 495 783 44 24

Николай Муравьев - генеральный директор

---

MediaDirectionGroup – ведущая группа коммуникационных и специализированных агентств. Мы обладаем глобальной экспертизой, опытом и высокими стандартами, которые многие годы успешно применяем на российском рынке. 


О нас
Агентства
MEDIA DIRECTION TALKS
Вакансии
Новости
Контакты
Пресс-служба
+7 495 783 09 88
pr@mediadirectiongroup.ru
удалить
поиск
 
 
MediaDirectionGroup – ведущая группа коммуникационных и специализированных агентств. Мы обладаем глобальной экспертизой, опытом и высокими стандартами, которые многие годы успешно применяем на российском рынке. 

Наша история

2018
Media Direction Group названа медиасетью года на фестивале Red Apple: Festival Of Media.

2016
Открыто специализированное агентство Media Direction Programmatic.

2016
Cоздано агентство спортивного маркетинга Media Direction Sport.

2015
Медиагруппа OMD MD|PHD Group получила новое имя Media Direction Group.

2014
Вновь запущено медийное агентство Media Wise, принадлежащее Omnicom Media Group.

2012
В составе медиагруппы появилось digital-агентство Proximity Media, выделившееся из Proximity Russia.

2010
На базе подразделения BBDO Interactive, занимающегося интернет-проектами, и digital-подразделения Code Of Trade образовано агентство Media Direction Digital.

2008
Из спонсорского отдела Code Оf Trade сформировано агентство FUSE Media Direction Group.

2007
Для координации работы растущих медийных структур создана группа OMD MD|PHD Group, в которую вошли медийные агентства OMD Media Direction и PHD, баинговая компания Code Оf Trade и маркетинговое агентство BrandScience.

2006
На базе агентства Media Wise запущено медийное агентство PHD, вошедшее в одноименную крупную международную сеть. Исследовательское подразделение OMD Media Direction преобразовано в маркетинговое агентство BrandScience.

2005
Баинговое подразделение OMD Media Direction выделено в компанию Code Оf Trade для консолидации закупок обоих медийных агентств.

2003
Открылось второе медийное агентство - Media Wise.

1997
На базе медиа-отдела агентства BBDO Moscow создано медийное агентство OMD Media Direction в составе международной сети OMD.

1989
Основано BBDO Moscow. Компания сформирована для рекламного обслуживания «Пепси-Колы» - первого западного бренда, не только массово представленного в Советском Союзе, но и производящегося на его территории.


## Задание
Вопрос:
{context}{question}

Результат:
'''

db_answer_template = '''
-----------------------
Ответ:
{answer}

-----------------------
Источники:
{files}
'''

files_template = '''
Файл: {filename} Страница: {page}
'''

error_report = '''
Недостаточно информации для формирования отчета.
'''

error_question = '''
Пожалуйста, переформулируйте запрос.
'''

error_no_data = '''
Информация не найдена.
'''