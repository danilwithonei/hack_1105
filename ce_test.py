from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder('DiTy/cross-encoder-russian-msmarco', max_length=512, device='cuda')

query = ["Как за 10 лет изменилось количество телепрограмм, привлекающих более 4-х млн. зрителей в Великобритании?"]
documents = [
    """Media usage reflects this societal trendAs
toward diversity. People can now set theirdiverse, brands must reflect this evolution in
own TV schedules and choose to follow theirhow they approach their campaigns if they
preferred topics, giving rise to a Generationare to appeal to audiences. This includes
P whose media consumption is personalizedacting to reduce potential bias in the
to their diverse, unique lives.technology they use.
In the UK, the number of TV programsModels may be trained on datasets that
attracting more than four million viewerscontain human biases, for example, in
declined by more than 50% in less than tenrepresentation
years, as viewers choose from a wide rangethat some groups use, and therefore,
of sources the ones that resonate the mostthese models""",
    "Стоматолог, также известный как стоматолог-хирург, является медицинским работником, который специализируется на стоматологии, отрасли медицины, специализирующейся на зубах, деснах и полости рта.",
    "Дядя Женя работает врачем стоматологом",
    "Плоды малины употребляют как свежими, так и замороженными или используют для приготовления варенья, желе, мармелада, соков, а также ягодного пюре. Малиновые вина, наливки, настойки, ликёры обладают высокими вкусовыми качествами.",
]

predict_result = reranker_model.predict([[query[0], documents[0]]])
print(predict_result)
# `array([0.88126713], dtype=float32)`

rank_result = reranker_model.rank(query[0], documents)
print(rank_result)
# `[{'corpus_id': 0, 'score': 0.88126713},
#  {'corpus_id': 2, 'score': 0.001042091},
#  {'corpus_id': 3, 'score': 0.0010417715},
#  {'corpus_id': 1, 'score': 0.0010344835},
#  {'corpus_id': 4, 'score': 0.0010244923}]`
