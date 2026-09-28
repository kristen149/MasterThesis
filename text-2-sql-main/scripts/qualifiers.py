import pandas as pd

BASE_TOTAL = 205
PARAPHRASE_TOTAL = 1450
BP_TOTAL = 1655

class Qualifier(object):
    base_paraphrase = {}
    base = {}
    paraphrase = {}
    
    def __init__(self, base_paraphrase, *args, model_key='opr_gpt35', **kwargs):
        self.base_paraphrase = base_paraphrase
        self.model_key = model_key
        self.base = self.base_paraphrase.query("Base_paraphrase == 'base'")
        self.paraphrase = self.base_paraphrase.query("Base_paraphrase == 'paraphrase'")

    def get_counts_percentages(self, column):
        base_counts = self.base[column].value_counts()
        paraphrase_counts = self.paraphrase[column].value_counts()
        bp_counts = self.base_paraphrase[column].value_counts()
        base_percentages = [count / BASE_TOTAL * 100 for count in base_counts]
        paraphrase_percentages = [count / PARAPHRASE_TOTAL * 100 for count in paraphrase_counts]
        bp_percentages = [count / BP_TOTAL * 100 for count in bp_counts]
        return base_counts, base_percentages, paraphrase_counts, paraphrase_percentages, bp_counts, bp_percentages

    def get_results_em(self, column):
        hardness_col = f'hardness_{self.model_key}'
        score_col = f'score_{self.model_key}'
        base_total = self.base.query(f"{hardness_col} != 'no_hardness'")[column].value_counts()
        paraphrase_total = self.paraphrase.query(f"{hardness_col} != 'no_hardness'")[column].value_counts()
        base_ok_em = self.base.query(f"{score_col} == 1 and {hardness_col} != 'no_hardness'")[column].value_counts()
        paraphrase_ok_em = self.paraphrase.query(f"{score_col} == 1 and {hardness_col} != 'no_hardness'")[column].value_counts()
        return base_total, base_ok_em, paraphrase_total, paraphrase_ok_em

    def get_results_ex(self, column):
        score_col = f'score_{self.model_key}'
        base_total = self.base[column].value_counts()
        paraphrase_total = self.paraphrase[column].value_counts()
        base_ok_ex = self.base.query(f"{score_col} == 1")[column].value_counts()
        paraphrase_ok_ex = self.paraphrase.query(f"{score_col} == 1")[column].value_counts()
        return base_total, base_ok_ex, paraphrase_total, paraphrase_ok_ex

#Event and Case Level (PMp)
class Qualifier1(Qualifier):
    def __init__(self, base_paraphrase, *args, **kwargs):
        super().__init__(base_paraphrase, *args, **kwargs)

#Perspective Level (PMp)
class Qualifier2(Qualifier):
    def __init__(self, base_paraphrase, *args, **kwargs):
        purpose = base_paraphrase
        purpose['Purpose_classification'] = purpose['Purpose_classification'].str.split(';')
        purpose = purpose.explode('Purpose_classification')
        purpose['Purpose_classification_class'] = purpose['Purpose_classification'].str.split(' - ').str.get(0)
        super().__init__(purpose, *args, **kwargs)

#Process mining concepts on SELECT (PMp)
class Qualifier3(Qualifier):
    def __init__(self, base_paraphrase, *args, **kwargs):
        projection = base_paraphrase
        projection['Projection_classification'] = projection['Projection_classification'].str.split(';')
        projection = projection.explode('Projection_classification')
        super().__init__(projection, *args, **kwargs)

#Process mining concepts on WHERE (PMp)
class Qualifier4(Qualifier):
    def __init__(self, base_paraphrase, *args, **kwargs):
        condition = base_paraphrase
        condition['Condition_classification'] = condition['Condition_classification'].str.split(';')
        condition = condition.explode('Condition_classification')
        super().__init__(condition, *args, **kwargs)

#WH question Level (NLp)
class Qualifier5(Qualifier):
    def __init__(self, base_paraphrase, *args, **kwargs):
        wh_question = base_paraphrase
        wh_question['Wh_classification'] = wh_question['Wh_classification'].str.split(';')
        wh_question = wh_question.explode('Wh_classification')
        super().__init__(wh_question, *args, **kwargs)

#Aggregation on SELECT (SQLp)
class Qualifier6(Qualifier):
    def __init__(self, base_paraphrase, *args, **kwargs):
        super().__init__(base_paraphrase, *args, **kwargs)

#Have GROUP BY (SQLp)
class Qualifier7(Qualifier):
    def __init__(self, base_paraphrase, *args, **kwargs):
        super().__init__(base_paraphrase, *args, **kwargs)

#SQL complexity (SQLp)
class Qualifier8(Qualifier):
    def __init__(self, base_paraphrase, *args, **kwargs):
        super().__init__(base_paraphrase, *args, **kwargs)

#Value, generic or domain (PMp)
class Qualifier9(Qualifier):
    def __init__(self, base_paraphrase, *args, **kwargs):
        super().__init__(base_paraphrase, *args, **kwargs)