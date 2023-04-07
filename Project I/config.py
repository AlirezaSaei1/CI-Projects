import json
class Configurations:

    @classmethod
    def load_config(cls, file):
        with open(file) as f:
            json_data = json.load(f)
            
        cls.cost = json_data['tower_construction_cost']
        cls.maintenance = json_data['tower_maintanance_cost']
        cls.levels = json_data['user_satisfaction_levels']
        cls.scores = json_data['user_satisfaction_scores']


    @classmethod
    def show_configurations(cls):
        return f'''
        ---------------Tower Configurations:---------------
        Consturction Cost: {cls.cost}$
        Cost per 1MB/s: {cls.maintenance}$
        Customer Satisfaction Levels: {cls.levels}
        Customer Satisfaction Scores: {cls.scores}
        ---------------------------------------------------
        '''
    
    @classmethod
    def get_score(cls, bw):
        if bw < cls.levels[0]:
            return 0.0
        
        if bw > cls.levels[-1]:
            return cls.scores[-1]
        
        for i in range(1, len(cls.levels)):
            if cls.levels[i-1] <= bw < cls.levels[i]:
                return cls.scores[i-1]
