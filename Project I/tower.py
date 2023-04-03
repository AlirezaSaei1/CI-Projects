class Tower:
    
    def __init__(self, json_data) -> None:
        self.cost = json_data['tower_construction_cost']
        self.maintenance = json_data['tower_maintanance_cost']
        self.levels = json_data['user_satisfaction_levels']
        self.scores = json_data['user_satisfaction_scores']

    def __str__(self) -> str:
        return f'''
        ---------------Tower Configurations:---------------
        Consturction Cost: {self.cost}$
        Cost per 1MB/s: {self.maintenance}$
        Customer Satisfaction Levels: {self.levels}
        Customer Satisfaction Scores: {self.scores}
        ---------------------------------------------------
        '''

    def get_score(self, bw):
        if bw < self.levels[0]:
            return 0.0
        
        if bw > self.levels[-1]:
            return self.scores[-1]
        
        for i in range(1, len(self.levels)):
            if self.levels[i-1] <= bw < self.levels[i]:
                return self.scores[i-1]
