class Tower:
    def __init__(self, json_data) -> None:
        self.cost = json_data['tower_construction_cost']
        self.maintenance = json_data['tower_maintanance_cost']
        self.satisfaction_level = json_data['user_satisfaction_levels']
        self.satisfaction_score = json_data['user_satisfaction_scores']
