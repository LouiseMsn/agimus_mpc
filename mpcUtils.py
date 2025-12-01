
class StagesDefinition:
    constraints : list = []
    terminal_costs : list = []
    stage_dep_costs : list = []
    stage_indep_costs : list = []

    def __repr__(self)->str:
        return f'constraints: {self.constraints}\n\n'\
        f'terminal costs: {self.terminal_costs}\n\n'\
        f'stage dependant costs: {self.stage_dep_costs}\n\n'\
        f'stage independant costs: {self.stage_indep_costs}\n\n'
