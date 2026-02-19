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

def getIndexesFromJointNames(robot_wrapper, joint_names):
    """Returns the list of indexes corresponding to the list of joint names passed in input.

    Args:
        robot_wrapper (pin.RobotWrapper): 
        joint_names (List(str)): list of joints names

    Raises:
        ValueError: if one of the names in joint_names does not exist in the joint list of the robot

    Returns:
        List(int): list of indexes 
    """
    name_list = robot_wrapper.model.names.tolist()
    indexes = []
    for joint_name in joint_names:
        try :
            index = name_list.index(joint_name)
        except:
            raise ValueError(f'{joint_name} does not match any of the joints name in the model')
        indexes.append(index)
    return indexes