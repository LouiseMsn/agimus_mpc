import pinocchio as pin
from pinocchio.rpy import rpyToMatrix, matrixToRpy
import numpy as np
from pinocchio import SE3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d, RBFInterpolator
import math
from copy import deepcopy


class PatternGenerator:
    """
    A class to generate patterns for glue spreading.
    """
    def __init__(self, object_dim:list, object_center):
        """
        Initialize the PatternGenerator with the dimensions of the object.

        :param object_dim: A tuple representing the dimensions of the object (length, width, height).
        :param object_center: A tuple representing the center of the object (x, y, z).
        """
        self.object_length = object_dim[0]  # Assuming object_dim is (length, width, height)
        self.object_width = object_dim[1]
        self.object_height = object_dim[2]
        self.object_center = object_center #(0, 0, 0)  # Default center, can be modified later

    def generate_pattern(self, pattern_type, step=10, stride=0.2, orientation='vertical'):
        if pattern_type == 'zigzag':
            return self.zig_zag_SE3(step=step, stride=stride, orientation=orientation)
        elif pattern_type == 'spiral':
            # return self.spiral_from_center(stride=stride)
            raise ValueError("This option is not working for now")
        elif pattern_type == 'zigzag_curve':
            return self.zig_zag_curve_SE3(step=step, stride=stride, orientation=orientation)
        else:
            raise ValueError("Unknown pattern type")
        
    def zig_zag_SE3(self, step=10, stride=0.2, orientation='vertical'):
        positions = self.zigzag(step, stride, orientation)
        points = add_orientation(positions, np.array([np.pi, 0., 0.]))
        return points
    
    def zigzag(self, step=10, stride=0.2, orientation='vertical'):
        """
        Draws a zigzag pattern.

        Parameters:
        - start: starting point (x, y, z)
        - length: length of each zigzag segment
        - nb: number of zigzag segments
        - step: number of points per segment
        - stride: offset between segments
        - orientation: 'vertical', 'horizontal', or 'diagonal'
        """
        x, y, z = [], [], []
        if (orientation == 'vertical'):
            x_tmp, y_tmp, z_tmp = self.object_center[0] - self.object_length / 2, \
                self.object_center[1] - self.object_width / 2, \
                self.object_center[2] + self.object_height / 2
            num_lines = int(self.object_length // stride)
            if self.object_length % stride == 0:
                num_lines += 1
        elif (orientation == 'horizontal'):
            x_tmp, y_tmp, z_tmp = self.object_center[0] - self.object_length / 2, \
                self.object_center[1] - self.object_width / 2, \
                self.object_center[2] + self.object_height / 2
            # Ensure the last line is on the edge if division is exact
            num_lines = int(self.object_width / stride)
            if self.object_width % stride == 0:
                num_lines += 1
        else:
            raise ValueError("orientation must be 'vertical', 'horizontal'")

        for i in range(num_lines):
            direction = 1 if i % 2 == 0 else -1  # alternate direction

            for j in range(step + 1):
                if orientation == 'vertical':
                    x.append(x_tmp)
                    y.append(y_tmp + direction * j * self.object_width / step)
                    z.append(z_tmp)
                elif orientation == 'horizontal':
                    x.append(x_tmp + direction * j * self.object_length / step)
                    y.append(y_tmp)
                    z.append(z_tmp)

            if orientation == 'vertical':
                x_tmp += stride
                y_tmp = y[-1]  # continue from last y
                z_tmp = z[-1]
            elif orientation == 'horizontal':
                y_tmp += stride
                x_tmp = x[-1]  # continue from last x
                z_tmp = z[-1]
        return [np.array([x[i],y[i],z[i]]) for i in range(len(x))]

    def zig_zag_curve_SE3(self, step=10, stride=0.2, orientation='vertical'):
        positions = self.zig_zag_curve_pos(step, stride, orientation)
        points = add_orientation(positions, None)
        return points

    def zig_zag_curve_pos(self, step=10, stride=0.2, orientation='vertical'):
        """
        Dessine un motif en zigzag avec des courbes aux extrémités.

        Parameters:
        - step: nombre de points par segment
        - stride: décalage entre les segments
        - orientation: 'vertical' ou 'horizontal'
        - radius: rayon des courbes aux extrémités
        """
        x, y, z = [], [], []
        if (orientation == 'vertical'):
            x_tmp, y_tmp, z_tmp = self.object_center[0] - self.object_length / 2, \
                self.object_center[1] - self.object_width / 2, \
                self.object_center[2] + self.object_height / 2
            num_lines = int(self.object_length // stride)
            if self.object_length % stride == 0:
                num_lines += 1
        elif (orientation == 'horizontal'):
            x_tmp, y_tmp, z_tmp = self.object_center[0] - self.object_length / 2, \
                self.object_center[1] - self.object_width / 2, \
                self.object_center[2] + self.object_height / 2
            # Ensure the last line is on the edge if division is exact
            num_lines = int(self.object_width / stride)
            if self.object_width % stride == 0:
                num_lines += 1
        else:
            raise ValueError("orientation must be 'vertical', 'horizontal'")

        for i in range(num_lines):
            direction = 1 if i % 2 == 0 else -1  # alternate direction
            if i == 0 or i == num_lines - 1:
                start_step = 0
                end_step = step
            else:
                start_step = 0
                end_step = step-1

            for j in range(start_step, end_step):
                if orientation == 'vertical':
                    x.append(x_tmp)
                    y.append(y_tmp + direction * j * self.object_width / step)
                    z.append(z_tmp)
                elif orientation == 'horizontal':
                    x.append(x_tmp + direction * j * self.object_length / step)
                    y.append(y_tmp)
                    z.append(z_tmp)

            if i < num_lines - 1:  # évite d’ajouter une courbe après la dernière ligne
                if orientation == 'vertical':
                    x_tmp += stride
                    y_tmp = y[-1]
                    z_tmp = z[-1]
                    dir_str = '+y' if direction == 1 else '-y'
                    curve_start = (x_tmp - stride, y_tmp, z_tmp)
                else:
                    y_tmp += stride
                    x_tmp = x[-1]
                    z_tmp = z[-1]
                    dir_str = '+x' if direction == 1 else '-x'
                    curve_start = (x_tmp, y_tmp - stride, z_tmp)

                # Ajout de la courbe de transition
                traj_curve = self.curve_arc(
                    start_point=curve_start,
                    radius=stride / 2,
                    dir=dir_str,
                    step=step // 2
                )
                x_curve = [traj_curve[i][0] for i in range(len(traj_curve))]
                y_curve = [traj_curve[i][1] for i in range(len(traj_curve))]
                z_curve = [traj_curve[i][2] for i in range(len(traj_curve))]
                x.extend(x_curve)
                y.extend(y_curve)
                z.extend(z_curve)

        return [np.array([x[i],y[i],z[i]]) for i in range(len(x))]

    import numpy as np

    def curve_arc(self, start_point, radius, dir, angle=np.pi, step=10):
        """
        Dessine un arc de cercle dans le plan XY.

        Args:
            start_point (tuple): point de départ (x, y, z)
            radius (float): rayon de la courbe
            dir (str): direction de la courbe ('+x', '-x', '+y', '-y')
            angle (float): angle de l'arc en radians (par défaut = pi/2 = 90°)
            step (int): nombre de points pour dessiner la courbe

        Returns:
            x, y, z (list): listes des coordonnées des points de la courbe
        """
        x, y, z = [], [], []
        angle_step = angle / step

        for i in range(1, step):
            theta = i * angle_step
            if dir == '+x':
                xi = start_point[0] + radius * np.sin(theta)
                yi = start_point[1] + radius * (1 - np.cos(theta))
            elif dir == '-x':
                xi = start_point[0] - radius * np.sin(theta)
                yi = start_point[1] + radius * (1 - np.cos(theta))
            elif dir == '+y':
                xi = start_point[0] + radius * (1 - np.cos(theta))
                yi = start_point[1] + radius * np.sin(theta)
            elif dir == '-y':
                xi = start_point[0] + radius * (1 - np.cos(theta))
                yi = start_point[1] - radius * np.sin(theta)
            else:
                raise ValueError("dir must be '+x', '-x', '+y', or '-y'")

            x.append(xi)
            y.append(yi)
            z.append(start_point[2])  # z constant

        return [np.array([x[i],y[i],z[i]]) for i in range(len(x))]

    # this one is not working as intended
    def spiral_from_center(self, stride=1.0):
        """
        Génère une spirale polygonale (carrée) qui part du centre et s'étend vers l'extérieur,
        sans dépasser les dimensions de l'objet.

        Args:
            stride (float): espacement entre chaque "tour"

        Returns:
            x, y, z: listes des coordonnées des points de la spirale
        """
        # Initialisation
        x = [self.object_center[0]]
        y = [self.object_center[1]]
        z = [self.object_center[2] + self.object_height / 2]

        # Limites de l'objet
        x_min = self.object_center[0] - self.object_length / 2
        x_max = self.object_center[0] + self.object_length / 2
        y_min = self.object_center[1] - self.object_width / 2
        y_max = self.object_center[1] + self.object_width / 2

        # Directions: droite, haut, gauche, bas
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        lengths = [self.object_length, self.object_width]  # longueur côté courant
        current_lengths = [stride, stride]  # longueur à parcourir pour chaque direction
        dir_idx = 0  # index de direction
        steps = 0

        while True:
            dx, dy = directions[dir_idx % 4]
            # Détermine la longueur maximale possible sans dépasser les bords
            if dx != 0:
                # Mouvement en x
                if dx > 0:
                    max_len = min(current_lengths[0], x_max - x[-1])
                else:
                    max_len = min(current_lengths[0], x[-1] - x_min)
            else:
                # Mouvement en y
                if dy > 0:
                    max_len = min(current_lengths[1], y_max - y[-1])
                else:
                    max_len = min(current_lengths[1], y[-1] - y_min)

            # Si la longueur à parcourir est nulle ou négative, on s'arrête
            if max_len <= 0:
                break

            # Ajoute le nouveau point
            new_x = x[-1] + dx * max_len
            new_y = y[-1] + dy * max_len
            x.append(new_x)
            y.append(new_y)
            z.append(z[-1])

            # Prépare la longueur pour le prochain tour
            if dir_idx % 2 == 1:
                current_lengths[0] += stride
                current_lengths[1] += stride

            dir_idx += 1
            steps += 1

            # Arrête si on touche les bords
            if not (x_min <= new_x <= x_max and y_min <= new_y <= y_max):
                break

        return [np.array([x[i],y[i],z[i]]) for i in range(len(x))]


class Interpolator:
    def my_log6(self, M : pin.SE3):
        return pin.log6(M)
    
    def my_exp6(self, twist : pin.Motion):
        return pin.exp6(twist)
    
    def my_log3(self, M: pin.SE3):
        twist = pin.Motion()
        twist.linear = M.translation
        twist.angular = pin.log3(M.rotation)
        return twist
    
    def my_exp3(self, twist : pin.Motion):
        M = pin.SE3()
        M.translation = twist.linear
        M.rotation = pin.exp3(twist.angular)
        return M
    
    def my_dist(self, a, b):
        twist = self.my_log3(a.actInv(b))
        weight = np.array([1.]*3 + [0.1]*3)
        return np.linalg.norm(twist*weight)

    def __init__(self, waypoints, speed ):
        self.waypoints = waypoints
        self.v = speed
        self.distances = self.getDistances()
        self.d_total = sum(self.distances)
        self.dt = [d / self.v for d in self.distances]
        self.t_total = sum(self.dt)

    def getDistances(self):
        """Calculates the list of the distances between two consecutive points with accounts to translation and rotation.

        Returns:
            List(float): list of the distances
        """
        distances = []
        for i in range(len(self.waypoints)-1):
            current_pt = self.waypoints[i]
            next_pt = self.waypoints[i+1]
            d_segment = self.my_dist(current_pt, next_pt)
            distances.append(d_segment)
        return distances
    
    def __call__(self, t):
        if t > self.t_total:
            return (self.waypoints[-1] , pin.Motion(np.zeros(6)))  # (last waypoint , null vel)
        else: 
            i = 0
            while t > self.dt[i]:
                t = t - self.dt[i]
                i += 1

            current_pt = self.waypoints[i]
            next_pt = self.waypoints[i+1]
            
            # A = pin.log6(current_pt.actInv(next_pt)) 
            # B = pin.exp6(alpha * A)

            # linear interpolation
            
            twist_local = pin.Motion(self.my_log6(current_pt.actInv(next_pt))/self.dt[i]) # speed to apply to go from current_p     
        
            pose_local = self.my_exp6(twist_local * t) # integrate the twist over t to get the transformation from current point to point(t)

            pose_world =  current_pt.act(pose_local) # apply the transformation to the current point to get the pose of point(t)
            twist_world = pose_world.act(twist_local)

            return (pose_world, twist_world)
                
class SplineGenerator:
    """
    Class that interpolates the `waypoints` into a spline and calculates the orientation between them so that the X axis faces the nex waypoint and the Z axis faces downward
    """
    def __init__(self, start_pos, start_ori, waypoints,
                 v_start=1.0, v_spread=1.0,
                 start_kernel='linear', spread_kernel='cubic'):
        if len(waypoints) < 2:
            raise ValueError("At least two waypoints are required.")

        self.start_pos = np.array(start_pos)
        self.start_ori = np.array(start_ori)
        self.waypoints = np.array(waypoints)
        self.v_start = v_start
        self.v_spread = v_spread
        self.start_kernel = start_kernel
        self.spread_kernel = spread_kernel

        self.start_traj = None
        self.spread_traj = None
        self.last_valid_orientation_ref = np.array([0.0, 0.0, 0.0])
        self.t_total = 0

        self._compute_and_set_times()
        self._compute_full_traj()
        self._compute_full_ori()

    def _compute_and_set_times(self):
        """
        Computes the time it will take to execute the start, spread and total trajectories
        Args:
            None
        Returns:
            None
        Sets:
            self.t_start : (float) time from the current pose of the end effector `self.start_pos` to the first waypoint `waypoint[0]` at a `self.v_start` speed
            self.t_spread : (float) time from the first waypoint to the last at a `self.v_spread` speed
            self.t_total : (float) time of the complete trajectory
        """
        # Distance for first segment
        d_start = np.linalg.norm(self.waypoints[0] - self.start_pos)
        t_start = d_start / self.v_start

        # Distances for the spread segment
        d_spread = np.sum([
            np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
            for i in range(len(self.waypoints) - 1)
        ])

        t_spread = d_spread / self.v_spread

        # Save time partitions
        self.t_start = t_start
        self.t_total = t_start + t_spread

        # Create time arrays for both segments
        self.time_start = np.array([0.0, self.t_start])[:, None]
        self.time_spread = np.linspace(self.t_start, self.t_total, len(self.waypoints))[:, None]

    def _compute_start_trj(self):
        """
        Initiates the interpolator for the trajectory between the start pose and the first waypoint

        Args:
            None
        Returns:
            None
        Sets:
            self.start_traj (RBFInterpolator)
        """
        points = np.vstack([self.start_pos, self.waypoints[0]])
        self.start_traj = RBFInterpolator(self.time_start, points, kernel=self.start_kernel)

    def _compute_spread_traj(self):
        """
        Initiates the interpolator for the trajectory between the first waypoint and the last

        Args:
            None
        Returns:
            None
        Sets:
            self.spread_traj (RBFInterpolator)
        """
        points = self.waypoints[0:]
        self.spread_traj = RBFInterpolator(self.time_spread, points, kernel=self.spread_kernel)

    def _compute_full_traj(self):
        """
        Initiates the interpolators for the trajectory between the start pose and the first waypoint and between the first waypoint and the last.

        Args:
            None
        Returns:
            None
        Sets:
            self.start_traj : (RBFInterpolator) interpolator for the pose of the start trajectory
            self.spread_traj : (RBFInterpolator) interpolator for the pose of the spread trajectory
        """
        self._compute_start_trj()
        self._compute_spread_traj()

    def get_interpolated_pose(self, t):
        """
        Returns the pose of the end effector at a given `t` (secs)
        Args:
            t : time in seconds (float)
        Returns:
            pose : (list[float]) pose of end effector for t
        Sets:
            None
        """
        t = np.clip(t, 0.0, self.t_total)
        if t <= self.t_start:
            return self.start_traj(np.array([[t]]))[0]
        else:
            return self.spread_traj(np.array([[t]]))[0]

    def _compute_start_orientation(self, spread_orientations):
        """
        Instanciates the interpolator from the start orientation to the orientation of the first pattern waypoint
        Args:
            spread_orientations : list of the orientations for the spreading part of the trajectory
        Returns:
            None
        Sets:
            start_ori_traj : (RBFInterpolator) interpolator for the orientation of the start trajectory
        """
        orientations = np.vstack((self.start_ori, spread_orientations[0]))
        self.start_ori_traj = RBFInterpolator(self.time_start, orientations, kernel=self.start_kernel)

    def _compute_spread_orientation(self):
        """
        Computes the orientation of the end effector waypoint by waypoint so that the X axis points to the next waypoint and the Z axis points down.
        Also initializes the interpolator between the first waypoint to the last.
        Args:
            None
        Returns:
            spread_orientations : (list[float]) list of orientations for the spread part of the trajectory
        Sets:
            start_ori_traj : (RBFInterpolator) interpolator for the orientation of the spread trajectory
        """
        points = self.waypoints[0:]
        spread_orientations = []
        for i in range(len(points)-1):
            spread_orientations.append(self._compute_local_orientation(points[i], points[i+1]))
        spread_orientations.append(spread_orientations[-1]) # repeat last orientation to match the number of waypoints
        self.spread_ori_traj = RBFInterpolator(self.time_spread, spread_orientations, kernel=self.spread_kernel)
        return spread_orientations

    def _compute_local_orientation(self, current_point, next_point):
        """
        Computes the orientation between two points so that the X axis points from the `current_point` to the `next_point` and the Z axis points down
        Args:
            None
        Returns:
            orientation : (list[float]) Roll Pitch Yaw orientation of the effector
        Sets:
            None
        """
        direction_vector = next_point - current_point
        roll = np.pi
        pitch = 0
        yaw = np.arctan2(direction_vector[1], direction_vector[0])
        orientation = np.array([roll, pitch, yaw])
        return orientation

    def _compute_full_ori(self):
        """
        Computes the orientation for the full trajectory
        Args:
            None
        Returns:
            None
        Sets:
            None
        """
        spread_ori = self._compute_spread_orientation()
        self._compute_start_orientation(spread_ori)

    def get_interpolated_ori(self,t):
        """
        Returns the orientation of the end effector at a given `t` (secs)
        Args:
            t : time in seconds (float)
        Returns:
            orientation : (list[float]) Roll Pitch Yaw of the end effector
        Sets:
            None
        """
        t = np.clip(t, 0.0, self.t_total)
        if t <= self.t_start:
            return self.start_ori_traj(np.array([[t]]))[0]
        else:
            return self.spread_ori_traj(np.array([[t]]))[0]

class TestTrajs:
    # "struct" class used to regroup test trajectory generators
    def line(self, start_point:list, end_point:list):
        # add check to verify if point is valid
        if len(start_point)<3 :
            raise ValueError("Wrongly defined start point")
        elif len(end_point)<3:
            raise ValueError("Wrongly defined end point")
        else:
            trajectory = np.array([start_point, end_point])
        return trajectory

    def sine(self, amplitude=1.0, period=1.0, sine_axis="z", ampl_axis="x", start_point=[1,1,1], length=1.0, dist_between_points = 0.1):

        if sine_axis not in ["x", "X", "y", "Y", "z", "Z"]:
            raise ValueError("Invalid sine_axis value")
        elif ampl_axis not in ["x", "X", "y", "Y", "z", "Z"]:
            raise ValueError("Invalid sine_axis value")

        indexDict = {"x":0,"X":0, "y":1, "Y":1, "z":2,"Z":2} # to get the translation axis name -> array index
        trajectory = []

        axis_start_point = start_point[indexDict[sine_axis]]
        axis_stop_point = axis_start_point + length
        number_of_points = abs(int((axis_stop_point - axis_start_point) / dist_between_points))
        i_table = np.linspace(start=axis_start_point, stop= axis_stop_point,num = number_of_points)
        i_table = np.linspace(start=0,stop=length, num=number_of_points)

        for i in i_table:
            sin_val = amplitude * math.sin((1/period) * i) # calculate the sin value for the current i
            current_point = deepcopy(start_point) # copy the start point
            current_point[indexDict[sine_axis]] += sin_val # add the sine value to the correct axis
            current_point[indexDict[ampl_axis]]+=i

            trajectory.append(np.array(current_point))

        
        points = add_orientation(trajectory, None)
        return points

        # amplitude * sin(period*(x-length_offset)) + height_offset

def RPY2Mat(roll,pitch,yaw):
    """
    Converts a rotation in roll pitch yaw to a rotation matrix. # TODO Pin function?
    Args:
        roll, pitch, yaw : (float)
    Returns:
        R : (np.array 3x3) Rotation matrix
    """

    Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
    Ry = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
    Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])
    R = Rz @ Ry @ Rx
    return R

def draw_frame(ax, pose: SE3,scale=[1, 1, 1]):
    """
    Draws 3 arrows in the `ax` plot showing XYZ pose and RPY rotation.
    Args:
        ax : (Axes) plot
    Returns:
        None
    """
    origin = pose.translation
    R = pose.rotation

    colors = ['r', 'g', 'b']
    for i in range(3):
        axis = R[:, i]  # vecteur direction
        ax.quiver(
            origin[0], origin[1], origin[2],
            axis[0]*scale[i], axis[1]*scale[i], axis[2]*scale[i],
            color=colors[i], arrow_length_ratio = 0.01, length=0.05
        )

def add_orientation(positions, orientation):
    points = []
    for i in range(len(positions)-1):
        if type(orientation) == type(np.empty(1)):
            ori = rpyToMatrix(orientation)
        else:
            ori = computeMatrixOrientation(positions[i], positions[i+1])
        point = pin.SE3(ori, positions[i])
        points.append(point)
    return points


def computeMatrixOrientation(current_point, next_point):
    direction_vector = next_point - current_point
    roll = np.pi
    pitch = 0
    yaw = np.arctan2(direction_vector[1], direction_vector[0]) + np.pi/2
    orientation = rpyToMatrix(roll, pitch, yaw)
    return orientation

if __name__=="__main__":
    start = pin.SE3(rpyToMatrix(-3.14128088,  0.05769075,  0.00540671), np.array([ 2.99996436e-01, -1.34822114e-07,  4.60813723e-01]))
    patternGen = PatternGenerator([0.5,0.5,0], (0.5, 0,0.1))
    # positions = [start] + patternGen.generate_pattern('zigzag_curve',stride=0.1)
    patternGen = PatternGenerator([0.3,0.3,0], (0.5, 0,0.1))
    positions = [start] +  patternGen.generate_pattern('zigzag',stride=0.05)
    
    # test_trajs = TestTrajs()
    # startsin = [0.3, -0., 0.2]
    # positions = test_trajs.sine(start_point=startsin,length=1,period=0.05,amplitude=0.1, dist_between_points=0.01, sine_axis="Y", ampl_axis="X")
    interpolator = Interpolator(positions, 0.1)

    # Debug of trajectory of adding orientation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    while i < len(positions):
        orientation = matrixToRpy(positions[i].rotation)
        print(orientation)
        roll = orientation[0]
        pitch = orientation[1]
        yaw = orientation[2]
        pose = positions[i].translation
        R = RPY2Mat(roll, pitch, yaw)
        print((R))
        print((pose))
        pose_6d = pin.SE3(R, pose)
        draw_frame(ax, pose_6d)
        ax.scatter(*pose, marker="^", c="r",alpha=0.5,s=15)
        i += 1

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Interpolation position + orientation 3D (RBF)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Debug of trajectory of adding orientation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    t = 0
    while t <= interpolator.t_total:
        point, _ = interpolator(t)
        orientation = matrixToRpy(point.rotation)
        print(orientation)
        roll = orientation[0]
        pitch = orientation[1]
        yaw = orientation[2]
        pose = point.translation
        R = RPY2Mat(roll, pitch, yaw)
        print((R))
        print((pose))
        pose_6d = pin.SE3(R, pose)
        draw_frame(ax, pose_6d)
        ax.scatter(*pose, marker="^", c="r",alpha=0.5,s=15)
        t += 0.1

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Interpolation position + orientation 3D (RBF)")
    ax.legend()
    plt.tight_layout()
    plt.show()
