import pinocchio as pin
import numpy as np
from pinocchio import SE3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d, RBFInterpolator


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
            return self.zigzag(step=step, stride=stride, orientation=orientation)
        elif pattern_type == 'spiral':
            return self.spiral_from_center(stride=stride)
        elif pattern_type == 'zigzag_curve':
            return self.zig_zag_curve(step=step, stride=stride, orientation=orientation)
        else:
            raise ValueError("Unknown pattern type")

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
        return x, y, z

    def zig_zag_curve(self, step=10, stride=0.2, orientation='vertical'):
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
                x_curve, y_curve, z_curve = self.curve_arc(
                    start_point=curve_start,
                    radius=stride / 2,
                    dir=dir_str,
                    step=step // 2
                )
                x.extend(x_curve)
                y.extend(y_curve)
                z.extend(z_curve)

        return x, y, z


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

        return x, y, z

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
            print("yippie")

        return x, y, z


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
        print(len(self.time_spread))
        print(len(spread_orientations))
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

if __name__=="__main__":
    patternGen = PatternGenerator([1,1,0], (0.5,0,0.2))
    x,y,z = patternGen.generate_pattern('zigzag_curve',stride=0.5)
    positions :list = []
    for i in range (len(x)):
        positions.append(np.array([x[i], y[i], z[i]]))

    # positions = [np.array([0.5, 0.0, 0.2]),
    #             np.array([ 0.5, 0.0, 0.5]),
    #             np.array([0.35, 0.35, 0.5]),
    #             np.array([0.35, 0.35, 0.2]),
    #             np.array([0.0, 0.5, 0.2]),
    #             np.array([0.0, 0.5, 0.5]),
    #             np.array([-0.35, 0.35, 0.5]),
    #             np.array([-0.35, 0.35, 0.2]),
    #             np.array([-0.5, 0.0, 0.2]),
    #             np.array([-0.5, 0.0, 0.5]),
    #             np.array([-0.35, -0.35, 0.5]),
    #             np.array([-0.35, -0.35, 0.2]),
    #             np.array([0.0, -0.5, 0.2]),
    #             np.array([0.0, -0.5, 0.5]),
    #             np.array([0.35, -0.35,  0.5]),
    #             np.array([0.35, -0.35,  0.2]),
    #             np.array([0.5, 0.0, 0.2])]

    duration = 7
    start_pose = [ 3.33970764e-01, -3.08143602e-16,  5.40159383e-01]
    start_ori = [ 2.77295717, -0.34585912 , 0.85050551]
    spline = SplineGenerator(start_pose, start_ori, waypoints=positions)

    # Debug of trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    t = 0
    while t < spline.t_total:
        orientation = spline.get_interpolated_ori(t)
        roll = orientation[0]
        pitch = orientation[1]
        yaw = orientation[2]
        pose = spline.get_interpolated_pose(t)
        R = RPY2Mat(roll, pitch, yaw)
        print((R))
        print((pose))
        pose_6d = pin.SE3(R, pose)
        draw_frame(ax, pose_6d)
        ax.scatter(*pose, marker="^", c="r",alpha=0.5,s=15)
        t = t + 0.02


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Interpolation position + orientation 3D (RBF)")
    ax.legend()
    plt.tight_layout()
    plt.show()


    # traj = []
    # dt = 0.01
    # times = np.arange(0, duration + dt, dt)
    # for t in times:
    #     pose = spline.interpolate_pose(t)
    #     traj.append(pose)
    #     # print("pose:" + str(pose))
    # traj = np.array(traj)
    # wp_pos = np.array(positions)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="Trajectoire interpolée", marker=".", color='blue',alpha=0.5)
    # ax.scatter(wp_pos[:, 0], wp_pos[:, 1], wp_pos[:, 2], label="Waypoints", color='red', s=50)
    # # ax.plot(x,y,z, label="Pattern generator", marker=".", color='cyan')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title("Interpolation position 3D (RBF)")
    # ax.legend()
    # ax.set_box_aspect([1, 1, 1])
    # plt.tight_layout()
    # plt.show()
