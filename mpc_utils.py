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

        Retourne:
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

        return x, y, z

class SplineGenerator:
    def __init__(self, start, waypoints,
                 v_start=1.0, v_spread=1.0,
                 start_kernel='linear', spread_kernel='cubic'):
        if len(waypoints) < 2:
            raise ValueError("At least two waypoints are required.")

        self.start = np.array(start)
        self.waypoints = np.array(waypoints)
        self.v_start = v_start
        self.v_spread = v_spread
        self.start_kernel = start_kernel
        self.spread_kernel = spread_kernel

        self.start_traj = None
        self.spread_traj = None
        self.last_valid_orientation_ref = np.array([0.0, 0.0, 0.0])

        self._compute_times()
        self.compute_full_traj()

    def _compute_times(self):
        # Distance for first segment
        d_start = np.linalg.norm(self.waypoints[0] - self.start)
        t_start = d_start / self.v_start

        print("time to go from start to waypoint 0 " + str(t_start))

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

    def compute_start_trj(self):
        points = np.vstack([self.start, self.waypoints[0]])
        self.start_traj = RBFInterpolator(self.time_start, points, kernel=self.start_kernel)

    def compute_spread_traj(self):
        points = self.waypoints[0:]
        self.spread_traj = RBFInterpolator(self.time_spread, points, kernel=self.spread_kernel)

    def compute_full_traj(self):
        self.compute_start_trj()
        self.compute_spread_traj()

    def interpolate_pose(self, t):
        t = np.clip(t, 0.0, self.t_total)

        if t <= self.t_start:
            return self.start_traj(np.array([[t]]))[0]
        else:
            return self.spread_traj(np.array([[t]]))[0]

    def interpolate_ori(self, t, dt=0.01):
        t = np.clip(t, 0.0, self.t_total - dt)
        t_next = np.clip(t + dt, dt, self.t_total)
        print(t)
        print(self.t_start)
        if t <= self.t_start:
            return np.array([np.pi, 0., 0.])
        else:
            current_point = self.interpolate_pose(t)
            next_point = self.interpolate_pose(t_next)
            direction_vector = next_point - current_point

            roll = np.pi
            pitch = 0.0
            yaw = np.arctan2(direction_vector[1], direction_vector[0])

            self.last_valid_orientation_ref = np.array([roll, pitch, yaw])
            return self.last_valid_orientation_ref


if __name__=="__main__":
    patternGen = PatternGenerator([1,1,0], (0.5,0,0))
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


    # positions = [ np.array([1, 0,  0. ]), np.array([1.1,0,0]), np.array([1.2,0,0])]
    # positions = [np.array([0.5, 0.0, 0.5])]


    duration = 10
    # spline = SplineGenerator(positions, [0, 0, 1], duration, start_kernel='cubic', spread_kernel='cubic')
    spline = SplineGenerator(start=[-1,-1,0],waypoints=positions,v_start=1.0, v_spread=1.0)


    traj = []
    dt = 0.01
    times = np.arange(0, duration + dt, dt)
    for t in times:
        pose = spline.interpolate_pose(t)
        print(spline.interpolate_ori(t))
        traj.append(pose)
    traj = np.array(traj)

    wp_pos = np.array(positions)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label="Trajectoire interpolée", marker=".", color='blue',alpha=0.5)
    ax.scatter(wp_pos[:, 0], wp_pos[:, 1], wp_pos[:, 2], label="Waypoints", color='red', s=50)
    # ax.plot(x,y,z, label="Pattern generator", marker=".", color='cyan')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Interpolation position 3D (RBF)")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()
