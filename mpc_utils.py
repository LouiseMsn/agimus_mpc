import pinocchio as pin
import numpy as np
from pinocchio import SE3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d, RBFInterpolator


# class SplineGenerator:
#     def __init__(self, waypoints, duration, kernel='cubic'):
#         if len(waypoints) < 2:
#             raise ValueError("At least two waypoints are required.")
#         self.waypoints = np.array(waypoints)
#         self.duration = duration
#         self.time = np.linspace(0, duration, len(waypoints))[:, None]  # (N, 1)
#         self.rbf = RBFInterpolator(self.time, self.waypoints, kernel=kernel)

#     def interpolate(self, t):
#         """Return interpolated position at time t (clamped to [0, duration])"""
#         if t <= 0:
#             return self.waypoints[0]
#         if t >= self.duration:
#             return self.waypoints[-1]
#         return self.rbf(np.array([[t]]))[0]  # Output is (1,3)

# ==============================================================================

class PatternGenerator:
    """
    A class to generate patterns for glue spreading.
    """
    def __init__(self, object_dim:list):
        """
        Initialize the PatternGenerator with the dimensions of the object.

        :param object_dim: A tuple representing the dimensions of the object (length, width, height).
        :param object_center: A tuple representing the center of the object (x, y, z).
        """
        self.object_length = object_dim[0]  # Assuming object_dim is (length, width, height)
        self.object_width = object_dim[1]
        self.object_height = object_dim[2]
        self.object_center = (0, 0, 0)  # Default center, can be modified later

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
            print("yippie")

        return x, y, z

    # def transform_trajectory(self, traj, translation, rotation):
    #     """
    #     Applique une transformation c>omplète (rotation + translation)
    #     à une trajectoire exprimée en position dans le repère de tf.header.frame_id.

    #     - traj : tuple de listes (x_list, y_list, z_list)
    #     - tf : TransformStamped représentant la pose cible (rotation + translation)

    #     Retourne : liste de points transformés [(x', y', z'), ...]
    #     """
    #     transformed = []

    #     # Récupère quaternion et translation du TransformStamped
    #     q = rotation
    #     t = translation
    #     quat = [q.x, q.y, q.z, q.w]
    #     trans = np.array([t.x, t.y, t.z])

    #     # Matrice de transformation 4x4
    #     T = np.eye(4)
    #     T[:3, :3] = tf_transformations.quaternion_matrix(quat)[:3, :3]
    #     T[:3, 3] = trans

    #     for i, (x, y, z) in enumerate(zip(*traj)):
    #         # homogénéisation du point
    #         point = np.array([x, y, z, 1.0])
    #         transformed_point = T @ point
    #         transformed.append(tuple(transformed_point[:3]))

    #     return transformed


class SplineGenerator:
    def __init__(self, waypoints, start_pos, duration, kernel='cubic'):
        waypoints.insert(0,start_pos)
        if len(waypoints) < 2:
            raise ValueError("At least two waypoints are required.")
        self.waypoints = np.array(waypoints)
        self.duration = duration
        self.time = np.linspace(0, duration, len(waypoints))[:, None]  # Shape: (N, 1)
        self.rbf = RBFInterpolator(self.time, self.waypoints, kernel=kernel)
        self.last_valid_orientation_ref = np.array([0.0, 0.0, 0.0])  # Default to 0 yaw

    def interpolate_pose(self, t):
        """Return interpolated position at time t (clamped to [0, duration])"""
        t = np.clip(t, 0.0, self.duration)
        return self.rbf(np.array([[t]]))[0]  # Shape: (3,)

    def interpolate_ori(self, t, dt=0.01):
        """
        Compute the orientation based on the tangent (direction) of the trajectory.

        Args:
            t (float): current time
            dt (float): small time increment for computing direction vector

        Returns:
            orientation (np.array): [roll, pitch, yaw] in radians
        """
        t = np.clip(t, 0.0, self.duration - dt)
        t_next = np.clip(t + dt, dt, self.duration)

        current_point = self.rbf(np.array([[t]]))[0]
        next_point = self.rbf(np.array([[t_next]]))[0]
        direction_vector = next_point - current_point


        # Compute yaw angle from the direction in XY plane
        roll = 0.0
        pitch = 0.0
        yaw = np.arctan2(direction_vector[1], direction_vector[0])


        self.last_valid_orientation_ref = np.array([roll, pitch, yaw])
        return self.last_valid_orientation_ref




if __name__=="__main__":

    patternGen=PatternGenerator([1,1,0])
    x,y,z = patternGen.generate_pattern('zigzag_curve',stride=0.1)
    positions :list = []
    for i in range (len(x)):
        positions.append(np.array([x[i], y[i], z[i]]))



    # print(waypoints)
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


    print(positions)
    # positions = [ np.array([-0.5, -0.5,  0. ])]
    duration = 3
    spline = SplineGenerator(positions, [0, 0, 1], duration, kernel='cubic')

    traj = []
    dt = 0.01
    times = np.arange(0, duration + dt, dt)
    for t in times:
        pose = spline.interpolate_pose(t)
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
