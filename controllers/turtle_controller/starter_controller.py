"""student_controller controller."""
import matplotlib  
matplotlib.use('TkAgg')

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

N = 5 # number of known landmarks
landmark_dist_noise = .1
landmark_heading_noise = .05
WORLD_MIN = -2.5
WORLD_MAX = 2.5
VISUALIZE_STEPS = 20


class StudentController:
    def __init__(self):
        self._state_size = 2 # initially, we only need to know x and y of the robot and the number of landmarks
        self._pose = np.zeros(self._state_size) # know initial values of the pose is 0
        self._map_with_ordering = {} # key will be which landmark this is
        self._prev_variance = np.eye(self._state_size) * 0.01 # since know robot initial state, make the variance very small
        self._sigma = .005
        self._known_landmark = False
        self._visualize_count = 0
        self._fig, self._ax = plt.subplots()
        plt.ion()  # interactive mode on
        self._ax.set_xlim(-2.5, 2.5)
        self._ax.set_ylim(-2.5, 2.5)
        self._ax.set_aspect('equal')
        self._ax.grid(True)

    def step(self, sensors):
        """
        Compute robot control as a function of sensors.

        Input:
        sensors: dict, contains current sensor values.

        Output:
        control_dict:   dict, contains control for "left_motor" and "right_motor"
        estimated_pose: list, contains float values representing the robot's pose,
                        (x,y,orientation).
                        The pose should be given using a right-handed coordinate
                        system: positive x is the right side of the arena, positive
                        y is the top side of the arena, theta increases as the
                        robot turns counter-clockwise.
        """
        control_dict = {"left_motor": 0.0, "right_motor": 0.0}

        # TODO: add your controllers here.

        self._actual_theta = sensors["heading"]
        print("robot heading:", self._actual_theta)
        print("robot_odometry:", sensors["odometry"])
        # 1)  Prediction Step
        pose_prediction = self._pose.copy()
        
        pose_prediction, variance_prediction = self.prediction(sensors, pose_prediction)

        # 2)  Correction Step
        print(sensors["observed_landmarks"])
        pose_correction, variance_correction = self.correction(sensors, pose_prediction, variance_prediction)
        # pose_correction, variance_correction = pose_prediction, variance_prediction # testing prediction step

        # update to save values for next steps.
        self._pose = pose_correction
        self._prev_variance = variance_correction

        estimated_pose = pose_correction[0:2].tolist() + [self._actual_theta]
        estimated_map = self.array2dict()
        print("estimated pose: ", estimated_pose)
        print("estimated map: ", estimated_map)
        print("robot_variance:", variance_correction[0:2,0:2])

        # visualize again if hit step count
        if self._visualize_count % VISUALIZE_STEPS == 0:
            self.visualize_slam(self._pose, self._prev_variance, sensors)
            self._visualize_count = 0
        self._visualize_count += 1

        control_dict["left_motor"] = 1
        control_dict["right_motor"] = 1
        

        return control_dict, pose_correction, estimated_map
    
    # predicts both the pose and the variance
    def prediction(self, sensors, pose_prediction):
        # mu_bar_t, below is getting the estimate mean pose
        delta_distance, delta_theta = sensors["odometry"]
        pose_prediction[0] += delta_distance * math.cos(self._actual_theta) # update x position with new theta
        pose_prediction[1] += delta_distance * math.sin(self._actual_theta) # update y position with new theta
        print("pose_prediction[0]:", pose_prediction[0])
        print("pose_prediction[1]:", pose_prediction[1])

        # sigma_bar Below is to get the variance of the prediction
        G_r = np.array([[1, 0], 
                      [0, 1]]) # Jacobian of the motion model for robot

        # create entire jacobian 
        G = np.eye(self._state_size)      # start with identity
        G[0:2, 0:2] = G_r           # insert robot motion Jacobian
        
        variance_prediction = G @ self._prev_variance @ G.T
        variance_prediction_noise = np.eye(self._state_size) * self._sigma**2
        variance_prediction += variance_prediction_noise # add noise

        return pose_prediction, variance_prediction
    
    # get correction value for both pose and variance
    def correction(self, sensors, pose_prediction, variance_prediction):
        # 1) iterate through all landmarks and update matrixes through each step.
        variance_correction = variance_prediction
        pose_correction = pose_prediction
        for landmark_id, (landmark_dist, landmark_heading) in sensors["observed_landmarks"].items():

            # check if known or unknown correspondance
            if not self._known_landmark and landmark_id.startswith("BOX"):
                self._known_landmark = True

            # since never seen this landmark before, we will add it to the map and skip calculations
            if self._known_landmark:
                # 2) figure out which slot to put the landmark in
                if landmark_id not in self._map_with_ordering:
                    # 2a) first find the first place I can start iterating through
                    # assign next available slot
                    landmark_slot = len(self._map_with_ordering)
                    self._map_with_ordering[landmark_id] = landmark_slot

                    # compute index in state
                    landmark_start = 2 + 2 * landmark_slot

                    # 🔴 EXPAND STATE VECTOR
                    self._pose = np.concatenate((pose_correction, np.zeros(2)))
                    pose_correction = self._pose

                    # 🔴 EXPAND COVARIANCE MATRIX
                    old_size = variance_correction.shape[0]
                    new_size = old_size + 2

                    new_cov = np.zeros((new_size, new_size))
                    new_cov[:old_size, :old_size] = variance_correction
                    variance_correction = new_cov

                    self._state_size = new_size
                    # calculate where the robot thinks the landmark is knowing robot position and where the relative position from landmark is
                    pose_correction[landmark_start] = pose_correction[0] + landmark_dist * np.cos(landmark_heading + self._actual_theta)
                    pose_correction[landmark_start + 1] = pose_correction[1] + landmark_dist * np.sin(landmark_heading + self._actual_theta)
                    theta = self._actual_theta
                    r = landmark_dist
                    bearing = landmark_heading
 
                    Gz = np.array([
                        [np.cos(theta + bearing), -r * np.sin(theta + bearing)],
                        [np.sin(theta + bearing),  r * np.cos(theta + bearing)]
                    ]) # jacobian with respect to odometry readings

                    R = np.diag([landmark_dist_noise**2, landmark_heading_noise**2])

                    P_rr = variance_correction[0:2, 0:2]

                    P_ll = Gz @ R @ Gz.T + P_rr
                    
                    variance_correction[landmark_start:landmark_start+2, landmark_start:landmark_start+2] = P_ll # when new and just added, have more of a uncertainty
                    
                    # Robot covariance of how the robot is related to the the landmark
                    variance_correction[0:2, landmark_start:landmark_start+2] = P_rr
                    variance_correction[landmark_start:landmark_start+2, 0:2] = P_rr.T
                    
                    continue
                
                # if we have already seen this landmark before, just calculate this as before.
                landmark_slot = self._map_with_ordering[landmark_id]
                landmark_start = 2 + 2 * landmark_slot
                    
            else:
                # unknown correspondance course
                print("have not implemented this yet, but need to predict which landmark this is")
                # probably need to iterate through each pair of landmark x and y and see which one is the closest to it

            # 4) H_t, Jacobian of the observation model
            dx = pose_correction[landmark_start] - pose_correction[0]
            dy = pose_correction[landmark_start + 1] - pose_correction[1]
            r = math.sqrt(dx**2 + dy**2)

            H_t = np.zeros((2, self._state_size))

            H_t[:,0:2] = np.array([[-dx/r, -dy/r],
                            [dy/r**2, -dx/r**2]]) # set the robot H_t values
            
            H_t[:,landmark_start:landmark_start+2] = np.array([[dx/r, dy/r],
                                                          [-dy/r**2, dx/r**2]]) # set the landmark H_t value
            
            # 5) K_t, solving for gain for correction step
            R = np.diag([landmark_dist_noise**2, landmark_heading_noise**2]) # observation noise covariance that is gotten from look at code in turtle_controller.py
            K_t = variance_correction @ H_t.T @ np.linalg.inv(H_t @ variance_correction @ H_t.T + R)

            # 6) solve for sigma_t or the correction variance
            variance_correction = (np.eye(self._state_size) - K_t @ H_t) @ variance_correction

            # 7) solve for mu_t or the correction pose
            # predicted measurement h_predict
            h_predict = np.array([
                r,
                np.arctan2(dy, dx) - self._actual_theta
            ])
            # observation / measured landmark
            z_hat = np.array([
                landmark_dist,
                landmark_heading
            ])

            error = z_hat - h_predict
            error[1] = np.arctan2(np.sin(error[1]), np.cos(error[1]))

            pose_correction = pose_correction + K_t @ error # this is updating mu_t term

        return pose_correction, variance_correction

    # Creates map from numpy array
    def array2dict(self):
        estimated_map = {}

        # invert mapping: slot → landmark_id
        slot_to_landmark = {v: k for k, v in self._map_with_ordering.items()}

        for slot, landmark_id in slot_to_landmark.items():
            landmark_idx = 2 + 2 * slot
            estimated_map[landmark_id] = [
                self._pose[landmark_idx],
                self._pose[landmark_idx + 1]
            ]

        return estimated_map

    def plot_covariance_ellipse(self, mean, cov, ax, n_std=2.0, color='blue'):
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            return

        eigenvals, eigenvecs = np.linalg.eig(cov)

        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])

        width = 2 * n_std * np.sqrt(max(eigenvals[0], 1e-9))
        height = 2 * n_std * np.sqrt(max(eigenvals[1], 1e-9))

        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=np.degrees(angle),
            edgecolor=color,
            fc='None',
            lw=2
        )

        ax.add_patch(ellipse)        

    def visualize_slam(self, state, covariance, sensors):
        ax = self._ax
        ax.cla()  # clear previous frame

        # --- DRAW BOUNDARY BOX (5x5 centered at origin) ---
        rect = plt.Rectangle((-2.5, -2.5), 5, 5, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # --- Robot ---
        x, y = state[0], state[1]
        theta = sensors["heading"]
        ax.plot(x, y, 'ro', label="Robot")
        ax.arrow(x, y, 0.3*np.cos(theta), 0.3*np.sin(theta), head_width=0.1, color='r')
        self.plot_covariance_ellipse([x, y], covariance[0:2, 0:2], ax, color='red')

        # --- Landmarks ---
        for landmark_id, slot in self._map_with_ordering.items():
            idx = 2 + 2 * slot

            lx = state[idx]
            ly = state[idx+1]

            ax.plot(lx, ly, 'bs')

            # extract number (Box_4 → 4)
            try:
                label_num = landmark_id.split('_')[1]
            except:
                label_num = landmark_id

            ax.text(lx + 0.05, ly + 0.05, f"L{label_num}")

            cov_ll = covariance[idx:idx+2, idx:idx+2]
            self.plot_covariance_ellipse([lx, ly], cov_ll, ax, color='blue')

            ax.plot([x, lx], [y, ly], 'g--', alpha=0.3)

        # --- Observations ---
        for _, (dist, bearing) in sensors["observed_landmarks"].items():
            obs_x = x + dist * np.cos(theta + bearing)
            obs_y = y + dist * np.sin(theta + bearing)
            ax.plot(obs_x, obs_y, 'gx')

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()

        plt.draw()
        plt.pause(0.001)  # tiny pause to allow GUI to update