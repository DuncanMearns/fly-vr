import numpy as np


class ActuatorStimulus:

    def __init__(self, config: dict):
        self.config = config

    @property
    def sample_rate(self):
        return self.config.get('update_rate', 60.)

    @property
    def x_max(self):
        return self.config['actuator_limits']["x"]

    @property
    def y_max(self):
        return self.config['actuator_limits']["y"]

    @property
    def actuator_x0(self):
        return self.config['actuator_origin']["x"]

    @property
    def actuator_y0(self):
        return self.config['actuator_origin']["y"]

    @property
    def dy_mm(self):
        return (self.config["male_px"]["y"] - self.config["female_px"]["y"]) / self.config["scale_factor"]

    @property
    def dz_mm(self):
        return (self.config["female_px"]["z"] - self.config["male_px"]["z"]) / self.config["scale_factor"]

    def polar_to_actuator(self, polar_coords):
        # Account for elevation
        dists = np.clip(polar_coords[:, 1], self.dz_mm, None)
        dists_y = np.sqrt(np.square(dists) - np.square(self.dz_mm))
        # Compute male position
        male_xy = np.array([self.actuator_x0, self.actuator_y0 + self.dy_mm])
        # Clip distances
        dists_y = np.clip(dists_y, male_xy[1] - self.actuator_y0, None)
        # Convert polar to cartesian w.r.t. male position
        theta = np.radians(polar_coords[:, 0])
        xy = np.column_stack([np.sin(theta), np.cos(theta)]) * dists_y[:, None]
        xy = male_xy - xy
        # Ensure in actuator range
        xy[:, 0] = np.clip(xy[:, 0], 0, self.x_max)
        xy[:, 1] = np.clip(xy[:, 1], 0, self.y_max)
        return xy

    def __call__(self, *, angles=None, distances=None, amplitude=2.5, angular_speed=20, linear_speed=10,
                 randomize=True, n_cycles=1, duration=0, n_repetitions=1, pause=5):
        """Create a stimulus.

        Parameters
        ----------
        angles: list
            Angles (degrees) where stimulus presented
        distances: list
            Distances (mm) where stimulus presented
        amplitude: float
            Amplitude of stimulus oscillation
        angular_speed: float
            Angular speed of stimulus movement (degrees / s)
        linear_speed: float
            Linear (in-out distance) speed of stimulus movement (mm / s)
        randomize: bool or str
            Which parts of stimulus order to randomize
        n_cycles: int
            Number of oscillations of stimulus at a given position
        duration: float
            Dwell time of stimulus at a given position (s)
        n_repetitions: int
            Number of times stimulus is presented at each position
        pause: float
            Pause time at start of stimulus (s)
        """
        assert (angles is not None) and (distances is not None), "angles and distances must be specified"
        angles = np.atleast_1d(angles)
        distances = np.atleast_1d(distances)
        n_angles = len(angles)
        n_distances = len(distances)

        # Randomization
        i, j = np.meshgrid(np.arange(n_angles), np.arange(n_distances), indexing="ij")
        if randomize in [True, "all"]:
            ij = np.column_stack([i.ravel(), j.ravel()])
            ij = np.random.permutation(ij)
        elif randomize == "angles":
            order = np.random.permutation(n_angles)
            i = i[order]
            j = j[order]
            ij = np.column_stack([i.ravel(), j.ravel()])
        elif randomize == "distances":
            for row in j:
                np.random.shuffle(row)
            ij = np.column_stack([i.ravel(), j.ravel()])
        else:
            ij = np.column_stack([i.ravel(), j.ravel()])

        # Repetitions
        ij = np.tile(ij, (n_repetitions, 1))
        # Create keypoint sequence
        angles_dists = np.column_stack([angles[ij[:, 0]], distances[ij[:, 1]]])
        n_keypoints = len(angles_dists)

        # print(angles_dists)

        # Create oscillation
        cycle_distance = 4 * amplitude
        cycle_time = cycle_distance / angular_speed
        dwell_time = max([cycle_time * n_cycles, duration])

        # Generate cycle
        n_samples_per_cycle = int(self.sample_rate * cycle_time)
        cycle = np.zeros(n_samples_per_cycle)
        up = n_samples_per_cycle // 2
        down = n_samples_per_cycle - up
        cycle[:up] = np.linspace(-1, 1, up + 1)[:-1]
        cycle[up:] = np.linspace(1, -1, down + 1)[:-1]
        cycle *= amplitude
        cycle = np.roll(cycle, -n_samples_per_cycle // 4)
        # print(np.abs(np.diff(cycle)).mean() * self.sample_rate)
        # plt.plot(cycle)
        # plt.show()

        n_samples_per_dwell = int(dwell_time * self.sample_rate)
        n_samples = n_samples_per_dwell * n_keypoints

        total_cycles, remainder = divmod(n_samples, n_samples_per_cycle)
        all_cycles = np.hstack([np.tile(cycle, total_cycles), cycle[:remainder]])

        oscillations = [all_cycles[i * n_samples_per_dwell: (i + 1) * n_samples_per_dwell] for i in range(n_keypoints)]

        parts = []
        for angle_dist, oscillation in zip(angles_dists, oscillations):
            theta = angle_dist[0] + oscillation
            dist = np.ones(len(oscillation)) * angle_dist[1]
            parts.append(np.column_stack([theta, dist, np.zeros(len(theta))]))  # label = 0 for position

        # Add pause to beginning
        pause_frames = np.zeros((int(pause * self.sample_rate), 4))
        pause_frames[:, -2:] = -1  # label = -1 for pause

        sequence = [pause_frames]
        parts = iter(parts)
        k = -1
        while True:
            try:
                part = next(parts)
            except StopIteration:
                break
            theta0, d0 = sequence[-1][-1, :2]
            theta1, d1 = part[-1, :2]
            ddist = np.abs(d1 - d0)
            dtheta = np.abs(theta1 - theta0)
            t_dist = ddist / linear_speed
            t_theta = dtheta / angular_speed

            # label = 1 for in-out
            transition1 = np.linspace(d0, d1, int(t_dist * self.sample_rate))
            transition1 = np.column_stack([
                np.ones(len(transition1)) * theta0,
                transition1,
                np.ones(len(transition1)),
                np.ones(len(transition1)) * k,
            ])
            # label = 2 for movement to next position
            transition2 = np.linspace(theta0, theta1, int(t_theta * self.sample_rate))
            transition2 = np.column_stack([
                transition2,
                np.ones(len(transition2)) * d1,
                np.ones(len(transition2)) * 2,
                np.ones(len(transition2)) * k,
            ])
            # Concatenate transitions
            transition = np.vstack([transition1, transition2])

            sequence.append(transition)
            k += 1  # increment sequence index when new stimulus starts
            if (k % (len(angles) * len(distances))) == 0:
                k = 0
            sequence.append(np.column_stack([part, np.ones(len(part)) * k]))

        # Add pause to end
        pause_frames = np.zeros((int(pause * self.sample_rate), 4))
        pause_frames[:, 1] = self.y_max
        pause_frames[:, -2:] = -1  # label = -1 for pause
        sequence.append(pause_frames)

        stimulus = np.concatenate(sequence, axis=0)
        actuator_trace = self.polar_to_actuator(stimulus)
        stimulus = np.hstack([actuator_trace, stimulus])

        return stimulus


if __name__ == "__main__":
    import yaml
    from matplotlib import pyplot as plt
    with open(r"C:\Users\murthylab\Desktop\Duncan_data\configs\duncan_experiment\actuator_config.yaml") as f:
        config = yaml.safe_load(f)
    stim = ActuatorStimulus(config)

    xydt = stim(angles=[-15, 0, 15], distances=[1, 2, 3, 4, 5], amplitude=45, angular_speed=30, linear_speed=10, randomize=True, n_cycles=3, duration=0, n_repetitions=3, pause=5)

    male_xyz = np.array([stim.actuator_x0, stim.actuator_y0 + stim.dy_mm, stim.dz_mm])
    xyz = np.column_stack([xydt[:, :2], np.zeros(len(xydt))])

    d = np.linalg.norm(xyz - male_xyz, axis=1)
    plt.plot(xydt[:, 3])
    plt.plot(xydt[:, 4])
    plt.plot(xydt[:, 5])
    # plt.plot(d)
    # plt.plot(xydt[:, 0])
    plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(*xydt[:, :2].T)
    # ax.scatter([male_xyz[0]], [male_xyz[1]])
    # ax.set_aspect("equal")
    # plt.show()
