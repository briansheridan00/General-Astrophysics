# Import relevant packages for the code. 
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


# Define a class for a solar system body. 
class Body:
    """Represents a celestial body."""

    def __init__(self, mass: float, position: np.ndarray, velocity: np.ndarray, name: str = "Unnamed", is_fictional: bool = False):
        """
        Initializes the Body.

        Args:
            mass (float): The mass of the body.
            position (np.ndarray): The initial 2D position of the body [x, y].
            velocity (np.ndarray): The initial 2D velocity of the body [vx, vy].
            name (str, optional): The name of the body. Defaults to "Unnamed".
            is_fictional (bool, optional):  Indicates if the body is fictional. Defaults to False.
        """
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.name = name
        self.trajectory = [self.position.copy()]
        self.is_fictional = is_fictional # Store if the body is fictional

    def __repr__(self):
        return f"Body(name='{self.name}', mass={self.mass:.2e}, position={self.position}, velocity={self.velocity})"

    def update_position(self, dt: float):
        """Updates the position using the current velocity and time step."""
        self.position += self.velocity * dt
        self.trajectory.append(self.position.copy())

    def update_velocity(self, acceleration: np.ndarray, dt: float):
        """Updates the velocity using the acceleration and time step."""
        self.velocity += acceleration * dt


# Define a class for the solar system including integration of motion. 
class SolarSystem:
    """Represents a system of celestial bodies."""

    GRAVITATIONAL_CONSTANT = 6.67430e-11  # N m^2/kg^2

    def __init__(self, bodies: List[Body] = None):
        """
        Initializes the SolarSystem.

        Args:
            bodies (List[Body], optional): A list of Body objects. Defaults to an empty list.
        """
        self.bodies = bodies if bodies is not None else []

    def add_body(self, body: Body):
        """Adds a body to the solar system."""
        self.bodies.append(body)

    def calculate_forces(self) -> Dict[Body, np.ndarray]:
        """
        Calculates the gravitational forces between all bodies.

        Returns:
            Dict[Body, np.ndarray]: A dictionary where keys are bodies and values are the net force on each body.
        """
        forces = {body: np.array([0.0, 0.0], dtype=float) for body in self.bodies}
        for i, body1 in enumerate(self.bodies):
            for j in range(i + 1, len(self.bodies)):
                body2 = self.bodies[j]
                r_vector = body2.position - body1.position
                distance = np.linalg.norm(r_vector)
                if distance == 0:
                    continue  # Avoid division by zero for coincident bodies

                force_magnitude = (self.GRAVITATIONAL_CONSTANT * body1.mass * body2.mass) / (distance ** 2)
                force_direction = r_vector / distance  # Unit vector
                force_on_1 = force_magnitude * force_direction
                force_on_2 = -force_on_1  # Newton's third law

                forces[body1] += force_on_1
                forces[body2] += force_on_2
        return forces

    def calculate_accelerations(self, forces: Dict[Body, np.ndarray]) -> Dict[Body, np.ndarray]:
        """
        Calculates the acceleration of each body based on the forces.

        Args:
            forces (Dict[Body, np.ndarray]): A dictionary of forces on each body.

        Returns:
            Dict[Body, np.ndarray]: A dictionary of accelerations for each body.
        """
        accelerations = {}
        for body in self.bodies:
            accelerations[body] = forces[body] / body.mass
        return accelerations

    def update_system_rk4(self, dt: float):
        """
        Updates the positions and velocities of all bodies using the 4th order Runge-Kutta method.

        Args:
            dt (float): The time step.
        """

        # 1. Calculate initial values
        forces1 = self.calculate_forces()
        accelerations1 = self.calculate_accelerations(forces1)
        for body in self.bodies:
            body.position0 = body.position.copy()  # Store initial position
            body.velocity0 = body.velocity.copy()  # Store initial velocity
        # 2. Calculate k1 values
        k1_v = {body: accelerations1[body] * dt for body in self.bodies}
        k1_r = {body: body.velocity * dt for body in self.bodies}

        # 3. Calculate values at t + dt/2
        for body in self.bodies:
            body.position = body.position0 + 0.5 * k1_r[body]
            body.velocity = body.velocity0 + 0.5 * k1_v[body]

        # 4. Calculate k2 values
        forces2 = self.calculate_forces()
        accelerations2 = self.calculate_accelerations(forces2)
        k2_v = {body: accelerations2[body] * dt for body in self.bodies}
        k2_r = {body: (body.velocity0 + 0.5 * k1_v[body]) * dt for body in self.bodies}

        # 5. Calculate values at t + dt/2
        for body in self.bodies:
            body.position = body.position0 + 0.5 * k2_r[body]
            body.velocity = body.velocity0 + 0.5 * k2_v[body]

        # 6. Calculate k3 values
        forces3 = self.calculate_forces()
        accelerations3 = self.calculate_accelerations(forces3)
        k3_v = {body: accelerations3[body] * dt for body in self.bodies}
        k3_r = {body: (body.velocity0 + 0.5 * k2_v[body]) * dt for body in self.bodies}

        # 7. Calculate values at t + dt
        for body in self.bodies:
            body.position = body.position0 + k3_r[body]
            body.velocity = body.velocity0 + k3_v[body]

        # 8. Calculate k4 values
        forces4 = self.calculate_forces()
        accelerations4 = self.calculate_accelerations(forces4)
        k4_v = {body: accelerations4[body] * dt for body in self.bodies}
        k4_r = {body: (body.velocity0 + k3_v[body]) * dt for body in self.bodies}

        # 9. Update position and velocity using weighted average of k values
        for body in self.bodies:
            body.position = body.position0 + (k1_r[body] + 2*k2_r[body] + 2*k3_r[body] + k4_r[body]) / 6
            body.velocity = body.velocity0 + (k1_v[body] + 2*k2_v[body] + 2*k3_v[body] + k4_v[body]) / 6
            body.trajectory.append(body.position.copy())

    def get_body_positions(self) -> List[np.ndarray]:
        """Returns a list of the current positions of all bodies."""
        return [body.position for body in self.bodies]

    def get_body_trajectories(self) -> List[List[np.ndarray]]:
        """Returns a list of the trajectories of all bodies."""
        return [body.trajectory for body in self.bodies]


# Create a funciton to run the simulaiton and plot the result. 
def run_simulation(bodies: List[Body], years : float = 0.5):
    """
    Runs a small simulation of the solar system.
    """

    # Create the solar system
    solar_system = SolarSystem(bodies=bodies)

    # Simulation parameters
    dt = 3600 * 24  # 1 day in seconds 
    time_steps = int(years * 365 * 24 * 3600 / dt) # Number of steps 

    # Run the simulation
    for _ in range(time_steps):
        solar_system.update_system_rk4(dt)

    # Get the trajectories of the bodies
    trajectories = solar_system.get_body_trajectories()

    # Plot the trajectories
    plt.figure(figsize=(8, 8))
    plt.title("2D Solar System Simulation")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True)

    # Define some constants for plotting
    SOLAR_SYSTEM_PLANET_COLORS = {'Sun': 'yellow', 
                                'Mercury': 'gray', 
                                'Venus': 'gold', 
                                'Earth': 'blue', 
                                'Mars': 'red', 
                                'Jupiter': 'brown',
                                'Saturn': 'yellow',
                                'Uranus': 'cyan', 
                                'Neptune': 'blue'}  # True colors of real planets 
    MIN_MASS = min(body.mass for body in solar_system.bodies)
    MAX_MASS = max(body.mass for body in solar_system.bodies)
    MASS_SCALE_FACTOR = 1000  # Can adjust this to improve visualisation 

    # Plot the trajectories for each body
    for i, trajectory in enumerate(trajectories):
        body = solar_system.bodies[i]
        x_coords = [pos[0] for pos in trajectory]
        y_coords = [pos[1] for pos in trajectory]

        # Determine color based on whether the body is fictional
        if body.is_fictional:
            color = np.random.rand(3)  # Random color for fictional planets (rgb) 
        else:
            color = SOLAR_SYSTEM_PLANET_COLORS.get(body.name, 'gray')  # Default to gray if not found

        # Calculate scatter point size and scale with mass 
        size = (body.mass - MIN_MASS) / (MAX_MASS - MIN_MASS) * MASS_SCALE_FACTOR + 10  # Size is at least 10

        # Alpha values for each point in the trajectory, more recent = more opaque
        num_points = len(trajectory)
        alpha_values = np.linspace(0.05, 0.7, num_points) 

        #plt.plot(x_coords, y_coords, label=body.name, color=color, alpha=0.7) # added alpha to the line
        plt.scatter(x_coords, y_coords, c=[color] * num_points, s=[size] * num_points, alpha=alpha_values)  # Size varies, alpha varies
        plt.scatter(x_coords[-1], y_coords[-1], marker='o', s=50, color=color, label=f'{body.name}')  # Mark the starting position

    plt.legend()
    plt.axis('equal')  # Ensure equal aspect ratio for correct orbit shape
    plt.show()



