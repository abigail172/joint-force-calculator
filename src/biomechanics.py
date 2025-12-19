"""
This is a collection of functions I built to understand how forces act on 
human joints. Pretty useful for understanding gaits, movememnt, 
or just figuring out why your knee starts to hurt when you run!

I'm starting with static analysis but I plan to add dynamics later too

NOTE: This assumes static equilibrium and ignores muscle co-contraction,
so real life forces are probably higher than what is calculated in these 
functions.

"""

"""
Biomechanics calculations for joint forces

I built this to better understand the forces acting on joints during movement.
Started simple with static analysis, but planning to add dynamics later.

Note: This assumes static equilibrium and ignores muscle co-contraction,
so real forces are probably higher than what these functions calculate.

- Abigail Wu
"""

import numpy as np
from typing import Optional


class Segment:
    """A body part with mass and length"""
    
    def __init__(self, name, mass, length, com_ratio=0.5, rg_ratio=0.3):
        self.name = name
        self.mass = mass
        self.length = length
        self.com_ratio = com_ratio
        self.rg_ratio = rg_ratio
    
    @property
    def moment_of_inertia(self):
        radius_of_gyration = self.length * self.rg_ratio
        return self.mass * radius_of_gyration ** 2
    
    def com_position(self, prox_pos, dist_pos):
        return prox_pos + self.com_ratio * (dist_pos - prox_pos)
    
    def __repr__(self):
        return f"{self.name}: {self.mass:.2f}kg, {self.length:.3f}m"


# Winter, 2009 - Biomechanics and Motor Control of Human Movement
BODY_PARTS = {
    'head': {'mass_ratio': 0.081, 'com_ratio': 0.5, 'rg_ratio': 0.303},
    'trunk': {'mass_ratio': 0.497, 'com_ratio': 0.5, 'rg_ratio': 0.496},
    'upper_arm': {'mass_ratio': 0.028, 'com_ratio': 0.436, 'rg_ratio': 0.322},
    'forearm': {'mass_ratio': 0.016, 'com_ratio': 0.43, 'rg_ratio': 0.303},
    'hand': {'mass_ratio': 0.006, 'com_ratio': 0.506, 'rg_ratio': 0.297},
    'thigh': {'mass_ratio': 0.100, 'com_ratio': 0.433, 'rg_ratio': 0.323},
    'shank': {'mass_ratio': 0.0465, 'com_ratio': 0.433, 'rg_ratio': 0.302},
    'foot': {'mass_ratio': 0.0145, 'com_ratio': 0.5, 'rg_ratio': 0.475},
}


def create_segment(part_name, body_mass, length):
    """
    Make a body segment with realistic proportions.
    Your thigh is roughly 10% of your body weight, for example.
    
    These are average values from Winter (2009), but obviously 
    everyone's proportions are different.
    """
    if part_name not in BODY_PARTS:
        options = ', '.join(BODY_PARTS.keys())
        raise ValueError(f"Haven't heard of '{part_name}'. Try: {options}")
    
    data = BODY_PARTS[part_name]
    part_mass = body_mass * data['mass_ratio']
    
    return Segment(part_name, part_mass, length, 
                  data['com_ratio'], data['rg_ratio'])


def calculate_joint_force(segment, prox, dist, force_at_dist, accel=None):
    """
    Work backwards to find joint forces. If you know the force at your 
    ankle, you can figure out what your knee is dealing with.
    
    Limitations: Assumes single rigid segment, ignores ligament forces,
    and treats joints as simple hinges. Real joints are way more complex.
    Also assumes static equilibrium if no acceleration is given.
    """
    if accel is None:
        accel = np.zeros(len(prox))
    
    weight = np.zeros(len(prox))
    weight[-1] = -segment.mass * 9.81
    
    return force_at_dist + weight - segment.mass * accel


def calculate_moment(force, where_force_is, where_joint_is):
    """
    How much turning force (torque) does this create?
    Like when you use a wrench - force times distance from the bolt.
    """
    distance = where_force_is - where_joint_is
    return distance[0] * force[1] - distance[1] * force[0]


def calculate_muscle_force(torque_wanted, distance_to_muscle):
    """
    Figure out how hard a muscle has to pull.
    Small moment arm = you need more force.
    
    Note: This ignores pennation angle and assumes 100% efficiency,
    so actual muscle forces are probably 10-20% higher.
    """
    if distance_to_muscle <= 0:
        raise ValueError("Distance needs to be positive!")
    return torque_wanted / distance_to_muscle


def calculate_angle(point1, middle, point3):
    """Measure the angle at the middle point using dot product"""
    v1 = point1 - middle
    v2 = point3 - middle
    
    dot = np.dot(v1, v2)
    lengths = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = np.clip(dot / lengths, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_angle))


def calculate_power(torque, angular_speed):
    """
    How much power is being generated or absorbed?
    Positive = generating (like jumping)
    Negative = absorbing (like landing)
    """
    return torque * angular_speed


def body_weight_force(kg):
    """Turn mass into force"""
    return kg * 9.81


def newtons_to_body_weights(force, mass):
    """Express force as multiples of body weight"""
    return force / body_weight_force(mass)


def estimate_segment_length(height, part):
    """
    Guess segment length from total height.
    These are averages from anthropometric studies - everyone's different!
    """
    ratios = {
        'thigh': 0.245,
        'shank': 0.246, 
        'foot': 0.152,
        'upper_arm': 0.186,
        'forearm': 0.146,
        'trunk': 0.288,
    }
    
    if part not in ratios:
        raise ValueError(f"Don't have data for '{part}'")
    
    return height * ratios[part]
