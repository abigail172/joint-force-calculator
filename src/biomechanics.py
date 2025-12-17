"""
Biomechanics Toolkit - Calculate forces on joints during movement

This is a collection of functions I built to understand how forces act on 
human joints. Pretty useful for understanding gaits, movememnt, 
or just figuring out why your knee starts to hurt when you run!

"""

import numpy as np
from typing import Optional


class Segment:
    """
    Represents a body segment (like your thigh or forearm).
    
    This keeps track of the physical properties we need for calculations:
    mass, length, where the center of mass is, etc.
    """
    
    def __init__(self, name, mass, length, com_ratio=0.5, rg_ratio=0.3):
        self.name = name
        self.mass = mass
        self.length = length
        self.com_ratio = com_ratio
        self.rg_ratio = rg_ratio
    
    @property
    def moment_of_inertia(self):
        """How hard it is to rotate this segment (kg⋅m²)"""
        radius_of_gyration = self.length * self.rg_ratio
        return self.mass * radius_of_gyration ** 2
    
    def com_position(self, proximal_pos, distal_pos):
        """
        Figure out where the center of mass is between two joint positions.
        """
        return proximal_pos + self.com_ratio * (distal_pos - proximal_pos)
    
    def __repr__(self):
        return f"{self.name}: {self.mass:.2f} kg, {self.length:.3f} m"


# Winter, 2009 - Biomechanics and Motor Control of Human Movement
BODY_SEGMENT_DATA = {
    'head': {'mass_ratio': 0.081, 'com_ratio': 0.5, 'rg_ratio': 0.303},
    'trunk': {'mass_ratio': 0.497, 'com_ratio': 0.5, 'rg_ratio': 0.496},
    'upper_arm': {'mass_ratio': 0.028, 'com_ratio': 0.436, 'rg_ratio': 0.322},
    'forearm': {'mass_ratio': 0.016, 'com_ratio': 0.43, 'rg_ratio': 0.303},
    'hand': {'mass_ratio': 0.006, 'com_ratio': 0.506, 'rg_ratio': 0.297},
    'thigh': {'mass_ratio': 0.100, 'com_ratio': 0.433, 'rg_ratio': 0.323},
    'shank': {'mass_ratio': 0.0465, 'com_ratio': 0.433, 'rg_ratio': 0.302},
    'foot': {'mass_ratio': 0.0145, 'com_ratio': 0.5, 'rg_ratio': 0.475},
}


def create_segment(segment_name, body_mass, length):
    """
    Quick way to create a segment using typical body proportions.
    
    For example, a thigh is typically 10% of body mass, so if you
    weigh 70kg, your thigh is about 7kg.
    """
    if segment_name not in BODY_SEGMENT_DATA:
        available = ', '.join(BODY_SEGMENT_DATA.keys())
        raise ValueError(f"Don't know about '{segment_name}'. Try one of: {available}")
    
    data = BODY_SEGMENT_DATA[segment_name]
    segment_mass = body_mass * data['mass_ratio']
    
    return Segment(
        name=segment_name,
        mass=segment_mass,
        length=length,
        com_ratio=data['com_ratio'],
        rg_ratio=data['rg_ratio']
    )
def calculate_joint_force(segment, prox_pos, dist_pos, dist_force, 
                         acceleration=None, gravity=9.81):
    """
    Calculate the reaction force at a joint using inverse dynamics.
    
    This is basically Newton's 2nd law (F = ma) applied to body segments.
    We work backwards from known forces (like ground reaction force) to
    figure out what's happening at the joints.
    
    segment: the Segment object we're analyzing
    prox_pos: position of proximal joint [x, y] (closer to body center)
    dist_pos: position of distal joint [x, y] (farther from body center)  
    dist_force: force acting at the distal end [Fx, Fy] in Newtons
    acceleration: how fast the segment's center of mass is accelerating [ax, ay]
                  (if None, assumes static/slow movement)
    gravity: acceleration due to gravity (9.81 m/s² on Earth)
    
    Returns: force at the proximal joint [Fx, Fy] in Newtons
    
    Example:
        If you know the ground pushes up on your foot with 500N, you can
        calculate how much force your knee experiences.
    """
    dims = len(prox_pos)
    if acceleration is None:
        acceleration = np.zeros(dims)
    
    # Weight force (always pulls downward)
    weight = np.zeros(dims)
    weight[-1] = -segment.mass * gravity  # negative because it's downward
    
    # Inertial force from acceleration (F = ma)
    inertial = segment.mass * acceleration
    
    # Balance the forces (what goes in must equal what goes out)
    proximal_force = dist_force + weight - inertial
    
    return proximal_force


def calculate_moment(force, force_pos, joint_pos, segment=None, 
                    angular_accel=0.0):
    """
    Calculate torque (moment) about a joint.
    
    This tells you how much rotational force is being applied.
    Think of it like using a wrench - the force times the distance
    from the bolt gives you the turning power.
    
    force: the force vector [Fx, Fy] in Newtons
    force_pos: where the force is applied [x, y]
    joint_pos: center of the joint [x, y]
    segment: if provided, includes rotational inertia effects
    angular_accel: angular acceleration in rad/s²
    
    Returns: moment in N⋅m (positive = counterclockwise)
    """
    # Distance from joint to where force acts
    r = force_pos - joint_pos
    
    # Cross product in 2D: this is the "perpendicular force times distance" calculation
    moment = r[0] * force[1] - r[1] * force[0]
    
    # If the segment is rotating, add the inertial component
    if segment and angular_accel != 0.0:
        moment += segment.moment_of_inertia * angular_accel
    
    return moment


def calculate_muscle_force(moment_needed, moment_arm, pennation_angle=0.0):
    """
    Figure out how much force a muscle needs to produce.
    
    Muscles create torque at joints. If you know how much torque you need
    and how far the muscle attaches from the joint, you can calculate the
    required muscle force.
    
    moment_needed: the torque required at the joint (N⋅m)
    moment_arm: distance from joint to muscle attachment (meters)
    pennation_angle: angle of muscle fibers relative to tendon (degrees)
                     (most muscles have fibers at an angle, not straight)
    
    Returns: required muscle force in Newtons
    
    Example:
        To hold a 10kg weight with your elbow at 90°, your biceps might need
        to produce 300N of force depending on the moment arm.
    """
    if moment_arm <= 0:
        raise ValueError("Moment arm must be positive - check your measurements!")
    
    # Account for pennation angle (only component along tendon counts)
    angle_rad = np.radians(pennation_angle)
    force = moment_needed / (moment_arm * np.cos(angle_rad))
    
    return force


def calculate_angle(point1, vertex, point3):
    """
    Calculate the angle formed by three points.
    
    This is useful for measuring joint angles from motion capture data.
    
    point1: first point [x, y] (like hip position)
    vertex: middle point [x, y] (like knee - this is where angle is measured)
    point3: third point [x, y] (like ankle position)
    
    Returns: angle in degrees (0-180)
    
    Example:
        To measure knee angle, pass in hip, knee, and ankle positions.
        180° = fully straight, smaller angles = more bent.
    """
    # Create vectors from vertex to the other two points
    v1 = point1 - vertex
    v2 = point3 - vertex
    
    # Use dot product to find angle between vectors
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    # Handle floating point errors (cos must be between -1 and 1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def calculate_power(moment, angular_velocity):
    """
    Calculate mechanical power at a joint.
    
    Power = Moment × Angular Velocity
    
    moment: joint moment in N⋅m
    angular_velocity: how fast the joint is rotating in rad/s
    
    Returns: power in Watts
    
    Positive power = muscle is generating energy (concentric contraction)
    Negative power = muscle is absorbing energy (eccentric contraction)
    
    Example:
        When you jump, your leg joints generate power to accelerate you upward.
        When you land, they absorb power to slow you down.
    """
    return moment * angular_velocity


# Some quick helper functions

def body_weight_force(mass_kg):
    """Convert body mass to weight force in Newtons"""
    return mass_kg * 9.81


def newtons_to_body_weights(force_n, body_mass_kg):
    """Express a force as multiples of body weight (BW)"""
    return force_n / body_weight_force(body_mass_kg)


def estimate_segment_length(height_m, segment_name):
    """
    Rough estimate of segment length based on total height.
    These are typical proportions - real people vary!
    """
    length_ratios = {
        'thigh': 0.245,
        'shank': 0.246,
        'foot': 0.152,
        'upper_arm': 0.186,
        'forearm': 0.146,
        'trunk': 0.288,
    }
    
    if segment_name not in length_ratios:
        raise ValueError(f"Don't have length estimate for '{segment_name}'")
    
    return height_m * length_ratios[segment_name]
