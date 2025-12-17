"""
Core biomechanics calculations for joint force analysis.

This module provides functions and classes for calculating forces,
moments, and other biomechanical quantities during human movement.
"""

import numpy as np
from typing import Tuple, List, Optional
import warnings


class Segment:
    """
    Represents a body segment with mass and geometric properties.
    
    Attributes
    ----------
    name : str
        Name of the segment (e.g., 'thigh', 'shank')
    mass : float
        Mass in kilograms
    length : float
        Length in meters
    com_ratio : float
        Ratio of COM position from proximal end (0-1)
    radius_of_gyration_ratio : float
        Radius of gyration as ratio of segment length
    """
    
    def __init__(self, name: str, mass: float, length: float, 
                 com_ratio: float = 0.5, rg_ratio: float = 0.3):
        self.name = name
        self.mass = mass
        self.length = length
        self.com_ratio = com_ratio
        self.rg_ratio = rg_ratio
        
    @property
    def moment_of_inertia(self) -> float:
        """Calculate moment of inertia about COM (kg⋅m²)"""
        radius_of_gyration = self.length * self.rg_ratio
        return self.mass * radius_of_gyration ** 2
    
    def com_position(self, proximal_pos: np.ndarray, 
                    distal_pos: np.ndarray) -> np.ndarray:
        """
        Calculate center of mass position.
        
        Parameters
        ----------
        proximal_pos : np.ndarray
            Position of proximal joint [x, y, z]
        distal_pos : np.ndarray
            Position of distal joint [x, y, z]
        
        Returns
        -------
        np.ndarray
            COM position [x, y, z]
        """
        return proximal_pos + self.com_ratio * (distal_pos - proximal_pos)
    
    def __repr__(self):
        return (f"Segment(name='{self.name}', mass={self.mass:.2f} kg, "
                f"length={self.length:.3f} m)")


# Anthropometric data (Winter, 2009)
SEGMENT_PARAMETERS = {
    'head': {'mass_ratio': 0.081, 'com_ratio': 0.5, 'rg_ratio': 0.303},
    'trunk': {'mass_ratio': 0.497, 'com_ratio': 0.5, 'rg_ratio': 0.496},
    'upper_arm': {'mass_ratio': 0.028, 'com_ratio': 0.436, 'rg_ratio': 0.322},
    'forearm': {'mass_ratio': 0.016, 'com_ratio': 0.43, 'rg_ratio': 0.303},
    'hand': {'mass_ratio': 0.006, 'com_ratio': 0.506, 'rg_ratio': 0.297},
    'thigh': {'mass_ratio': 0.100, 'com_ratio': 0.433, 'rg_ratio': 0.323},
    'shank': {'mass_ratio': 0.0465, 'com_ratio': 0.433, 'rg_ratio': 0.302},
    'foot': {'mass_ratio': 0.0145, 'com_ratio': 0.5, 'rg_ratio': 0.475},
}


def create_segment_from_body_mass(segment_name: str, body_mass: float, 
                                 length: float) -> Segment:
    """
    Create a segment using anthropometric ratios.
    
    Parameters
    ----------
    segment_name : str
        Name of segment (must be in SEGMENT_PARAMETERS)
    body_mass : float
        Total body mass in kg
    length : float
        Segment length in meters
    
    Returns
    -------
    Segment
        Configured segment object
    """
    if segment_name not in SEGMENT_PARAMETERS:
        raise ValueError(f"Unknown segment: {segment_name}")
    
    params = SEGMENT_PARAMETERS[segment_name]
    mass = body_mass * params['mass_ratio']
    
    return Segment(
        name=segment_name,
        mass=mass,
        length=length,
        com_ratio=params['com_ratio'],
        rg_ratio=params['rg_ratio']
    )


def calculate_joint_reaction_force(
    segment: Segment,
    proximal_pos: np.ndarray,
    distal_pos: np.ndarray,
    distal_force: np.ndarray,
    acceleration: Optional[np.ndarray] = None,
    gravity: float = 9.81
) -> np.ndarray:
    """
    Calculate joint reaction force using inverse dynamics.
    
    This implements the equation:
    F_prox = F_dist + m*a_COM - m*g
    
    Parameters
    ----------
    segment : Segment
        Body segment object
    proximal_pos : np.ndarray
        Proximal joint position [x, y] or [x, y, z]
    distal_pos : np.ndarray
        Distal joint position [x, y] or [x, y, z]
    distal_force : np.ndarray
        Force at distal end [Fx, Fy] or [Fx, Fy, Fz] in Newtons
    acceleration : np.ndarray, optional
        COM acceleration [ax, ay] or [ax, ay, az] in m/s²
        If None, assumes static equilibrium
    gravity : float, optional
        Gravitational acceleration (default: 9.81 m/s²)
    
    Returns
    -------
    np.ndarray
        Proximal joint reaction force in Newtons
    """
    # Handle dimensions
    dims = len(proximal_pos)
    if acceleration is None:
        acceleration = np.zeros(dims)
    
    # Gravitational force (always acts downward)
    gravity_force = np.zeros(dims)
    gravity_force[-1] = -segment.mass * gravity
    
    # Inertial force (ma)
    inertial_force = segment.mass * acceleration
    
    # Sum of forces (Newton's 2nd law)
    proximal_force = distal_force + gravity_force - inertial_force
    
    return proximal_force


def calculate_joint_moment(
    force: np.ndarray,
    force_position: np.ndarray,
    joint_position: np.ndarray,
    segment: Optional[Segment] = None,
    angular_acceleration: float = 0.0
) -> float:
    """
    Calculate moment (torque) about a joint.
    
    M = r × F + I*α
    
    Parameters
    ----------
    force : np.ndarray
        Force vector [Fx, Fy] in Newtons
    force_position : np.ndarray
        Position where force acts [x, y]
    joint_position : np.ndarray
        Joint center position [x, y]
    segment : Segment, optional
        If provided, includes inertial moment (I*α)
    angular_acceleration : float, optional
        Angular acceleration in rad/s² (default: 0)
    
    Returns
    -------
    float
        Joint moment in N⋅m (positive = counterclockwise)
    """
    # Moment arm
    r = force_position - joint_position
    
    # 2D cross product: r × F = rx*Fy - ry*Fx
    moment = r[0] * force[1] - r[1] * force[0]
    
    # Add inertial moment if segment provided
    if segment is not None and angular_acceleration != 0.0:
        inertial_moment = segment.moment_of_inertia * angular_acceleration
        moment += inertial_moment
    
    return moment


def calculate_muscle_force(
    required_moment: float,
    moment_arm: float,
    pennation_angle: float = 0.0,
    efficiency: float = 1.0
) -> float:
    """
    Calculate muscle force needed to produce a given joint moment.
    
    F_muscle = M_joint / (d * cos(θ) * η)
    
    Parameters
    ----------
    required_moment : float
        Required joint moment in N⋅m
    moment_arm : float
        Muscle moment arm in meters
    pennation_angle : float, optional
        Muscle pennation angle in degrees (default: 0)
    efficiency : float, optional
        Mechanical efficiency 0-1 (default: 1.0)
    
    Returns
    -------
    float
        Required muscle force in Newtons
    """
    if moment_arm <= 0:
        raise ValueError("Moment arm must be positive")
    if not 0 < efficiency <= 1:
        raise ValueError("Efficiency must be between 0 and 1")
    
    theta_rad = np.radians(pennation_angle)
    force = required_moment / (moment_arm * np.cos(theta_rad) * efficiency)
    
    return force


def calculate_angle_2d(point1: np.ndarray, vertex: np.ndarray, 
                      point3: np.ndarray) -> float:
    """
    Calculate angle formed by three points in 2D.
    
    Parameters
    ----------
    point1 : np.ndarray
        First point [x, y]
    vertex : np.ndarray
        Vertex point (where angle is measured) [x, y]
    point3 : np.ndarray
        Third point [x, y]
    
    Returns
    -------
    float
        Angle in degrees (0-180)
    """
    v1 = point1 - vertex
    v2 = point3 - vertex
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def joint_power(moment: float, angular_velocity: float) -> float:
    """
    Calculate mechanical power at a joint.
    
    P = M * ω
    
    Parameters
    ----------
    moment : float
        Joint moment in N⋅m
    angular_velocity : float
        Angular velocity in rad/s
    
    Returns
    -------
    float
        Mechanical power in Watts
        Positive = concentric (generating), Negative = eccentric (absorbing)
    """
    return moment * angular_velocity

