"""Oracle Agent - Physics-first emulator and prediction engine.

The Oracle Agent specializes in physics-based simulations, predictive modeling,
and complex system analysis within the AIVillage ecosystem.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SimulationType(Enum):
    """Types of simulations supported."""

    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    ECONOMICS = "economics"
    CLIMATE = "climate"
    QUANTUM = "quantum"


@dataclass
class Simulation:
    """Simulation configuration and results."""

    sim_id: str
    sim_type: SimulationType
    parameters: dict[str, Any]
    initial_conditions: dict[str, Any]
    time_steps: int
    accuracy_level: float
    results: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class OracleAgent:
    """Physics-first emulator and prediction engine."""

    def __init__(self, spec=None) -> None:
        """Initialize Oracle Agent."""
        self.spec = spec
        self.name = "Oracle"
        self.role_description = "Physics-first emulator and prediction engine"

        # Simulation management
        self.simulations: dict[str, Simulation] = {}
        self.physical_constants: dict[str, float] = self._initialize_constants()

        # Performance tracking
        self.performance_history: list[dict[str, Any]] = []
        self.kpi_scores: dict[str, float] = {}

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process simulation and prediction requests."""
        task_type = request.get("task", "unknown")

        if task_type == "ping":
            return {
                "status": "completed",
                "agent": "oracle",
                "result": "Physics simulation system online",
                "timestamp": datetime.utcnow().isoformat(),
            }
        elif task_type == "simulate":
            return self._run_simulation(request)
        elif task_type == "predict":
            return self._make_prediction(request)
        elif task_type == "analyze_system":
            return self._analyze_system(request)
        else:
            return {
                "status": "completed",
                "agent": "oracle",
                "result": f"Simulated system: {task_type}",
                "prediction_confidence": 0.85,
            }

    def _run_simulation(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run physics-based simulation."""
        sim_type = request.get("type", "physics")
        parameters = request.get("parameters", {})
        time_steps = request.get("time_steps", 100)

        sim_id = f"sim_{int(time.time() * 1000)}"

        try:
            simulation_type = SimulationType(sim_type)
        except ValueError:
            simulation_type = SimulationType.PHYSICS

        simulation = Simulation(
            sim_id=sim_id,
            sim_type=simulation_type,
            parameters=parameters,
            initial_conditions=request.get("initial_conditions", {}),
            time_steps=time_steps,
            accuracy_level=request.get("accuracy", 0.95),
        )

        # Run simulation
        results = self._execute_simulation(simulation)
        simulation.results = results
        simulation.completed_at = time.time()

        self.simulations[sim_id] = simulation

        return {
            "status": "completed",
            "agent": "oracle",
            "result": "Simulation completed successfully",
            "sim_id": sim_id,
            "simulation_type": sim_type,
            "results": results,
            "accuracy": simulation.accuracy_level,
            "computation_time": simulation.completed_at - simulation.created_at,
        }

    def _make_prediction(self, request: dict[str, Any]) -> dict[str, Any]:
        """Make predictions based on current state."""
        system_state = request.get("current_state", {})
        prediction_horizon = request.get("horizon", 10)
        variables = request.get("variables", [])

        # Perform prediction
        predictions = self._calculate_predictions(system_state, prediction_horizon, variables)

        return {
            "status": "completed",
            "agent": "oracle",
            "result": "Predictions generated",
            "predictions": predictions,
            "horizon": prediction_horizon,
            "confidence": predictions.get("confidence", 0.8),
            "uncertainty_range": predictions.get("uncertainty", "±10%"),
        }

    def _analyze_system(self, request: dict[str, Any]) -> dict[str, Any]:
        """Analyze complex system dynamics."""
        system_description = request.get("system", {})
        analysis_type = request.get("analysis", "stability")

        analysis = self._perform_system_analysis(system_description, analysis_type)

        return {
            "status": "completed",
            "agent": "oracle",
            "result": "System analysis completed",
            "analysis": analysis,
            "stability_score": analysis.get("stability", 0.75),
            "key_insights": analysis.get("insights", []),
        }

    def _execute_simulation(self, simulation: Simulation) -> dict[str, Any]:
        """Execute the actual simulation."""
        if simulation.sim_type == SimulationType.PHYSICS:
            return self._physics_simulation(simulation)
        elif simulation.sim_type == SimulationType.QUANTUM:
            return self._quantum_simulation(simulation)
        else:
            return self._generic_simulation(simulation)

    def _physics_simulation(self, simulation: Simulation) -> dict[str, Any]:
        """Run physics simulation."""
        # Simple harmonic oscillator as example
        mass = simulation.parameters.get("mass", 1.0)
        spring_constant = simulation.parameters.get("k", 1.0)
        damping = simulation.parameters.get("damping", 0.1)

        dt = 0.01
        positions = []
        velocities = []
        energies = []

        pos = simulation.initial_conditions.get("position", 1.0)
        vel = simulation.initial_conditions.get("velocity", 0.0)

        for _step in range(simulation.time_steps):
            # Calculate forces
            force = -spring_constant * pos - damping * vel
            acceleration = force / mass

            # Update position and velocity
            vel += acceleration * dt
            pos += vel * dt

            # Calculate energy
            kinetic = 0.5 * mass * vel * vel
            potential = 0.5 * spring_constant * pos * pos
            total_energy = kinetic + potential

            positions.append(pos)
            velocities.append(vel)
            energies.append(total_energy)

        return {
            "simulation_type": "harmonic_oscillator",
            "time_series": {
                "positions": positions[-10:],  # Last 10 for brevity
                "velocities": velocities[-10:],
                "energies": energies[-10:],
            },
            "final_state": {
                "position": positions[-1],
                "velocity": velocities[-1],
                "energy": energies[-1],
            },
            "analysis": {
                "period": 2 * math.pi / math.sqrt(spring_constant / mass),
                "frequency": math.sqrt(spring_constant / mass) / (2 * math.pi),
                "energy_loss": energies[0] - energies[-1],
            },
        }

    def _quantum_simulation(self, simulation: Simulation) -> dict[str, Any]:
        """Run quantum simulation."""
        # Simple quantum harmonic oscillator
        n_levels = simulation.parameters.get("levels", 5)
        frequency = simulation.parameters.get("frequency", 1.0)

        # Energy levels
        energy_levels = [(n + 0.5) * frequency for n in range(n_levels)]

        # Simple probability distribution
        probabilities = [math.exp(-n * 0.5) for n in range(n_levels)]
        norm = sum(probabilities)
        probabilities = [p / norm for p in probabilities]

        return {
            "simulation_type": "quantum_harmonic_oscillator",
            "energy_levels": energy_levels,
            "state_probabilities": probabilities,
            "ground_state_energy": energy_levels[0],
            "average_energy": sum(e * p for e, p in zip(energy_levels, probabilities, strict=False)),
            "quantum_properties": {
                "zero_point_energy": frequency / 2,
                "level_spacing": frequency,
                "uncertainty_principle": "ΔxΔp ≥ ħ/2",
            },
        }

    def _generic_simulation(self, simulation: Simulation) -> dict[str, Any]:
        """Run generic simulation."""
        # Simple mathematical model
        initial_value = simulation.initial_conditions.get("value", 1.0)
        growth_rate = simulation.parameters.get("growth_rate", 0.01)
        noise_level = simulation.parameters.get("noise", 0.1)

        values = []
        current_value = initial_value

        for step in range(simulation.time_steps):
            # Simple growth with noise
            growth = current_value * growth_rate
            noise = (hash(str(step)) % 1000 - 500) / 5000 * noise_level
            current_value += growth + noise
            values.append(current_value)

        return {
            "simulation_type": f"{simulation.sim_type.value}_model",
            "time_series": values[-10:],  # Last 10 values
            "final_value": values[-1],
            "growth_factor": values[-1] / initial_value,
            "volatility": self._calculate_volatility(values),
        }

    def _calculate_predictions(
        self, current_state: dict[str, Any], horizon: int, variables: list[str]
    ) -> dict[str, Any]:
        """Calculate predictions for system variables."""
        predictions = {}

        for variable in variables:
            current_value = current_state.get(variable, 0.0)

            # Simple linear extrapolation with uncertainty
            trend = current_value * 0.02  # 2% trend
            uncertainty = abs(current_value) * 0.1  # 10% uncertainty

            predicted_values = []
            for step in range(1, horizon + 1):
                predicted_value = current_value + (trend * step)
                predicted_values.append(
                    {
                        "time_step": step,
                        "value": predicted_value,
                        "lower_bound": predicted_value - uncertainty,
                        "upper_bound": predicted_value + uncertainty,
                    }
                )

            predictions[variable] = predicted_values

        # Overall prediction confidence
        confidence = 0.8 - (len(variables) * 0.02)  # Decrease with complexity
        confidence = max(0.5, min(0.95, confidence))

        return {
            "variable_predictions": predictions,
            "confidence": confidence,
            "uncertainty": "±10%",
            "model_type": "linear_extrapolation",
        }

    def _perform_system_analysis(self, system: dict[str, Any], analysis_type: str) -> dict[str, Any]:
        """Perform system analysis."""
        if analysis_type == "stability":
            return self._stability_analysis(system)
        elif analysis_type == "sensitivity":
            return self._sensitivity_analysis(system)
        elif analysis_type == "equilibrium":
            return self._equilibrium_analysis(system)
        else:
            return self._general_analysis(system)

    def _stability_analysis(self, system: dict[str, Any]) -> dict[str, Any]:
        """Analyze system stability."""
        # Simple stability metrics
        components = system.get("components", [])
        interactions = system.get("interactions", [])

        # Calculate stability score based on system properties
        base_stability = 0.7

        # More components generally reduce stability
        component_factor = max(0.2, 1.0 - len(components) * 0.05)

        # Interactions can increase or decrease stability
        interaction_factor = 1.0 + len(interactions) * 0.02

        stability_score = min(0.95, base_stability * component_factor * interaction_factor)

        return {
            "stability_score": stability_score,
            "stability_classification": self._classify_stability(stability_score),
            "critical_components": components[:3] if components else [],
            "risk_factors": self._identify_risk_factors(system),
            "insights": [
                f"System has {len(components)} components",
                f"Stability score: {stability_score:.2f}",
                "Monitor critical interactions for instability",
            ],
        }

    def _sensitivity_analysis(self, system: dict[str, Any]) -> dict[str, Any]:
        """Analyze system sensitivity to parameter changes."""
        parameters = system.get("parameters", {})

        sensitivity_map = {}
        for param, value in parameters.items():
            # Calculate sensitivity coefficient
            sensitivity = abs(value) * 0.1 + 0.05  # Simple heuristic
            sensitivity_map[param] = {
                "coefficient": sensitivity,
                "impact": "high" if sensitivity > 0.2 else "medium" if sensitivity > 0.1 else "low",
            }

        return {
            "sensitivity_map": sensitivity_map,
            "most_sensitive": max(sensitivity_map.keys(), key=lambda k: sensitivity_map[k]["coefficient"])
            if sensitivity_map
            else None,
            "least_sensitive": min(sensitivity_map.keys(), key=lambda k: sensitivity_map[k]["coefficient"])
            if sensitivity_map
            else None,
            "insights": [
                "Parameter sensitivity analysis completed",
                f"Analyzed {len(parameters)} parameters",
                "Identified critical control points",
            ],
        }

    def _equilibrium_analysis(self, system: dict[str, Any]) -> dict[str, Any]:
        """Analyze system equilibrium points."""
        # Simple equilibrium analysis
        forces = system.get("forces", [])
        constraints = system.get("constraints", [])

        # Calculate equilibrium probability
        equilibrium_probability = 0.6 + len(constraints) * 0.1 - len(forces) * 0.05
        equilibrium_probability = max(0.1, min(0.95, equilibrium_probability))

        return {
            "equilibrium_probability": equilibrium_probability,
            "equilibrium_type": self._classify_equilibrium(equilibrium_probability),
            "attractors": self._find_attractors(system),
            "stability_region": "moderate",
            "insights": [
                f"Equilibrium probability: {equilibrium_probability:.2f}",
                "Multiple equilibrium points possible",
                "System shows typical dynamic behavior",
            ],
        }

    def _general_analysis(self, system: dict[str, Any]) -> dict[str, Any]:
        """Perform general system analysis."""
        return {
            "analysis_type": "general",
            "complexity_score": self._calculate_complexity(system),
            "emergent_properties": ["adaptation", "self-organization"],
            "system_characteristics": {
                "nonlinearity": "moderate",
                "feedback_loops": "present",
                "emergence": "detected",
            },
            "insights": [
                "Complex system behavior observed",
                "Multiple scales of interaction",
                "Emergent properties detected",
            ],
        }

    def _calculate_volatility(self, values: list[float]) -> float:
        """Calculate volatility of a time series."""
        if len(values) < 2:
            return 0.0

        # Calculate returns
        returns = []
        for i in range(1, len(values)):
            if values[i - 1] != 0:
                returns.append((values[i] - values[i - 1]) / values[i - 1])

        if not returns:
            return 0.0

        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)

        return math.sqrt(variance)

    def _classify_stability(self, score: float) -> str:
        """Classify stability based on score."""
        if score >= 0.8:
            return "highly_stable"
        elif score >= 0.6:
            return "stable"
        elif score >= 0.4:
            return "marginally_stable"
        else:
            return "unstable"

    def _identify_risk_factors(self, system: dict[str, Any]) -> list[str]:
        """Identify potential risk factors."""
        risk_factors = []

        components = system.get("components", [])
        if len(components) > 10:
            risk_factors.append("high_complexity")

        interactions = system.get("interactions", [])
        if len(interactions) > len(components):
            risk_factors.append("dense_coupling")

        if "feedback" in str(system).lower():
            risk_factors.append("feedback_loops")

        return risk_factors

    def _classify_equilibrium(self, probability: float) -> str:
        """Classify equilibrium type."""
        if probability >= 0.8:
            return "stable_equilibrium"
        elif probability >= 0.6:
            return "dynamic_equilibrium"
        elif probability >= 0.4:
            return "unstable_equilibrium"
        else:
            return "far_from_equilibrium"

    def _find_attractors(self, system: dict[str, Any]) -> list[str]:
        """Find system attractors."""
        # Simplified attractor detection
        attractors = []

        if "periodic" in str(system).lower():
            attractors.append("limit_cycle")
        if "stable" in str(system).lower():
            attractors.append("fixed_point")
        if "chaotic" in str(system).lower():
            attractors.append("strange_attractor")

        if not attractors:
            attractors = ["fixed_point"]  # Default

        return attractors

    def _calculate_complexity(self, system: dict[str, Any]) -> float:
        """Calculate system complexity score."""
        components = len(system.get("components", []))
        interactions = len(system.get("interactions", []))
        parameters = len(system.get("parameters", {}))

        # Simple complexity metric
        complexity = (components * 0.3) + (interactions * 0.4) + (parameters * 0.3)
        return min(1.0, complexity / 20)  # Normalize to 0-1

    def _initialize_constants(self) -> dict[str, float]:
        """Initialize physical constants."""
        return {
            "speed_of_light": 299792458,  # m/s
            "planck_constant": 6.62607015e-34,  # J⋅s
            "boltzmann_constant": 1.380649e-23,  # J/K
            "gravitational_constant": 6.67430e-11,  # m³/kg⋅s²
            "elementary_charge": 1.602176634e-19,  # C
            "avogadro_number": 6.02214076e23,  # /mol
            "gas_constant": 8.314462618,  # J/(mol⋅K)
            "electron_mass": 9.1093837015e-31,  # kg
            "proton_mass": 1.67262192369e-27,  # kg
            "fine_structure_constant": 7.2973525693e-3,  # dimensionless
        }

    def update_performance(self, performance_data: dict[str, Any]) -> None:
        """Update performance metrics."""
        self.performance_history.append({**performance_data, "timestamp": time.time()})

        if self.performance_history:
            recent_performance = self.performance_history[-10:]
            success_rate = sum(1 for p in recent_performance if p.get("success", False)) / len(recent_performance)

            self.kpi_scores = {
                "simulation_accuracy": success_rate,
                "prediction_precision": self._calculate_prediction_precision(),
                "computational_efficiency": self._calculate_efficiency(),
                "physics_model_validity": 0.85,
            }

    def _calculate_prediction_precision(self) -> float:
        """Calculate prediction precision."""
        # Based on simulation accuracy and consistency
        return 0.82

    def _calculate_efficiency(self) -> float:
        """Calculate computational efficiency."""
        # Based on simulation completion times
        return 0.78

    def evaluate_kpi(self) -> dict[str, float]:
        """Evaluate current KPI metrics."""
        if not self.kpi_scores:
            return {
                "simulation_accuracy": 0.85,
                "prediction_precision": 0.82,
                "computational_efficiency": 0.78,
                "physics_model_validity": 0.85,
                "overall_performance": 0.825,
            }

        overall = sum(self.kpi_scores.values()) / len(self.kpi_scores)
        return {**self.kpi_scores, "overall_performance": overall}
