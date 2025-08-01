# Smart Routing for Electric Vehicles Using Graph Models and AI

A comprehensive operational research project implementing intelligent routing algorithms for electric vehicles navigating real-world urban networks with battery constraints and charging infrastructure.

## 🚗 Project Overview

This project addresses the unique challenges of electric vehicle (EV) route planning by developing and comparing multiple algorithmic approaches that consider:
- Battery capacity limitations
- Energy consumption modeling
- Charging station locations
- Real-world road network constraints

The system uses Los Angeles road network data to simulate realistic EV routing scenarios.

## 🎯 Key Features

- **Real-world road network modeling** using OpenStreetMap data
- **Battery-aware routing algorithms** with energy consumption simulation
- **Charging station integration** with capacity and availability modeling
- **Multiple algorithmic approaches** including classical and machine learning methods
- **Interactive visualizations** of routes, battery levels, and charging behavior
- **Performance comparison** across different routing strategies

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| OSMnx | Extract and visualize real-world road networks from OpenStreetMap |
| NetworkX | Graph modeling and pathfinding algorithms |
| Matplotlib | Visualization of routes and battery level progression |
| Shapely | Geometric operations for spatial computations |
| Pandas | Handling charging station datasets |
| NumPy | Numerical operations and array management |
| Scikit-learn | Multi-objective optimization using convex hulls |
| PyTorch | Deep Q-Network implementation for reinforcement learning |

## 🏗️ System Architecture

### 1. Graph Construction
- **Road Network Extraction**: 5km radius around Los Angeles city center
- **Charging Station Integration**: Mapping real charging stations to road network nodes
- **Node Selection**: Random start/goal pairs with connectivity verification

### 2. Battery Modeling
- **Energy Consumption**: Linear model based on distance and efficiency coefficient
- **Battery Parameters**: Configurable capacity, initial charge, and consumption rate
- **Charging Behavior**: Full recharge at designated charging stations

### 3. Routing Algorithms

#### Classical Approaches
- **Dijkstra's Algorithm**: Baseline shortest-path without battery constraints
- **Battery-Constrained DFS**: Exhaustive search of all feasible paths
- **Priority Queue Routing**: Battery-aware adaptation of Dijkstra's algorithm
- **Proactive Charging Strategy**: Rule-based approach with intelligent charging decisions

#### Machine Learning Approach
- **Deep Q-Network (DQN)**: Reinforcement learning agent that learns optimal routing policies
- **Custom Environment**: OpenAI Gym-compatible environment for graph navigation
- **State Space**: Current position, battery level, goal, and neighbor information
- **Reward Function**: Optimized for energy efficiency and goal achievement

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/ev-smart-routing.git
cd ev-smart-routing

# Create virtual environment
python -m venv ev_routing_env
source ev_routing_env/bin/activate  # On Windows: ev_routing_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies
```txt
osmnx>=1.2.0
networkx>=2.8
matplotlib>=3.5.0
shapely>=1.8.0
pandas>=1.4.0
numpy>=1.21.0
scikit-learn>=1.1.0
torch>=1.12.0
gym>=0.21.0
jupyter>=1.0.0
```

### Basic Usage

#### 1. Load and Visualize Road Network
```python
import osmnx as ox
import matplotlib.pyplot as plt

# Download Los Angeles road network
G = ox.graph_from_place("Los Angeles, California, USA", dist=5000, network_type='drive')

# Visualize the network
ox.plot_graph(G, figsize=(12, 8))
```

#### 2. Run Battery-Constrained Routing
```python
from routing_algorithms import BatteryConstrainedRouter

# Initialize router with battery parameters
router = BatteryConstrainedRouter(
    graph=G,
    battery_capacity=100,
    energy_efficiency=0.2,
    charging_stations=charging_station_nodes
)

# Find route from start to goal
route, battery_levels = router.find_route(start_node, goal_node)
```

#### 3. Train Reinforcement Learning Agent
```python
from rl_environment import GraphEnv
from dqn_agent import DQNAgent

# Create environment
env = GraphEnv(graph=G, start_node=start, goal_node=goal, 
               battery_capacity=100, charging_nodes=charging_stations)

# Initialize and train DQN agent
agent = DQNAgent(state_size=env.observation_space.shape[0], 
                 action_size=env.action_space.n)

# Training loop
for episode in range(1000):
    state = env.reset()
    # ... training logic
```

## 📊 Results and Performance

### Algorithm Comparison

| Algorithm | Success Rate | Avg. Distance | Computational Complexity | Use Case |
|-----------|-------------|---------------|-------------------------|----------|
| Dijkstra (baseline) | N/A | Shortest | O(E + N log N) | Distance optimization |
| Proactive Charging | 95% | +15% | O(I × S × (E + N log N)) | Real-time routing |
| Battery-Aware Priority | 92% | +8% | O(E + N log N) | Balanced performance |
| DQN Agent | 94% | +12% | O(training episodes) | Adaptive learning |

### Key Findings
- Energy-aware routing requires 8-15% longer distances than shortest-path algorithms
- Proactive charging strategy shows highest success rate for battery-constrained scenarios
- DQN agent demonstrates learning capability with 94% success rate after training
- Computational efficiency varies significantly between approaches

## 📈 Visualizations

The system generates comprehensive visualizations including:
- **Route Maps**: Complete paths with charging stops highlighted
- **Battery Level Curves**: Energy consumption and charging events over time
- **Training Progress**: DQN agent learning curves and performance metrics

## ⚠️ Performance Considerations

### Computational Warnings
- **DFS Algorithm**: Exponential complexity O(2^n) - not recommended for graphs >50 nodes
- **Large Networks**: Consider using simplified graphs or heuristic pruning for city-scale networks
- **Memory Usage**: DFS and extensive path storage can consume significant RAM

### Recommended Usage
- Use proactive charging for real-time applications
- Apply DQN for scenarios requiring adaptive behavior
- Employ simplified networks for algorithm development and testing

## 🔬 Research Applications

This project serves as a foundation for:
- **Transportation Planning**: Urban EV infrastructure development
- **Fleet Management**: Multi-vehicle routing optimization
- **Navigation Systems**: Energy-aware GPS applications
- **Smart Cities**: Integration with dynamic traffic and charging data

## 🤝 Contributing

Contributions are welcome! Areas for improvement include:
- Dynamic traffic integration
- Real-time charging station availability
- Multi-objective optimization techniques
- Fleet coordination algorithms
- Enhanced battery consumption models

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Authors**: Oussema FARHANI, Hedi KSENTINI, Oussema Feki
- **Supervisor**: Mr. Yessine HCHAICHI
- **Institution**: École Polytechnique de Tunisie, University of Carthage
- **Data Sources**: OpenStreetMap, Los Angeles charging station database

## 📚 References

- Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning
- Sachenbacher, M., et al. (2011). Efficient energy-optimal routing for electric vehicles

## 📞 Contact

For questions or collaboration opportunities, feel free to contact me

---

**Project Status**: ✅ Complete | **Last Updated**: 2025 | **Version**: 1.0
