#include <vector>
#include <cstdlib>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

struct MarketData {
    std::vector<double> expected_returns; 
    std::vector<std::vector<double>> covariance_matrix; 
    double risk_free_rate;
};

struct Particle {
    std::vector<double> position; // current portfolio weights
    std::vector<double> velocity; // velocity for updating position
    std::vector<double> best_position; // best position found by this particle
    double best_fitness; // best sharpe ratio or other fitness measure found by this particle
    double current_fitness; // current sharpe ratio or other fitness measure

    Particle(int num_assets) {
        position.resize(num_assets);
        velocity.resize(num_assets);
        best_position.resize(num_assets);
        best_fitness = -9999.0; 
        current_fitness = -9999.0;
        
        // Initialize with random positions
        for (int i = 0; i < num_assets; i++) {
            position[i] = (double)rand() / RAND_MAX;
        }
        normalize();
    }

    void update_velocity( std::vector<double>& global_best, double w, double c1, double c2) {
        // loop through each assest
        for (size_t i = 0; i < velocity.size(); i ++) {
            // Generate random numbers between 0.0 and 1.0
            double r1 = (double)rand() / RAND_MAX;
            double r2 = (double)rand() / RAND_MAX;

            velocity[i] = (w * velocity[i]) + (c1 * r1 * (best_position[i] - position[i])) + (c2 * r2 * (global_best[i] - position[i]));
            position[i] += velocity[i];
        }
    }

    void normalize() {
        double sum = 0.0;
        // force - weights to 0 [stock]
        for (size_t i = 0; i < position.size(); i++) {
            if (position[i] < 0.0) { position[i] = 0.0; }
            sum += position[i];
        }

        if (sum > 0.0) {
            for (size_t i = 0; i < position.size(); i++) {
                position[i] = position[i] / sum;
            }
        }
        
    }

    double calculate_fitness(const MarketData& data) {
        double port_return = 0.0;
        double port_variance = 0.0;

        // calculate expected portfolio return
        for (size_t i = 0; i < position.size(); i++) {
            port_return += position[i] * data.expected_returns[i];
            
            // calculate portfolio variance
            for (size_t j = 0; j < position.size(); j++) {
                port_variance += position[i] * position[j] * data.covariance_matrix[i][j];
            }
        }

        double volatility = std::sqrt(port_variance);
        
        // prevent division by zero
        if (volatility == 0.0) return -9999.0; 

        // return sharpe ratio
        return (port_return - data.risk_free_rate) / volatility; 
    }
};

class Swarm {
public:
    std::vector<Particle> particles;
    std::vector<double> global_best_position; // best weight
    double global_best_fitness; // best sharpe ratio

    // particles
    Swarm(int noOfparticles, int noOfassets) {
        global_best_position.resize(noOfassets, 0.0);
        global_best_fitness = -9999.0;

        for (int i = 0; i < noOfparticles; i++ ) {
            Particle particle = Particle(noOfassets);
            particles.push_back(particle);
        }
    }

    void optimize(int epochs, const MarketData& data) {
        for (int i = 0; i < epochs; i++) {
            for (size_t j = 0; j < particles.size(); j++) {
                double fitness = particles[j].calculate_fitness(data);
                
                if (fitness > particles[j].best_fitness) {
                    particles[j].best_fitness = fitness;
                    particles[j].best_position = particles[j].position; 
                }

                if (fitness > global_best_fitness) {
                    global_best_fitness = fitness;
                    global_best_position = particles[j].position;
                }
                
                // Pass global best to velocity, move the particle, and normalize
                particles[j].update_velocity(global_best_position, 0.7, 1.5, 1.5);
                particles[j].normalize();
            }
        }
    }
};

namespace py = pybind11;

PYBIND11_MODULE(pso_engine, m) {
    py::class_<MarketData>(m, "MarketData")
        .def(py::init<>())
        .def_readwrite("expected_returns", &MarketData::expected_returns)
        .def_readwrite("covariance_matrix", &MarketData::covariance_matrix)
        .def_readwrite("risk_free_rate", &MarketData::risk_free_rate);

    py::class_<Swarm>(m, "Swarm")
        .def(py::init<int, int>())
        .def("optimize", &Swarm::optimize)
        .def_readonly("global_best_position", &Swarm::global_best_position)
        .def_readonly("global_best_fitness", &Swarm::global_best_fitness);
}
