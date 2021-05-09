#include <iostream>
#include <armadillo>

void get_data(int& num_states, int& initial_state, int& final_state, int& num_periods, arma::fmat& mat_transition);

arma::frowvec compute_probability(const int& initial_state, const int& final_state, const int& num_periods, const arma::fmat& mat_transition);

int main(int arg, char* argv[]){
    arma::fmat mat_transition;
    int num_states, initial_state, final_state, num_periods;
    //get_data(num_states, initial_state, final_state, num_periods, mat_transition);

    /* Hard code */
    arma::fmat mat_a = { {0.5, 0.5, 0}, {0.2, 0.5, 0.3}, {0, 0.2, 0.8} };
    std::cout << "Probabilidad es: " << compute_probability(0, 1, 3, mat_a) << std::endl;
    /* End hardcode */
    return 0;
}

void get_data(int& num_states, int& initial_state, int& final_state, int& num_periods, arma::fmat& mat_transition){
    std::cout << "Calculo de probabilidades de primera vez" << std::endl;
    std::cout << "Ingresa el numero de estados: ";
    std::cin >> num_states;
    std::cout << "Ingresa el estado inicial: ";
    std::cin >> initial_state;
    std::cout << "Ingresa el estado final: ";
    std::cin >> final_state;
    std::cout << "Ingresa el numero de periodos: ";
    std::cin >> num_periods;

    mat_transition.zeros(num_states, num_states);
    std::cout << mat_transition << std::endl;

    for(int i = 0; i < static_cast<int>(mat_transition.n_rows); ++i)
        for(int j = 0; j < static_cast<int>(mat_transition.n_cols); ++j){
            std::cout << "Ingresa el elemento [" << i << ", " << j << "]: ";
            std::cin >> mat_transition(i, j);
        }
}

arma::frowvec compute_probability(const int& initial_state, const int& final_state, const int& num_periods, const arma::fmat& mat_transition){

    arma::frowvec probabilities(num_periods);
    probabilities(0) = mat_transition(initial_state, final_state);
    if(num_periods == 1) return probabilities;

    for(int i = 2; i <= num_periods; ++i){
        // P^n
        arma::fmat pow_matrix = arma::powmat(mat_transition, i);

        float sum = 0;
        for(int m = 1; m <= i - 1; ++m){
            // P^(n-m)
            arma::fmat sum_mat = arma::powmat(mat_transition, i - m);
            sum += probabilities(m - 1) * sum_mat(final_state, final_state);
        }

        probabilities(i - 1) = pow_matrix(initial_state, final_state) - sum;
    }

    return probabilities;
}

